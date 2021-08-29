# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 16:56:16 2021

@author: TIB001
"""
#TODO: learn how to interface with and modify mujoco envs (--> interface with a custom mujoco env)
#TODO: try pets on half-cheetah
#TODO: try with and report results for different reward functions and agent parameters (custom cartpole and half-cheetah)
#TODO: repeat each experiment with different random seeds (for K trials?), and report the mean and standard deviation of the cost for each condition
#TODO: add more comments/documentation

# %% Imports
import numpy as np
import gym
#------only for spyder IDE
for env in gym.envs.registration.registry.env_specs.copy():
     if 'custom' in env:
         print('Remove {} from registry'.format(env))
         del gym.envs.registration.registry.env_specs[env]
#------
import gym_custom
import matplotlib.pyplot as plt
import tqdm
from scipy.stats import truncnorm

import torch
from torch import nn
from torch.optim import Adam



#%% Functions

progress=lambda x: tqdm.trange(x, leave=True) #for visualizing/monitoring training progress

def set_seed(seed,env,det=True):
    import random
    # import os
    # os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    # torch.use_deterministic_algorithms(det)
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    env.seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def truncated_normal(tensor: torch.Tensor, mean: float = 0, std: float = 1,device="cpu"):
    torch.nn.init.normal_(tensor, mean=mean, std=std)
    while True:
        cond = torch.logical_or(tensor < mean - 2.0 * std, tensor > mean + 2.0 * std)
        if not torch.sum(cond):
            break
        tensor = torch.where(cond,torch.nn.init.normal_(torch.ones(tensor.shape, device=device), mean=mean, std=std),tensor,)
    return tensor


def collect_rollout(env,policy):
    #sample a rollout from the agent [Rollout length= len(A)]

    T=env._max_episode_steps #task horizon
    O, A, rewards= [env.reset()], [], []

    policy.reset() #policy is MPC #amounts to resetting CEM optimizer's prev sol to its initial value (--> array of size H with all values = avg of action space value range)
    for t in range(T):
        a=policy.act(O[t]) #first optimal action in sequence (initially random)

        o, r, done, _ = env.step(a) #execute first action from optimal actions
        
        A.append(a)
        O.append(o)
        rewards.append(r)
        
        if done:
            break
    
    return np.array(O), np.array(A), np.array(rewards), sum(rewards)


#model
class PE(nn.Module):
    #Probabilistic Ensemble: ensemble of B-many bootsrapped probabilistic NNs (i.e. output parameters of a prob distribution) used to approximate a dynamics model function:
    # a- ‘probabilistic networks[/network models/network dynamics models]’: to capture aleatoric uncertainty (inherent system stochasticity) [through each model encoding a distribution (as opposed to a point estimate/prediction)]
    # b- ‘ensembles’: to capture epistemic uncertainty (subjective uncertainty, due to limited data --> isolating it is especially useful for directing exploration [out of scope])
    
    def __init__(self, B, in_size, n, h, out_size,device):
        super().__init__()
        
        self.B=B
        self.n=n
        self.in_size=in_size
        self.out_size=out_size
        self.device=device
        
        self.max_logvar = nn.Parameter(0.5 * torch.ones(1, out_size // 2, device =self.device, dtype=torch.float32))
        self.min_logvar = nn.Parameter(-10.0 * torch.ones(1, out_size // 2, device =self.device, dtype=torch.float32))
        
        self.mu = nn.Parameter(torch.zeros(self.in_size), requires_grad=False)
        self.sigma = nn.Parameter(torch.ones(self.in_size), requires_grad=False)
        
        self.w0,self.b0=self.initialize(in_size,h)
        self.w1,self.b1=self.initialize(h,h)
        self.w2,self.b2=self.initialize(h,h)
        self.w3,self.b3=self.initialize(h,out_size)
        
    def initialize(self,in_size,out_size):
        #truncated normal for weights (i.e. draw from normal distribution with samples outside 2*std from mean discarded and resampled)
        #zeros for biases
        
        mu=0.0
        std=1.0/(2.0*np.sqrt(in_size))
        
        # w = nn.Parameter(torch.rand(self.B,in_size,out_size,device=self.device,dtype=torch.float32))
        # w.data = truncated_normal(w.data, mean=mu, std=std, device=self.device)
        # b=nn.Parameter(torch.rand(self.B, 1, out_size,device=self.device,dtype=torch.float32))
        # b.data.fill_(0.0)
        
        w = truncnorm.rvs(mu-2.0*std,mu+2.0*std,loc=mu,scale=std,size=(self.B,in_size,out_size))
        w = nn.Parameter(torch.tensor(w,device=self.device,dtype=torch.float32))
        b = nn.Parameter(torch.zeros(self.B,1,out_size,device=self.device,dtype=torch.float32))
        # torch.nn.init.zeros_(b.data)
        
        return w, b
    
    def fit_input_stats(self, inputs):
        #get mu and sigma of [all] input data fpr later normalization of model [batch] inputs

        mu = np.mean(inputs, axis=0, keepdims=True) #over cols (each observation/action col of input) and keeping same col size --> result has size = (1,input_size)
        sigma = np.std(inputs, axis=0, keepdims=True)
        sigma[sigma < 1e-12] = 1.0 #???: why 1 (and not 1e-12 e.g.)??

        self.mu.data = torch.from_numpy(mu).to(self.device).float()
        self.sigma.data = torch.from_numpy(sigma).to(self.device).float()
    
    def compute_decays(self):
        #returns decays.sum() #decays[layer] = decay_coeffs[layer]*MSE(w[layer]) #decay_coeffs=[input=0.0001, h=0.00025, output=0.0005]
        
        lin0_decays = 0.0001 * (self.w0 ** 2).sum() / 2.0
        lin1_decays = 0.00025 * (self.w1 ** 2).sum() / 2.0
        lin2_decays = 0.00025 * (self.w2 ** 2).sum() / 2.0
        lin3_decays = 0.0005 * (self.w3 ** 2).sum() / 2.0

        return lin0_decays + lin1_decays + lin2_decays + lin3_decays

    def forward(self,inputs):
        # input is 3D: [B,batch_size,input_size] --> input is a function of current observation/state
        # output is 3D: [B, batch_size, output_size] --> output size is obs/target_size * 2 (first half is for expectation/mu/mean of [Delta_]obs distribution and second half is for log(variance) of it) --> ∆s_t+1 = f (s_t; a_t) such that s_t+1 = s_t + ∆s_t+1
        
        #normalize inputs
        inputs = (inputs - self.mu) / self.sigma
        
        #fwd pass
        inputs = inputs.matmul(self.w0) + self.b0
        inputs = nn.SiLU()(inputs) 
        inputs = inputs.matmul(self.w1) + self.b1
        inputs = nn.SiLU()(inputs) 
        inputs = inputs.matmul(self.w2) + self.b2
        inputs = nn.SiLU()(inputs)
        inputs = inputs.matmul(self.w3) + self.b3
        
        #extract mean and log(var) from network output
        mean = inputs[:, :, :self.out_size // 2]
        logvar = inputs[:, :, self.out_size // 2:]
        
        #bounding variance (becase network gives arbitrary variance for OOD points --> could lead to numerical problems)
        logvar = self.max_logvar - nn.functional.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + nn.functional.softplus(logvar - self.min_logvar)
        
        return mean, logvar


#planner      
class MPC:
    def __init__(self,env,model,optimizer,H,p,pop_size,opt_max_iters,epochs):
        
        self.H=H
        self.p=p
        self.pop_size=pop_size
        self.opt_max_iters=opt_max_iters
        self.model=model
        self.epochs=epochs
        self.optimizer=optimizer
        
        self.ds=env.observation_space.shape[0] #state/observation dims
        self.da=env.action_space.shape[0] #action dims
        self.initial=True
        self.ac_lb= env.action_space.low #env.ac_lb
        self.ac_ub= env.action_space.high #env.ac_ub
        self.cost_obs= env.cost_o
        self.cost_acs= env.cost_a
        self.reset() #sol's initial mu/mean
        self.init_var= np.tile(((self.ac_ub - self.ac_lb) / 4.0)**2, [self.H]) #sol's intial variance
        self.ac_buff=np.empty((0,self.da))
        self.inputs=np.empty((0,self.model.in_size))
        self.targets=np.empty((0,self.ds))
        
        
    def reset(self):
        self.prev_sol = np.tile((self.ac_lb + self.ac_ub) / 2.0, [self.H])
    
    
    def train(self,rollout,b): #Train the policy with rollouts
        
        obs = rollout[0]
        acs = rollout[1]
        self.initial=False
        
        #1- Construct model training inputs
        inputs=np.concatenate([obs[:-1], acs], axis=-1)
        self.inputs=np.concatenate([self.inputs,inputs])
        
        #2- Construct model training targets
        targets = obs[1:] - obs[:-1]
        self.targets=np.concatenate([self.targets,targets])
        
        #3- Train the model
            
        #get mean & var of tr inputs
        self.model.fit_input_stats(self.inputs)
        #create random_idxs from 0 to (no. of tr samples - 1) with size = [B,no. of tr samples]
        idxs = np.random.randint(self.inputs.shape[0], size=[self.model.B, self.inputs.shape[0]])
        num_batches = int(np.ceil(idxs.shape[-1] / b)) #(no. of batches=roundup(no. of [model] training input examples so far / batch size))
        for _ in range(epochs): #for each epoch
            for batch_num in range(num_batches): # for each batch 
                # choose a batch from tr and target inputs randomly (i.e. pick random entries/rows/samples from inputs to construct a batch, via: input[random_idxs[:,batch_idxs]]) --> batch_idxs change with each inner iteration as a function of current batch no. and b, while rest stay constant; this also inserts an additional dimension at the beginning to inputs = B; random_idxs used for each net are shuffled row-wise with each outer loop; idxs are reset w/ every call to train func [i.e. each tr_ep] (which would also have different no. of tr samples)
                batch_idxs = idxs[:, batch_num * b : (batch_num + 1) * b]
                inputs = torch.from_numpy(self.inputs[batch_idxs]).to(self.model.device).float()
                targets = torch.from_numpy(self.targets[batch_idxs]).to(self.model.device).float()
                # Operate on batches:
                mean, logvar = self.model(inputs) #fwd pass
                var = torch.exp(-logvar)
                # Calculate grad, loss & backpropagate
                loss = (torch.square(mean - targets) * var + logvar).mean(-1).mean(-1).sum() #train losses: MSE loss + var loss #???: why does mean over target dimension make sense?
                loss += 0.01 * (self.model.max_logvar.sum() - self.model.min_logvar.sum()) # a constant (~= 0.42)
                # loss += self.model.compute_decays() #L2 regularization
                # loss_value=loss.item()
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
            # shuffle idxs
            idxs_of_idxs = np.argsort(np.random.uniform(size=idxs.shape), axis=-1)
            idxs = idxs[np.arange(idxs.shape[0])[:, None], idxs_of_idxs] #shuffles indicies of each row of idxs randomly (i.e. acc. to idxs_of_idxs)
        
    
    def act(self,ob):
        
        if self.initial:
            action = np.random.uniform(self.ac_lb, self.ac_ub, self.ac_lb.shape)
            
        else:
            while self.ac_buff.shape[0] == 0:
                sol=self.CEM(ob) #get CEM optimizer's sol
                self.prev_sol=np.concatenate([np.copy(sol)[self.da:], np.zeros(self.da)]) #update prev sol --> take out first action in sol and pad leftover sequence with trailing zeros to maintain same sol/prev sol shape #???: how is this correct??
                self.ac_buff = sol[:self.da].reshape(-1, self.da) #has the first action in the sequence = optimal action
            action, self.ac_buff = self.ac_buff[0], self.ac_buff[1:] #pop out the optimal action from buffer
        
        return action
        
    
    def CEM(self,ob): #MPC's optimizer: an action sequence optimizer
        #solution = action sequence = sample
        #population-based method (samples candidate solutions from a distribution then evaluates them and constructs next distribution with best sols from prev one [acc to the defined cost function])
        
        #action sequence is taken to be optimized once per iteration (i.e. per=1)
        sol_dim=self.H*self.da #dimension of an action sequence
        epsilon=0.001 #termination condition representing min variance allowable
        alpha=0.25 #0.1 #controls how much of mean & variance is used for next iteration
        lb=np.tile(self.ac_lb,[self.H])
        ub=np.tile(self.ac_ub,[self.H])
        
        mean, var, t = self.prev_sol, self.init_var, 0.0
        X = truncnorm(-2, 2, loc=np.zeros_like(mean), scale=np.ones_like(mean))

        while (t < self.opt_max_iters) and np.max(var) > epsilon:
            lb_dist = mean - lb
            ub_dist = ub - mean
            constrained_var = np.minimum(np.minimum((lb_dist / 2.0)**2, (ub_dist / 2.0)**2), var)
            
            #1- generate sols
            samples = X.rvs(size=[self.pop_size, sol_dim]) * np.sqrt(constrained_var) + mean #sample from destandardized/denormalized distributiion
            samples = samples.astype(np.float32)

            #2- propagate state particles & evaluate actions
            costs = self.get_costs(ob,samples)

            #3- update CEM distribution
            
            elites = samples[np.argsort(costs)][:self.pop_size//10] #first num_elites (here: =10% of population, for simplicity) items of samples after sorting them according to their costs (in ascending order) --> i.e. num_elites samples with lowest costs according to defined cost func
            # num_elites = CEM's number of top/best solutions (out of popsize of candidate sols) thet will be used to update the CEM distrib (i.e. obtain the distribution at the next iteration)
            
            #update distribution params
            mean_elites = np.mean(elites, axis=0)
            var_elites = np.var(elites, axis=0)
            #get portion of mu/mean and sigma/var used for next iteration 
            mean = alpha * mean + (1 - alpha) * mean_elites
            var = alpha * var + (1 - alpha) * var_elites

            t += 1

        return mean
    
    @torch.no_grad()
    def get_costs(self,ob,ac_seqs):
        
        dim=ac_seqs.shape[0]
        
        #reshape ac_seqs --> from (pop_size,sol_dim[H*da]) to (H,pop_size*p,da) --> pop_size*p = expand each candidate sol to a set of p particles
        ac_seqs = torch.from_numpy(ac_seqs).float().to(self.model.device)
        ac_seqs = ac_seqs.view(-1, self.H, self.da).transpose(0, 1)[:, :, None] #transform from (pop_size,sol_dim) to (H,pop_size,da,1)
        ac_seqs = ac_seqs.expand(-1, -1, self.p, -1).contiguous().view(self.H, -1, self.da) #transform from (H,pop_size,da,1) to (H,pop_size*p,da)
        
        #reshape obs --> from (ds) to (pop_size*p,ds)
        ob = torch.from_numpy(ob).float().to(self.model.device)
        obs = ob[None].expand(dim * self.p, -1)

        costs = torch.zeros(dim, self.p, device=self.model.device) #initialize costs ; size=[pop_size,p] #sum of costs over the planning horizon
        
        for t in range(self.H):
            curr_acs=ac_seqs[t]
            obs_next = self.TS(obs,curr_acs) #propagate state particles using the PE dynamics model (aka: predict next observations)
            
            cost=self.cost_obs(obs_next)+self.cost_acs(curr_acs) #evaluate actions
            cost = cost.view(-1, self.p) #reshape costs to (pop_size,p)
            costs += cost
            obs = obs_next
        
        costs[costs != costs] = 1e6 #replace NaNs with a high cost
        
        
        return costs.mean(dim=1).detach().cpu().numpy() #mean of costs # dim is dim to reduce (i.e. for dim=1, it will take the mean of each row (dim=0)) #i.e. here we average over the particles of each sol
    
    
    def TS(self,obs,curr_acs): #trajectory sampling: propagate state particles (aka: predict next observations)
        #implements TSinf
        
        #reshape obs  #(pop_size*p,ds) --> (B,pop_size*p/B,ds) #i.e. divide the pop_size*p observations among B networks
        dim=obs.shape[-1]
        obs_reshaped=obs.view(-1, self.model.B, self.p // self.model.B, dim).transpose(0, 1) #(pop_size*p,ds) --> (B,pop_size,p/B,ds)
        obs_reshaped=obs_reshaped.contiguous().view(self.model.B, -1, dim) #(B,pop_size,p/B,ds) --> (B,pop_size*p/B,ds)
        #reshape curr_acs  #(1,pop_size*p,da) --> (B,pop_size*p/B,da) #i.e. divide the pop_size*p curr_acs among B networks
        dim=curr_acs.shape[-1]
        curr_acs=curr_acs.view(-1, self.model.B, self.p // self.model.B, dim).transpose(0, 1)
        curr_acs=curr_acs.contiguous().view(self.model.B, -1, dim)
        
        inputs=torch.cat([obs_reshaped, curr_acs], dim=-1)
        mean, logvar = self.model(inputs) #here, input smaples will be = pop_size*p/B #???: does it make sense how inputs are normalized in this call??
        var = torch.exp(logvar)
        delta_obs_next=mean + torch.randn_like(mean, device=self.model.device) * var.sqrt() #var.sqrt() = std
        #reshape delta_obs_next/predictions [back to original shape]  #(B,pop_size,p/B,ds) --> (pop_size*p,ds)
        dim=delta_obs_next.shape[-1]
        delta_obs_next=delta_obs_next.view(self.model.B, -1, self.p // self.model.B, dim).transpose(0, 1)
        delta_obs_next=delta_obs_next.contiguous().view(-1,dim)
        
        obs_next=delta_obs_next+obs
        return obs_next

# %% Inputs
p=20 #no. of particles
B=5 #no. of bootstraps (nets in ensemble)
K=50 #no. of trials
tr_eps=30 #30 #200 #no. of training episodes/iterations
te_eps=10 #testing episodes
test=False
log_ival=1 #tr logging interval
n=3 #no. of NN layers
h=250 #500 #250 #size of hidden layers
H=12 #25 #planning horizon
# r=1 #no. of rollouts done in the environment for every training iteration AND no. of initial rollouts done before first train() call to controller #TODO: code is currently written for r=1; make code general to any r
epochs=5 #5 #100 #propagation method epochs
lr=0.001
b=32 #1 #batch size
pop_size=60 #400 #CEM population size: number of candidate solutions to be sampled every iteration 
opt_max_iters=5 #5 #CEM's max iterations (used as a termination condition)

# %% Environment
# env=gym.make('cartpole_custom-v1')
env = gym.make('halfcheetah_custom-v1')
ds=env.observation_space.shape[0] #state dims
da=env.action_space.shape[0] #action dims
set_seed(0,env)

# %% Initializations
in_size=ds+da
out_size=ds*2
device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = PE(B,in_size,n,h,out_size,device).to(device) #dynamics model
optimizer=Adam(model.parameters(),lr=lr) #model optimizer #TODO: use per-parameter options (and adjust model weights and biases into layers) to add different weight decays to each layer's weights
policy = MPC(env,model,optimizer,H,p,pop_size,opt_max_iters,epochs)
plot_rewards=[]

# %% Implementation

#1- Initialize data D with a random controller for one/r trial(s): sample an initial rollout from the agent with random policy
rollout = collect_rollout(env,policy)

# 2- Training
episodes=progress(tr_eps)
for episode in episodes:
    #train policy with [prev] rollout
    policy.train(rollout,b)
    #sample a [new] rollout from the agent with MPC policy
    rollout = collect_rollout(env,policy)
    #log iteration results & statistics
    plot_rewards.append(rollout[-1])
    if episode % log_ival == 0:
        log_msg="Rewards Sum: {:.2f}".format(rollout[-1])
        episodes.set_description(desc=log_msg); episodes.refresh()

# 3- Results & Plot
title="Training Rewards (Learning Curve)"
plt.figure(figsize=(16,8))
plt.grid(1)
plt.plot(plot_rewards)
plt.title(title)
plt.show()

# %% Testing
if test:
    ep_rewards=[]
    for _ in range(te_eps):
        o=env.reset() #observation
        O, A, rewards, done= [o], [], [], False
        policy.reset()
        while not done:
            a=policy.act(o) #first optimal action in sequence (initially random)
    
            o, r, done, _ = env.step(a) #execute first action from optimal actions
            
            A.append(a)
            O.append(o)
            rewards.append(r)
            
            env.render()
        
        ep_rewards.append(np.sum(rewards))
        
# Results & Plots
    title="Testing Control Actions"
    plt.figure(figsize=(16,8))
    plt.grid(1)
    plt.plot(A)
    plt.title(title)
    plt.show()
        
    title="Testing Rewards"
    plt.figure(figsize=(16,8))
    plt.grid(1)
    plt.plot(ep_rewards)
    plt.title(title)
    plt.show()
    
    env.close()
    
