# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 00:59:53 2021

"""

# %% TODOs
# TODO: upgrade to parallel data/rollout batch collection
# TODO: upgrade to parallel cost computation
# TODO: upgrade from MPC to PETS
# TODO: upgrade to tasks batch
# TODO: upgrade to learned variance
# TODO: include noise
# TODO: compare NLL loss to MSE loss
# TODO: modularize
# TODO: code clean-up


#%% Imports

#general
import numpy as np

#env
import gym
#------only for spyder IDE
for env in gym.envs.registration.registry.env_specs.copy():
     if 'custom' in env:
         print('Remove {} from registry'.format(env))
         del gym.envs.registration.registry.env_specs[env]
#------
import gym_custom

#visualization
import matplotlib.pyplot as plt
import tqdm

#utils
from scipy.stats import truncnorm
from collections import OrderedDict, namedtuple

#ML
import torch
torch.cuda.empty_cache()
from torch import nn
from torch.distributions import Normal, Independent , MultivariateNormal
from torch.optim import Adam

#%% Functions

#--------
# Utils
#--------

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
        
        
#--------
# Common
#--------

def collect_rollouts(env,controller,model,T,M,loss_func,b):
    #sample a rollout batch from the agent
    
    rollout_batch = []

    for _ in range(b):
        controller.reset() #amounts to resetting CEM optimizer's prev sol to its initial value (--> array of size H with all values = avg of action space value range)
        o=env.reset()
        O, A, rewards, O_dash= [], [], [], []
        for t in range(T):
            
            #adapt parameters to last M timesteps
            if A:
            # if len(A)>=M:
                
                #construct inputs and targets from previous M timesteps
                obs=np.array(O[-M:])
                acs=np.array(A[-M:])
                obs_dash=np.array(O_dash[-M:])
                inputs=np.concatenate([obs, acs], axis=-1)
                targets = obs_dash - obs
                # model.fit_input_stats(inputs)
                inputs = torch.from_numpy(inputs).to(model.device).float()
                targets = torch.from_numpy(targets).to(model.device).float()
                
                #calculate adapted parameters
                mean, dist = model(inputs)
                loss= - (dist.log_prob(targets)).mean(-1).sum() / M if loss_func=="nll" else (torch.square(mean - targets)).mean(-1).sum() / M
                # var=torch.exp(-logvar)
                # loss= - (dist.log_prob(targets)).mean(-1).sum() / M if loss_func=="nll" else (torch.square(mean - targets)*var+logvar).mean(-1).sum()/M
                theta_dash=model.update_params(loss)
            else:
                theta_dash=None
                
            a=controller.act(o,params=theta_dash) #use controller to plan and choose optimal action as first action in sequence
    
            o_dash, r, done, _ = env.step(a) #execute first action from optimal actions
            
            A.append(a)
            O.append(o)
            rewards.append(r)
            O_dash.append(o_dash)
            
            o=o_dash
            
            if done:
                break
        
        rollout_batch.append([np.array(O), np.array(A), np.array(rewards), np.array(O_dash)])

    return rollout_batch

#--------
# Models
#--------

class MLP(nn.Module):
    def __init__(self, in_size, n, h, out_size,alpha,var,device,fixed_var):
        super().__init__()
        
        self.n=n
        self.in_size=in_size
        self.out_size=out_size
        self.alpha=alpha
        self.fixed_var = fixed_var
        
        self.device=device
        
        self.logstd=torch.tensor([var]*self.out_size,dtype=torch.float32, device=device) if self.fixed_var else nn.Parameter(np.log(1.0)*torch.ones(1,out_size,device=device, dtype=torch.float32),requires_grad=True)
        
        # self.max_logvar = nn.Parameter(0.5 * torch.ones(1, out_size // 2, device =self.device, dtype=torch.float32))
        # self.min_logvar = nn.Parameter(-10.0 * torch.ones(1, out_size // 2, device =self.device, dtype=torch.float32))
        
        # self.mu = torch.zeros(1,self.in_size,device=device) # nn.Parameter(torch.zeros(1,self.in_size), requires_grad=False)
        # self.sigma = torch.ones(1,self.in_size,device=device) # nn.Parameter(torch.ones(1,self.in_size), requires_grad=False)
        
        self.layer1=nn.Linear(in_size, h)
        self.layer2=nn.Linear(h, h)
        self.layer3=nn.Linear(h, h)
        self.layer4=nn.Linear(h, out_size)
        
        self.nonlinearity=nn.functional.relu #nn.ReLU() 
        
        self.apply(MLP.initialize)
        self.to(device)
    
    @staticmethod
    def initialize(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias.data)
    
    def fit_input_stats(self, inputs):
        #get mu and sigma of [all] input data for later normalization of model [batch] inputs

        mu = np.mean(inputs, axis=0, keepdims=True) #over cols (each observation/action col of input) and keeping same col size --> result has size = (1,input_size)
        sigma = np.std(inputs, axis=0, keepdims=True)
        sigma[sigma < 1e-12] = 1.0

        self.mu.data = torch.from_numpy(mu).to(self.device).float()
        self.sigma.data = torch.from_numpy(sigma).to(self.device).float()
            
    def update_params(self, loss):
        grads = torch.autograd.grad(loss, self.parameters(), create_graph=False, retain_graph=False) #!!!: create_graph (and thus also retain_graph) is True in case of not first order approximation #???: why?
        #???: if it is false, do we have to compute hessian explicitly via the finite differences?
        new_params = OrderedDict()
        for (name,param), grad in zip(self.named_parameters(), grads):
            new_params[name]= param - self.alpha * grad
        return new_params
    
    def forward(self, inputs, params=None):
        if params is None:
            params=OrderedDict(self.named_parameters())
        
        #normalize inputs
        # inputs = (inputs - self.mu) / self.sigma
 
        inputs=self.nonlinearity(nn.functional.linear(inputs,weight=params['layer1.weight'],bias= params['layer1.bias']))
        inputs=self.nonlinearity(nn.functional.linear(inputs,weight=params['layer2.weight'],bias= params['layer2.bias']))
        inputs=self.nonlinearity(nn.functional.linear(inputs,weight=params['layer3.weight'],bias= params['layer3.bias']))
        mean=nn.functional.linear(inputs,weight=params['layer4.weight'],bias= params['layer4.bias'])
        
        #extract mean and log(var) from network output
        # mean = inputs[ :, :self.out_size // 2]
        # logvar = inputs[ :, self.out_size // 2:]
        
        # #bounding variance (becase network gives arbitrary variance for OOD points --> could lead to numerical problems)
        # logvar = self.max_logvar - nn.functional.softplus(self.max_logvar - logvar)
        # logvar = self.min_logvar + nn.functional.softplus(logvar - self.min_logvar)
        
        # var=torch.exp(-logvar)
        
        std = self.logstd if self.fixed_var else torch.exp(torch.clamp(params['logstd'], min=np.log(1e-6)))
        
        #TIP: MVN=Indep(Normal(),1) --> mainly useful (compared to Normal) for changing the shape og the result of log_prob
        return mean, Normal(mean,std)
        # return mean, logvar, Normal(mean,var)
        # return MultivariateNormal(mean,torch.diag(std[0]))
        # return mean, Independent(Normal(mean,self.var),1)

    
class MPC:
    def __init__(self,env,model,H,pop_size,opt_max_iters):
        self.H=H
        self.pop_size=pop_size
        self.opt_max_iters=opt_max_iters
        self.model=model
        
        self.ds=env.observation_space.shape[0] #state/observation dims
        self.da=env.action_space.shape[0] #action dims
        self.ac_lb= env.action_space.low #env.ac_lb
        self.ac_ub= env.action_space.high #env.ac_ub
        self.cost_obs= env.cost_o
        self.cost_acs= env.cost_a
        self.reset() #sol's initial mu/mean
        self.init_var= np.tile(((self.ac_ub - self.ac_lb) / 4.0)**2, [self.H]) #sol's intial variance

    def reset(self):
        self.prev_sol = np.tile((self.ac_lb + self.ac_ub) / 2.0, [self.H])
    
    def act(self, ob, params=None):
        self.params=params
        sol=self.CEM(ob) #get CEM optimizer's sol
        self.prev_sol=np.concatenate([np.copy(sol)[self.da:], np.zeros(self.da)])
        action = sol[:self.da] #has the first action in the sequence = optimal action
        return action
        
    def CEM(self, ob):
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
            samples = X.rvs(size=[self.pop_size, sol_dim]) * np.sqrt(constrained_var) + mean #sample from destandardized/denormalized distributiion #???: this method of random action seq generation is a bit weird because actions in an action sequence are not really related and does it also assume that all actions are available from all states?
            samples = samples.astype(np.float32)

            #2- propagate state & evaluate actions
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
        
        #reshape ac_seqs --> from (pop_size,sol_dim[H*da]) to (H,pop_size,da)
        ac_seqs = torch.from_numpy(ac_seqs).float().to(self.model.device)
        ac_seqs = ac_seqs.view(-1, self.H, self.da).transpose(0, 1) #transform from (pop_size,sol_dim) to (H,pop_size,da)
        
        #reshape obs --> from (ds) to (pop_size,ds)
        ob = torch.from_numpy(ob).float().to(self.model.device)
        obs = ob[None].expand(self.pop_size, -1)
        
        costs = torch.zeros(self.pop_size, device=self.model.device)
        
        for t in range(self.H):
            curr_acs=ac_seqs[t]
            
            #predict next observations
            inputs=torch.cat([obs, curr_acs], dim=-1)
            mean, dist  = self.model(inputs,self.params)
            # var=torch.exp(logvar)
            delta_obs_next = dist.sample() #mean + torch.randn_like(mean, device=self.model.device) * torch.sqrt(torch.as_tensor(self.model.var)) #* dist.scale #* var.sqrt() #
            obs_next=obs + delta_obs_next
            
            cost=self.cost_obs(obs_next)+self.cost_acs(curr_acs) #evaluate actions
            costs += cost
            obs = obs_next
        
        costs[costs != costs] = 1e6 #replace NaNs with a high cost
        
        
        return costs.detach().cpu().numpy()

# %% Main Function
def main():
    
    #%% Inputs
    #model / policy
    n=3 #no. of NN layers
    h=100 #512 #size of hidden layers
    var=1. #NN variance if using a fixed varaince
    fixed_var=False #whether to use fixed var
    
    #optimizer
    alpha=0.01 #adaptation step size / learning rate
    beta=0.001 #meta step size / learning rate
    
    #general
    # trials=30 #no. of trials
    tr_eps=30 #50 #no. of training episodes/iterations
    # eval_eps=1
    log_ival=1 #logging interval
    b=4 #32 #batch size: Number of rollouts to sample from each task
    # meta_b=16 #32 #number of tasks sampled
    seed=0
    
    #controller
    H=8 #planning horizon
    # epochs=5 #propagation method epochs
    pop_size=60 #1000 #CEM population size: number of candidate solutions to be sampled every iteration 
    opt_max_iters=5 #5 #CEM's max iterations (used as a termination condition)
    
    #algorithm
    M=16 #no. of prev timesteps
    K=M #no. of future timesteps (adaptation horizon)
    N=16 #no. of sampled tasks (fluid definition)
    ns=5 #3 #10 #task sampling frequency
    loss_funcs=["nll","mse"] #nll: negative log loss; mse: mean squared error
    loss_func= loss_funcs[0]
    traj_b=1 #4 #trajectory batch size (no. of trajectories sampled per sampled task)
    # gamma= 1. #discount factor
    
    #%% Initializations
    #common
    D=[] #dataset / experience (a list of rollouts [where each rollout is a list of arrays (s,a,r,s')])
    
    #environment
    env_names=['cartpole_custom-v1','halfcheetah_custom-v1']
    env_name=env_names[1]
    env=gym.make(env_name)
    task_name="cripple"
    T=env._max_episode_steps #task horizon
    ds=env.observation_space.shape[0] #state dims
    da=env.action_space.shape[0] #action dims
    
    #models
    in_size=ds+da
    out_size=ds#*2
    device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = MLP(in_size,n,h,out_size,alpha,var,device,fixed_var).to(device) #dynamics model
    controller = MPC(env,model,H,pop_size,opt_max_iters)
    optimizer = Adam(model.parameters(),lr=beta) #model optimizer
    
    #results 
    plot_tr_rewards=[]
    best_reward=-1e6
    
    set_seed(seed,env)
    
    #%% Sanity Checks
    assert H<K, "planning horizon H should be smaller than the adaptation horizon K, since the adapted model is only valid within the current context"
    assert T>=M+K, "rollout length (task horizon / max no. of env timesteps) T has to at least be equal to the prev + future adaptation timesteps"
    assert H<=T, "planning horizon must be at most the total no. of environment timesteps"
    
    #%% Algorithm
    
    episodes=progress(tr_eps)
    for episode in episodes:
        
        if episode==0 or episode % ns == 0:
            task = env.sample_task()
            env.reset_task(task,task_name)
            rollout_batch=collect_rollouts(env, controller, model, T,M,loss_func,b)
            reward_ep=np.mean([np.sum(rollout[2]) for rollout in rollout_batch])
            D.extend(rollout_batch)
        
        #adaptation
        losses=[]
        for i in range(N):
            
            #sample [a batch of] trajectories randomly from the dataset
            m_trajs, k_trajs = [], []
            for _ in range(traj_b):
                rollout = D[np.random.choice(len(D))] #pick a rollout at random
                m_start_idx = np.random.choice(len(rollout[0]) + 1 - M - K)
                m_traj=[r[m_start_idx: m_start_idx + M] for r in rollout]
                k_traj=[r[m_start_idx + M : m_start_idx + M + K] for r in rollout]
                m_trajs = m_traj if m_trajs==[] else [np.concatenate((m_trajs[dim], m_traj[dim])) for dim in range(len(m_traj))]
                k_trajs = k_traj if k_trajs==[] else [np.concatenate((k_trajs[dim], k_traj[dim])) for dim in range(len(k_traj))]
            
            #adapt params
            #construct inputs and targets from m_trajs
            obs=m_trajs[0]
            acs=m_trajs[1]
            obs_dash=m_trajs[-1]
            inputs=np.concatenate([obs, acs], axis=-1)
            targets = obs_dash - obs
            # model.fit_input_stats(inputs)
            inputs = torch.from_numpy(inputs).to(model.device).float()
            targets = torch.from_numpy(targets).to(model.device).float()
            #compute adapted parameters
            mean, dist  = model(inputs)
            loss= - (dist.log_prob(targets)).mean(-1).sum() / M if loss_func=="nll" else (torch.square(mean - targets)).mean(-1).sum() / M
            # var = torch.exp(-logvar)
            # loss= - (dist.log_prob(targets)).mean(-1).sum() / M if loss_func=="nll" else (torch.square(mean - targets)*var+logvar).mean(-1).sum()/M
            theta_dash = model.update_params(loss)
            
            #compute loss
            #construct inputs and targets from k_trajs
            obs=k_trajs[0]
            acs=k_trajs[1]
            obs_dash=k_trajs[-1]
            inputs=np.concatenate([obs, acs], axis=-1)
            targets = obs_dash - obs
            # model.fit_input_stats(inputs)
            inputs = torch.from_numpy(inputs).to(model.device).float()
            targets = torch.from_numpy(targets).to(model.device).float()
            #compute task loss
            mean, dist  = model(inputs,params=theta_dash)
            # var = torch.exp(-logvar)
            loss= - (dist.log_prob(targets)).mean(-1).sum() / K if loss_func=="nll" else (torch.square(mean - targets)).mean(-1).sum() / K
            # loss= - (dist.log_prob(targets)).mean(-1).sum() / K if loss_func=="nll" else (torch.square(mean - targets)*var+logvar).mean(-1).sum()/K
            losses.append(loss)
            
        #meta update
        meta_loss=torch.mean(torch.stack(losses))
        optimizer.zero_grad()
        meta_loss.backward(create_graph=False,retain_graph=False) #???: should any of them be True?
        optimizer.step()
        
        #save best running model [params]
        if reward_ep>best_reward: 
            best_reward=reward_ep
            torch.save(model.state_dict(), "saved_models/"+env_name+".pt")
            
        #log iteration results & statistics
        plot_tr_rewards.append(reward_ep)
        if episode % log_ival == 0:
            log_msg="Rewards Tr: {:.2f}".format(reward_ep)
            episodes.set_description(desc=log_msg); episodes.refresh()
        
    #%% Results & Plot
    title="Meta-Training Rewards (Learning Curve)"
    plt.figure(figsize=(16,8))
    plt.grid(1)
    plt.plot(plot_tr_rewards)
    plt.title(title)
    plt.show()

#%%
if __name__ == '__main__':
    main()