# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 00:59:53 2021

"""

# %% TODOs
# TODO: upgrade to parallel data/rollout batch collection
# TODO: upgrade to parallel cost computation (create a vectorized controller abstraction like the env)
# TODO: upgrade from MPC to PETS
# TODO: upgrade to tasks batch
# TODO: include noise
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

def collect_rollouts(env,controller,model,T,b,params=None):
    #sample a rollout batch from the agent
    Rollout = namedtuple('Rollout', ('states', 'actions', 'rewards','states_dash'))
    rollout_batch = []

    for _ in range(b):
        controller.reset() #amounts to resetting CEM optimizer's prev sol to its initial value (--> array of size H with all values = avg of action space value range)
        o=env.reset()
        rollout = []
        for t in range(T):
            a=controller.act(o,params=params) #use controller to plan and choose optimal action as first action in sequence
            o_dash, r, done, _ = env.step(a) #execute first action from optimal actions
            rollout.append((o,a,r,o_dash))
            
            o=o_dash
            
            if done:
                break
        
        # put rollout into batch
        states, actions, rewards, states_dash = zip(*rollout)
        states = torch.as_tensor(states) 
        actions = torch.as_tensor(actions)
        rewards = torch.as_tensor(rewards).unsqueeze(1)
        states_dash = torch.as_tensor(states_dash)
        
        rollout_batch.append(Rollout(states, actions, rewards, states_dash))
            
    #unpack rollouts' states, actions and rewards into dataset D
    states, actions, rewards, states_dash = zip(*rollout_batch)
    states=torch.cat(states,dim=1).view(T,b,-1).float().to(model.device)
    actions=torch.cat(actions,dim=1).view(T,b,-1).float().to(model.device)
    rewards=torch.cat(rewards,dim=1).view(T,b,-1).float().to(model.device)
    states_dash=torch.cat(states_dash,dim=1).view(T,b,-1).float().to(model.device)
    
    rollout_batch=[states, actions, rewards, states_dash]

    return rollout_batch


def compute_loss(loss_func,mean,targets,logvar,dist,var_type,model):
    if loss_func=="nll":
        loss= - dist.log_prob(targets)
    else: #mse
        if var_type=="out":
            var=torch.exp(-logvar)
            loss= torch.square(mean - targets)*var+logvar
        else:
            loss= torch.square(mean - targets)*dist.scale-torch.log(dist.scale)
    loss=loss.sum(-1).mean() #.mean()
    if var_type=="out":
        loss += 0.01 * (model.max_logvar.sum() - model.min_logvar.sum())
    return loss
    

#--------
# Models
#--------

class MLP(nn.Module):
    def __init__(self, in_size, n, h, out_size,alpha,var,device,var_type):
        super().__init__()
        
        self.n=n
        self.in_size=in_size
        self.out_size=out_size
        self.alpha=alpha
        self.var_type = var_type
        
        self.device=device
        
        self.logstd=nn.Parameter(np.log(1.0)*torch.ones(1,out_size,device=device, dtype=torch.float32),requires_grad=True) if var_type=="param" else torch.tensor([np.log(var)]*self.out_size,dtype=torch.float32, device=device)
        
        if var_type=="out":
            self.max_logvar = nn.Parameter(0.5 * torch.ones(1, out_size // 2, device =self.device, dtype=torch.float32))
            self.min_logvar = nn.Parameter(-10.0 * torch.ones(1, out_size // 2, device =self.device, dtype=torch.float32))
        
        self.mu = torch.zeros(1,self.in_size,device=device) # nn.Parameter(torch.zeros(1,self.in_size), requires_grad=False)
        self.sigma = torch.ones(1,self.in_size,device=device) # nn.Parameter(torch.ones(1,self.in_size), requires_grad=False)
        
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
            
    def update_params(self, loss, retain_graph=False):
        grads = torch.autograd.grad(loss, self.parameters(), create_graph=False, retain_graph=retain_graph) #???: should any of them be True?
        #!!!: create_graph (and thus also retain_graph) is True in case of not first order approximation #???: why?
        #???: if it is false, do we have to compute hessian explicitly via the finite differences?
        new_params = OrderedDict()
        for (name,param), grad in zip(self.named_parameters(), grads):
            new_params[name]= param - self.alpha * grad
        return new_params
    
    def forward(self, inputs, params=None, normalize=False):
        if params is None:
            params=OrderedDict(self.named_parameters())
        
        #normalize inputs
        if normalize: inputs = (inputs - self.mu) / self.sigma
 
        inputs=self.nonlinearity(nn.functional.linear(inputs,weight=params['layer1.weight'],bias= params['layer1.bias']))
        inputs=self.nonlinearity(nn.functional.linear(inputs,weight=params['layer2.weight'],bias= params['layer2.bias']))
        inputs=self.nonlinearity(nn.functional.linear(inputs,weight=params['layer3.weight'],bias= params['layer3.bias']))
        inputs=nn.functional.linear(inputs,weight=params['layer4.weight'],bias= params['layer4.bias'])
        
        if self.var_type == "out":
            # extract mean and log(var) from network output
            mean = inputs[ ..., :self.out_size // 2]
            logvar = inputs[ ..., self.out_size // 2:]
        
            # #bounding variance (becase network gives arbitrary variance for OOD points --> could lead to numerical problems)
            logvar = params['max_logvar'] - nn.functional.softplus(params['max_logvar'] - logvar) # self.max_logvar - nn.functional.softplus(self.max_logvar - logvar)
            logvar = params['min_logvar'] + nn.functional.softplus(logvar - params['min_logvar']) # self.min_logvar + nn.functional.softplus(logvar - self.min_logvar)
        
            std=torch.exp(-logvar) #torch.exp(logvar)
            
        else:
            mean=inputs
            std = torch.exp(torch.clamp(params['logstd'], min=np.log(1e-6))) if self.var_type=="param" else torch.exp(self.logstd)
            logvar=torch.clamp(params['logstd'], min=np.log(1e-6)) if self.var_type=="param" else self.logstd
        
        #TIP: MVN=Indep(Normal(),1) --> mainly useful (compared to Normal) for changing the shape og the result of log_prob
        return mean, logvar, Normal(mean,std)
        # return mean, logvar, Normal(mean,var)
        # return MultivariateNormal(mean,torch.diag(std[0]))
        # return mean, Independent(Normal(mean,self.var),1)

    
class MPC:
    def __init__(self,env,model,H,pop_size,opt_max_iters,optimizer,epochs,batch_size,b):
        self.H=H
        self.pop_size=pop_size
        self.opt_max_iters=opt_max_iters
        self.optimizer=optimizer
        self.epochs=epochs
        self.batch_size=batch_size
        self.model=model
        
        self.ds=env.observation_space.shape[0] #state/observation dims
        self.da=env.action_space.shape[0] #action dims
        self.ac_lb= env.action_space.low #env.ac_lb
        self.ac_ub= env.action_space.high #env.ac_ub
        self.cost_obs= env.cost_o
        self.cost_acs= env.cost_a
        self.reset() #sol's initial mu/mean
        self.initial=True
        self.init_var= np.tile(((self.ac_ub - self.ac_lb) / 4.0)**2, [self.H]) #sol's intial variance
        self.inputs=np.empty((0,b,self.model.in_size))
        self.targets=np.empty((0,b,self.model.out_size // 2)) if model.var_type=="out" else np.empty((0,self.model.out_size))
        # self.inputs=torch.empty((0,b,self.model.in_size),device=model.device)
        # self.targets=torch.empty((0,b,self.model.out_size // 2),device=model.device) if model.var_type=="out" else torch.empty((0,b,self.model.out_size),device=model.device)
        # self.inputs_dash=torch.empty((0,b,self.model.in_size),device=model.device)
        # self.targets_dash=torch.empty((0,b,self.model.out_size // 2),device=model.device) if model.var_type=="out" else torch.empty((0,b,self.model.out_size),device=model.device)

    def reset(self):
        self.prev_sol = np.tile((self.ac_lb + self.ac_ub) / 2.0, [self.H])
    
    def train(self, Ds, D_dashes, loss_func, normalize=False): #Train the policy with rollouts
        
        self.initial=False
        
        obs = [D[0] for D in Ds] 
        acs = [D[1] for D in Ds] 
        obs_dash= [D[-1] for D in Ds]
        
        #1- Construct model training inputs & targets
        #D
        inputs, targets=[], []
        for ob, ac, ob_dash in zip(obs,acs,obs_dash):
            inputs.append(torch.cat([ob, ac],-1))
            targets.append(ob_dash-ob)

        self.inputs=torch.cat([self.inputs]+inputs)
        self.targets=torch.cat([self.targets]+targets)
        # self.inputs=torch.cat(inputs)
        # self.targets=torch.cat(targets)
        
        obs = [D_dash[0] for D_dash in D_dashes] 
        acs = [D_dash[1] for D_dash in D_dashes] 
        obs_dash= [D_dash[-1] for D_dash in D_dashes]
        
        #D_dash
        inputs, targets=[], []
        for ob, ac, ob_dash in zip(obs,acs,obs_dash):
            inputs.append(torch.cat([ob, ac],-1))
            targets.append(ob_dash-ob)

        self.inputs_dash=torch.cat([self.inputs_dash]+inputs)
        self.targets_dash=torch.cat([self.targets_dash]+targets)
        # self.inputs_dash=torch.cat(inputs)
        # self.targets_dash=torch.cat(targets)
        
        #3- Train the model
        losses=[]
        #get mean & var of tr inputs
        if normalize: self.model.fit_input_stats(self.inputs)
        # self.model.fit_input_stats(self.inputs_dash)
        #create random_idxs from 0 to (no. of tr samples - 1) with size = [B,no. of tr samples]
        idxs = np.random.randint(self.inputs.shape[0], size=[self.inputs.shape[0]])
        num_batches = int(np.ceil(idxs.shape[-1] / self.batch_size)) #(no. of batches=roundup(no. of [model] training input examples so far / batch size))
        for _ in range(self.epochs): #for each epoch
            for batch_num in range(num_batches):
                batch_idxs = idxs[batch_num * self.batch_size : (batch_num + 1) * self.batch_size]
                inputs = self.inputs[batch_idxs] # torch.from_numpy(self.inputs[batch_idxs]).to(self.model.device).float()
                targets = self.targets[batch_idxs] #torch.from_numpy(self.targets[batch_idxs]).to(self.model.device).float()
                mean, logvar, dist  = self.model(inputs,normalize=normalize)
                loss= compute_loss(loss_func,mean,targets,logvar,dist,self.model.var_type,self.model)
                theta_dash = self.model.update_params(loss)
                # Operate on batches:
                inputs_dash = self.inputs_dash[batch_idxs] #torch.from_numpy(self.inputs_dash[batch_idxs]).to(self.model.device).float()
                targets_dash = self.targets_dash[batch_idxs] #torch.from_numpy(self.targets_dash[batch_idxs]).to(self.model.device).float()
                mean, logvar, dist = self.model(inputs_dash,params=theta_dash,normalize=normalize) #fwd pass
                loss= compute_loss(loss_func,mean,targets_dash,logvar,dist,self.model.var_type,self.model)
                # losses.append(loss)
    
            # meta_loss=torch.mean(torch.stack(losses))
                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)
                self.optimizer.step()
                
            # shuffle idxs
            idxs_of_idxs = np.argsort(np.random.uniform(size=idxs.shape), axis=-1)
            idxs = idxs[idxs_of_idxs] #shuffles indicies of each row of idxs randomly (i.e. acc. to idxs_of_idxs)
    
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
            mean, logvar, dist  = self.model(inputs,self.params)
            if self.model.var_type=="out":
                var=torch.exp(logvar)
                delta_obs_next = mean + torch.randn_like(mean, device=self.model.device) * var.sqrt()
            else:
                delta_obs_next = dist.sample()
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
    h=64 #100 #512 #size of hidden layers
    var=0.5 #1. #NN variance if using a fixed varaince
    var_types=[None,"out","param"] #out=output of the network; param=a learned parameter; None=fixed variance
    var_type=var_types[2]
    normalize=False #whether to normalize model inputs
    
    #optimizer
    alpha=0.01 #adaptation step size / learning rate
    beta=0.001 #meta step size / learning rate
    
    #general
    # trials=30 #no. of trials
    tr_eps=20 #50 #no. of training episodes/iterations
    # eval_eps=1
    log_ival=1 #logging interval
    b=4 #32 #batch size: Number of rollouts to sample from each task #=K=M
    meta_b=16 #32 #number of tasks sampled
    seed=1
    
    #controller
    H=8 #8 #planning horizon
    epochs=5 #propagation method epochs
    batch_size=32
    pop_size=60 #1000 #CEM population size: number of candidate solutions to be sampled every iteration 
    opt_max_iters=5 #CEM's max iterations (used as a termination condition)
    
    #algorithm
    loss_funcs=["nll","mse"] #nll: negative log loss; mse: mean squared error
    loss_func= loss_funcs[0]
    # gamma= 1. #discount factor
    
    #%% Initializations
    
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
    out_size=ds*2 if var_type=="out" else ds
    device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = MLP(in_size,n,h,out_size,alpha,var,device,var_type)#.to(device) #dynamics model
    optimizer = Adam(model.parameters(),lr=beta) #model optimizer
    controller = MPC(env,model,H,pop_size,opt_max_iters,optimizer,epochs,batch_size,b)
    
    
    #results 
    plot_tr_rewards=[]
    plot_val_rewards=[]
    best_reward=-1e6
    
    set_seed(seed,env)
    
    #%% Sanity Checks
    assert H<=T, "planning horizon must be at most the total no. of environment timesteps"
    
    #%% Algorithm
    
    episodes=progress(tr_eps)
    for episode in episodes:
        
        tasks = [env.sample_task() for _ in range(meta_b)]

        losses=[]
        rewards_tr_ep, rewards_val_ep = [], []
        # Ds, D_dashes=[], []
        
        for task in tasks:
            
            # #set env task to current task 
            env.reset_task(task,task_name)
            
            states,actions,rewards,states_dash=collect_rollouts(env,controller,model,T,b)
            rewards_tr_ep.append(rewards)
            # Ds.append([states,actions,rewards,states_dash])
            
            inputs=torch.cat([states,actions],-1)
            targets=states_dash-states
            mean, logvar, dist  = model(inputs,normalize=normalize)
            loss= compute_loss(loss_func,mean,targets,logvar,dist,model.var_type,model)
            theta_dash = model.update_params(loss,retain_graph=False)
            
            states,actions,rewards,states_dash=collect_rollouts(env,controller,model,T,b,params=theta_dash)
            rewards_val_ep.append(rewards)
            # D_dashes.append([states,actions,rewards,states_dash])

            inputs=torch.cat([states,actions],-1)
            targets=states_dash-states
            mean, logvar, dist  = model(inputs,params=theta_dash,normalize=normalize)
            loss= compute_loss(loss_func,mean,targets,logvar,dist,model.var_type,model)
            losses.append(loss)
            
        #meta update
        # controller.train(Ds, D_dashes, loss_func)

        meta_loss=torch.mean(torch.stack(losses))
        optimizer.zero_grad()
        meta_loss.backward(create_graph=True,retain_graph=True) #???: should any of them be True?
        optimizer.step()
        
        #compute & log results
        # compute rewards
        reward_ep = (torch.mean(torch.stack([torch.mean(torch.sum(rewards, dim=0)) for rewards in rewards_tr_ep], dim=0))).item()    #sum over T, mean over b, mean over tasks
        reward_val = (torch.mean(torch.stack([torch.mean(torch.sum(rewards, dim=0)) for rewards in rewards_val_ep], dim=0))).item()
        
        #save best running model [params]
        if reward_val>best_reward: 
            best_reward=reward_val
            torch.save(model.state_dict(), "saved_models/"+env_name+".pt")
            
        #log iteration results & statistics
        plot_tr_rewards.append(reward_ep)
        plot_val_rewards.append(reward_val)
        if episode % log_ival == 0:
            log_msg="Rewards Tr: {:.2f}, Rewards Val: {:.2f}".format(reward_ep,reward_val)
            episodes.set_description(desc=log_msg); episodes.refresh()
        
    #%% Results & Plot
    title="Meta-Training Training Rewards (Learning Curve)"
    plt.figure(figsize=(16,8))
    plt.grid(1)
    plt.plot(plot_tr_rewards)
    plt.title(title)
    plt.show()
    
    title="Meta-Training Testing Rewards (Learning Curve)"
    plt.figure(figsize=(16,8))
    plt.grid(1)
    plt.plot(plot_val_rewards)
    plt.title(title)
    plt.show()

#%%
if __name__ == '__main__':
    main()