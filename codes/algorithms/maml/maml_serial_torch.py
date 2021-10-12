# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 18:56:33 2021

@author: TIB001
"""

# %% TODOs
# TODO: make sure algorithm learns
# TODO: include [custom] environments in the main repo (and remove the need for the gym-custom repo) 
# TODO: study information theory, understand TRPO better, understand MAML better (incl. higher order derivatives in pytorch), summarize meta-learning, upload summaries to repo
# TODO: implements GrBAL/ReBAL (MAML + model-based) --> 1st major milestone 

# %% Imports
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
from collections import namedtuple, OrderedDict

#ML
import torch
torch.cuda.empty_cache()
from torch import nn
# from torch.optim import Adam
from torch.distributions import Normal, Independent, MultivariateNormal #, kl 
from torch.distributions.kl import kl_divergence
from torch.nn.utils.convert_parameters import parameters_to_vector, vector_to_parameters

#%% Functions

#--------
# Utils
#--------

progress=lambda x: tqdm.trange(x, leave=True) #for visualizing/monitoring training progress

def set_seed(seed,env,det=True):
    import random
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    env.seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
        
#--------
# Common
#--------

def collect_rollout_batch(env, policy, T, b, device, params=None): # a batch of rollouts
    rollout_batch = []
    
    #each rollout in the batch is the history of stepping through the environment once till termination
    for _ in range(b):
        s=env.reset()
        rollout = []
        
        for _ in range(T):
            state=torch.from_numpy(s).to(policy.device)
            with torch.no_grad():
                dist=policy(state,params)
                a=dist.sample().squeeze(0).cpu().numpy()#.item()
            s_dash, r, done, _ = env.step(a)
            rollout.append((s,a,r))
            s=s_dash
            if done:
                break
            
        # put rollout into batch
        states, actions, rewards = zip(*rollout)
        states = torch.as_tensor(states) 
        actions = torch.as_tensor(actions)
        rewards = torch.as_tensor(rewards).unsqueeze(1)
        
        rollout_batch.append(Rollout(states, actions, rewards))
            
    #unpack rollouts' states, actions and rewards into dataset D
    states, actions, rewards = zip(*rollout_batch)
    states=torch.cat(states,dim=1).view(T,b,-1).to(device)
    actions=torch.cat(actions,dim=1).view(T,b,-1).to(device)
    rewards=torch.cat(rewards,dim=1).view(T,b,-1).to(device)
    D=[states, actions, rewards]
    
    return D

def compute_advantages(states,rewards,value_net,gamma):
    
    values = value_net(states)
    values = values.squeeze(2).detach() if values.dim()>2 else values.detach()
    values = torch.nn.functional.pad(values,(0,0,0,1))
    
    deltas = rewards.squeeze(2) + gamma * values[1:] - values[:-1] #delta = r + gamma * v - v' #TD error
    advantages = torch.zeros_like(deltas, dtype=torch.float32).float()
    advantage = torch.zeros_like(deltas[0], dtype=torch.float32).float()
    
    for t in reversed(range(value_net.T)):
        advantage = advantage * gamma + deltas[t]
        advantages[t] = advantage
    
    #Normalize advantages to improve: learning, numerical stability & convergence
    advantages = (advantages - advantages.mean()) / (advantages.std()+np.finfo(np.float32).eps)
    
    return advantages

#--------
# Models
#--------

class PolicyNetwork(nn.Module):
    def __init__(self, in_size, n, h, out_size, device):
        super().__init__()
        
        self.device=device
        
        self.logstd = nn.Parameter(np.log(1.0)*torch.ones(1,out_size,device=device, dtype=torch.float32))
        
        self.layer1=nn.Linear(in_size, h)
        self.layer2=nn.Linear(h, h)
        self.layer3=nn.Linear(h, out_size)
        
        self.nonlinearity=nn.ReLU()
        
        self.apply(PolicyNetwork.initialize)
    
    @staticmethod
    def initialize(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias.data)
            
    def update_params(self, loss, alpha):
        grads = torch.autograd.grad(loss, self.parameters(), create_graph=False) #!!!: create_graph  is True in case of not first order approximation #???: why?
        new_params = OrderedDict()
        for (name,param), grad in zip(self.named_parameters(), grads):
            new_params[name]= param - alpha * grad
        return new_params
        
    def forward(self, inputs, params=None):
        if params is None:
            params=OrderedDict(self.named_parameters())
 
        inputs=self.nonlinearity(nn.functional.linear(inputs,weight=params['layer1.weight'],bias= params['layer1.bias']))
        inputs=self.nonlinearity(nn.functional.linear(inputs,weight=params['layer2.weight'],bias= params['layer2.bias']))
        mean=nn.functional.linear(inputs,weight=params['layer3.weight'],bias= params['layer3.bias'])
        
        std = torch.exp(torch.clamp(params['logstd'], min=np.log(1e-6))) #???: how come this [i.e logvar] is not the output of the network (even if it is still updated with GD)
        
        return Normal(mean,std) #???: how is this normal and not multivariate normal distribution (w/ iid vars)?
        # return MultivariateNormal(mean,torch.diag(std[0]))
        # return Independent(Normal(mean,std),1)
        

class ValueNetwork(nn.Module):
    def __init__(self, in_size, T, b, gamma, device, reg_coeff=1e-5):
        super().__init__()
        
        self.reg_coeff=reg_coeff
        self.device = device
        self.T = T
        self.b = b
        self.gamma=gamma
        
        self.ones = torch.ones(self.T,self.b,1,device=self.device,dtype=torch.float32)
        self.timestep= torch.cumsum(self.ones, dim=0) / 100. #torch.arange(self.T).view(-1, 1, 1) * self.ones / 100.0
        self.feature_size=2*in_size + 4
        self.eye=torch.eye(self.feature_size,dtype=torch.float32,device=self.device)
        
        self.w=nn.Parameter(torch.zeros((self.feature_size,1), dtype=torch.float32), requires_grad=False)
        
    def fit_params(self, states, rewards):
        
        reg_coeff = self.reg_coeff
        
        #create features
        features = torch.cat([states, states **2, self.timestep, self.timestep**2, self.timestep**3, self.ones],dim=2)
        features=features.view(-1, self.feature_size)

        #compute returns        
        G = torch.zeros(self.b,dtype=torch.float32,device=self.device)
        returns = torch.zeros((self.T,self.b),dtype=torch.float32,device=self.device)
        for t in reversed(range(T)):
            G = rewards[t].squeeze(0).squeeze(1)+self.gamma*G
            returns[t] = G
        returns = returns.view(-1, 1)
        
        #solve system of equations (i.e. fit) using least squares
        A = torch.matmul(features.t(), features).detach().cpu().numpy()
        B = torch.matmul(features.t(), returns).detach().cpu().numpy()
        for _ in range(5):
            try:
                # sol = torch.linalg.lstsq(A + reg_coeff * self.eye, B)[0]
                # sol = torch.linalg.lstsq(B, A + reg_coeff * self.eye)[0]
                sol = np.linalg.lstsq(A + reg_coeff * self.eye.detach().cpu().numpy(), B, rcond=-1)[0]
                
                if np.any(np.isnan(sol)):
                # if torch.isnan(sol).any() or torch.isinf(sol).any():
                    raise RuntimeError('NANs/Infs encountered')
                
                break
            except RuntimeError:
                reg_coeff *= 10
        else:
             raise RuntimeError('Unable to find a solution')
        
        #set weights vector
        # self.w.data=sol#.t()
        self.w.copy_(torch.as_tensor(sol))
        
    def forward(self, states):
        features = torch.cat([states, states **2, self.timestep, self.timestep**2, self.timestep**3, self.ones],dim=2)
        return torch.matmul(features,self.w)
    
    
#--------
# TRPO
#--------

def conjugate_gradients(Avp_f, b, rdotr_tol=1e-10, nsteps=10):
    """
    nsteps = max_iterations
    rdotr = residual
    """
    x = torch.zeros_like(b,dtype=torch.float32)
    r = b.clone().detach()
    p = b.clone().detach()
    rdotr = torch.dot(r, r)
    
    for i in range(nsteps):
        Avp = Avp_f(p,retain_graph=True).detach()
        alpha = rdotr / torch.dot(p, Avp)
        x += alpha * p
        r -= alpha * Avp
        new_rdotr = torch.dot(r, r)
        beta = new_rdotr / rdotr
        p = r + beta * p
        rdotr = new_rdotr
        if rdotr.item() < rdotr_tol:
            break
        
    return x.detach()


def line_search(policy, prev_loss, prev_pis, value_net, alpha, gamma, T, b, D_dashes, Ds, step, prev_params, max_grad_kl, max_backtracks=10, zeta=0.5):
    """backtracking line search"""
    
    for step_frac in [zeta**x for x in range(max_backtracks)]:
        vector_to_parameters(prev_params - step_frac * step, policy.parameters())
        
        #surrogate loss: compute new kl and loss
        kls, losses =[], []
        for D, D_dash, prev_pi in zip(Ds,D_dashes,prev_pis):
            
            #unpack
            states, actions, rewards = D
            
            #compute new adapted params 
            value_net.fit_params(states,rewards)
            advantages=compute_advantages(states,rewards,value_net,gamma)
            pi=policy(states)
            log_probs=pi.log_prob(actions)
            loss=-(advantages*log_probs.sum(dim=2)).mean() if log_probs.dim()>2 else -(log_probs*advantages).mean()
            theta_dash=policy.update_params(loss, alpha) 
            
            with torch.no_grad():  

                states, actions, rewards = D_dash
                advantages = compute_advantages(states,rewards,value_net,gamma)
                pi=policy(states,params=theta_dash)
                
                ratio=pi.log_prob(actions)-prev_pi.log_prob(actions)
                loss = - (advantages * torch.exp(ratio.sum(dim=2))).mean() if ratio.dim()>2 else - (advantages * torch.exp(ratio)).mean()
                losses.append(loss)
                
                #???: which version is correct?
                # kl=weighted_mean(kl_divergence(pi,prev_pi))
                kl=kl_divergence(prev_pi,pi).mean()
                
                kls.append(kl)
            
        loss=torch.mean(torch.stack(losses, dim=0))
        kl=torch.mean(torch.stack(kls, dim=0))
        
        #check improvement
        actual_improve = loss - prev_loss
        if (actual_improve.item() < 0.0) and (kl.item() < max_grad_kl):
            break
    else:
        vector_to_parameters(prev_params, policy.parameters())
        

def HVP(grad_kl,policy,damping):
    return lambda v, retain_graph: parameters_to_vector(torch.autograd.grad(torch.dot(grad_kl, v),policy.parameters(),retain_graph=retain_graph)) + damping * v

# %% Inputs
#model / policy
n=2 #no. of NN layers
h=64 #100 #size of hidden layers

#optimizer
alpha=0.5 #0.1 #adaptation step size / learning rate
# beta=0.001 #meta step size / learning rate #TIP: controlled and adapted by TRPO (in maml step) [to guarantee monotonic improvement, etc]

#general
# K=30 #no. of trials
tr_eps=20 #200 #no. of training episodes/iterations
log_ival=1 #logging interval
b=16 #32 #batch size: Number of trajectories to sample from each task
meta_b=15 #30 #number of tasks sampled

#VPG
gamma = 0.95

#TRPO
max_grad_kl=0.01
max_backtracks=10
accept_ratio=0.1
zeta=0.8 #0.5
rdotr_tol=1e-10
nsteps=10
damping=0.01 

# %% Environment
env_names=['cartpole_custom-v1','halfcheetah_custom-v1']
env_name=env_names[1]
env=gym.make(env_name)
T=env._max_episode_steps #200 #task horizon
ds=env.observation_space.shape[0] #state dims
da=env.action_space.shape[0] #action dims

# %% Initializations
Rollout = namedtuple('Rollout', ('states', 'actions', 'rewards'))
in_size=ds
out_size=da
device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
policy = PolicyNetwork(in_size,n,h,out_size,device).to(device) #dynamics model
value_net=ValueNetwork(in_size,T,b,gamma,device).to(device) 
plot_tr_rewards=[]
plot_val_rewards=[]
rewards_tr_ep=[]
rewards_val_ep=[]
best_reward=-1e6
theta_dashes=[]
D_dashes=[]
Ds=[]

set_seed(0,env)

# %% Implementation

"""
- modify env to include a function to sample tasks (how to sample the tasks is designed manually)
- create a batch sampler (to sample the batch of tasks):
    - init: a queue, a list of envs (len = n_workers), vectorized environment (the main process manager in charge of num_workers sub-processes interacting with the env)
- create an MLP policy
- ???: create a linear feature baseline (to be used in the VPG alg to ) --> a nn.Module comprised of one linear layer of size (n_features,1) taking manually designed features as inputs (torch.cat([obs,obs**2,al,al**2,al**3,ones], dim=2); where al=action_list)
- create the meta-learner
- for each meta-epoch do:
    - use batch sampler to sample a batch of [meta]tasks --> resolves to sampling [a batch of] tasks from the env
    - meta-learner sample / inner loop (for each task in the sampled batch do):
        - reset tasks (using sampler)
        - sample K trajectories D using policy (f_theta):
            - create batch episodes object
            - collect rollouts according to policy
            - append rollouts to batch episodes object
            - return the batch episodes object
        - adaptation --> compute adapted parameters:
            - fit baseline to episodes
            - calculate loss of episodes --> VPG with GAE
            - calculate grad of loss [grad_theta L_Ti(f_theta)]
            - update parameters w/ GD, i.e. using the equation: theta'_i=theta-alpha*grad_L
        - sample K trajectories D' using f_theta'_i & append to episodes
    - meta-learner step (outer loop) --> using TRPO (but without A2C) to update meta-parameters theta:
        - loss=surrogate_loss(episodes)
        - compute [full] step
        - do line search [which updates parameters within]
    - log rewards
"""

episodes=progress(tr_eps)
for episode in episodes:
    
    #sample batch of tasks
    tasks = env.sample_tasks(meta_b) 
    
    for task in tasks:
        
        #set env task to current task 
        env.reset_task(task) 
        
        #sample b trajectories/rollouts using f_theta
        D=collect_rollout_batch(env, policy, T, b, device)
        Ds.append(D)
        states, actions, rewards = D
        rewards_tr_ep.append(rewards)
        
        #compute loss (via: VPG w/ baseline)
        value_net.fit_params(states,rewards)
        advantages=compute_advantages(states,rewards,value_net,gamma)
        pi=policy(states)
        log_probs=pi.log_prob(actions)
        loss=-(advantages*log_probs.sum(dim=2)).mean() if log_probs.dim()>2 else -(log_probs*advantages).mean()
        
        #compute adapted params (via: GD)
        theta_dash=policy.update_params(loss, alpha) 
        # theta_dashes.append(theta_dash)
        
        #sample b trajectories/rollouts using f_theta'
        D_dash=collect_rollout_batch(env, policy, T, b, device, params=theta_dash) 
        D_dashes.append(D_dash)
    
    #update meta-params (via: TRPO) 
    #compute surrogate loss
    kls, losses, pis=[], [], []
    for D, D_dash in zip(Ds,D_dashes):
        
        #compute loss (via: VPG w/ baseline)
        states, actions, rewards = D
        value_net.fit_params(states,rewards)
        advantages=compute_advantages(states,rewards,value_net,gamma)
        pi=policy(states)
        log_probs=pi.log_prob(actions)
        loss=-(log_probs.sum(dim=2)*advantages).mean() if log_probs.dim()>2 else -(log_probs*advantages).mean()
        theta_dash=policy.update_params(loss, alpha)         
        
        #unpack
        states, actions, rewards = D_dash
        rewards_val_ep.append(rewards)
        
        advantages = compute_advantages(states,rewards,value_net,gamma)
        
        pi=policy(states,params=theta_dash)
        pi_fixed=Normal(loc=pi.loc.detach(),scale=pi.scale.detach())
        pis.append(pi_fixed)
        prev_pi=pi_fixed
        ratio=pi.log_prob(actions)-prev_pi.log_prob(actions)
        loss = - (advantages * torch.exp(ratio.sum(dim=2))).mean() if ratio.dim()>2 else - (advantages * torch.exp(ratio)).mean()
        losses.append(loss)
        
        #???: which version is correct?
        # kl=weighted_mean(kl_divergence(pi,prev_pi))
        kl=kl_divergence(prev_pi,pi).mean()
        
        kls.append(kl)

    prev_loss=torch.mean(torch.stack(losses, dim=0))
    grads = parameters_to_vector(torch.autograd.grad(prev_loss, policy.parameters(),retain_graph=True))
    kl=torch.mean(torch.stack(kls, dim=0))
    grad_kl=parameters_to_vector(torch.autograd.grad(kl, policy.parameters(),create_graph=True))
    hvp=HVP(grad_kl,policy,damping)
    search_step_dir=conjugate_gradients(hvp, grads)
    max_length=torch.sqrt(2.0 * max_grad_kl / torch.dot(search_step_dir, hvp(search_step_dir,retain_graph=False)))
    full_step=search_step_dir*max_length
    
    prev_params = parameters_to_vector(policy.parameters())
    line_search(policy, prev_loss, pis, value_net, alpha, gamma, T, b, D_dashes, Ds, full_step, prev_params, max_grad_kl, max_backtracks, zeta)

    reward_ep = (torch.mean(torch.stack([torch.mean(torch.sum(rewards, dim=0)) for rewards in rewards_tr_ep], dim=0))).item()    #sum over T, mean over b, stack horiz one reward per task, mean of tasks
    reward_val=(torch.mean(torch.stack([torch.mean(torch.sum(rewards, dim=0)) for rewards in rewards_val_ep], dim=0))).item()
    #save best running model [params]
    if reward_ep>best_reward: 
        best_reward=reward_ep
        torch.save(policy.state_dict(), "saved_models/"+env_name+".pt")
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