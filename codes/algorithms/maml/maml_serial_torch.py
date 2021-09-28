# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 18:56:33 2021

@author: TIB001
"""

# %% TODOs
# TODO: make serial implementation
# TODO: make sure algorithm runs and learns
# TODO: add support for parallelism
# TODO: make sure algorithm runs and learns
# TODO: include [custom] environments in the main repo (and remove the need for the gym-custom repo) 
# TODO: study information theory, understand TRPO better, understand MAML better (incl. higher order derivatives in pytorch), summarize meta-learning, upload summaries to repo
# TODO: make some progress with the supervisor stuff
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
# from scipy.stats import truncnorm
from collections import namedtuple, OrderedDict

#ML
import torch
from torch import nn
# from torch.optim import Adam
from torch.distributions import Normal, kl #, MultivariateNormal
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
    # rollout_batch_rewards = []
    
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
        
        # rollout_batch_rewards.append(rewards.sum().item())
    
    #unpack rollouts' states, actions and rewards into dataset D
    states, actions, rewards = zip(*rollout_batch)
    states=torch.cat(states,dim=1).view(T,b,-1).to(device)
    actions=torch.cat(actions,dim=1).view(T,b,-1).to(device)
    rewards=torch.cat(rewards,dim=1).view(T,b,-1).to(device)
    D=[states, actions, rewards]
    
    # return rollout_batch #, rollout_batch_rewards
    return D

def compute_advantages(states,rewards,value_net,gamma):
    
    values = value_net(states).squeeze(2).detach()
    values = torch.nn.functional.pad(values,(0,0,0,1))
    
    deltas = rewards.squeeze(2) + gamma * values[1:] - values[:-1] #delta = r + gamma * v - v' #TD error
    advantages = torch.zeros_like(deltas).float()
    advantage = torch.zeros_like(deltas[0]).float()
    
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
        
        self.logvar = nn.Parameter(np.log(1)*torch.ones(1,out_size,device=device, dtype=torch.float32)) #???
        
        self.layer1=nn.Linear(in_size, h)
        self.layer2=nn.Linear(h, h)
        self.layer3=nn.Linear(h, h)
        self.layer4=nn.Linear(h, out_size)
        
        self.nonlinearity=nn.ReLU()
        
        self.apply(PolicyNetwork.initialize)
    
    @staticmethod
    def initialize(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias.data)
            
    def update_params(self, loss, alpha):
        grads = torch.autograd.grad(loss, self.parameters(), create_graph=False)
        new_params = OrderedDict()
        for (name,param), grad in zip(self.named_parameters(), grads):
            new_params[name]= param - alpha * grad
        return new_params
        
    def forward(self, inputs, params=None):
        if params is None:
            params=OrderedDict(self.named_parameters())
 
        inputs=self.nonlinearity(nn.functional.linear(inputs,weight=params['layer1.weight'],bias= params['layer1.bias']))
        inputs=self.nonlinearity(nn.functional.linear(inputs,weight=params['layer2.weight'],bias= params['layer2.bias']))
        inputs=self.nonlinearity(nn.functional.linear(inputs,weight=params['layer3.weight'],bias= params['layer3.bias']))
        mean=nn.functional.linear(inputs,weight=params['layer4.weight'],bias= params['layer4.bias'])
        
        var = torch.exp(torch.clamp(params['logvar'], min=1e-6)) #???: how come this [i.e logvar] is not the output of the network (even if it is still updated with GD)
        
        return Normal(mean,var) #???: how is this normal and not multivariate normal distribution (w/ iid vars)?
        # return MultivariateNormal(mean,var)
        

class ValueNetwork(nn.Module):
    def __init__(self, in_size, T, b, device, reg_coeff=1e-5):
        super().__init__()
        
        self.reg_coeff=reg_coeff
        self.device = device
        self.T = T
        self.b = b
        
        self.ones = torch.ones(self.T,self.b,1,device=self.device)
        self.timestep= torch.cumsum(self.ones, dim=0) / 100. #torch.arange(self.T).view(-1, 1, 1) * self.ones / 100.0
        self.feature_size=2*in_size + 4
        self.eye=torch.eye(self.feature_size,dtype=torch.float32,device=self.device)
        
        self.linear = nn.Linear(self.feature_size,1,bias=False)
        self.linear.weight.data.zero_()
        
    def fit_params(self, states, rewards):
        
        reg_coeff = self.reg_coeff
        
        #create features
        features = torch.cat([states, states **2, self.timestep, self.timestep**2, self.timestep**3, self.ones],dim=2)
        features=features.view(-1, self.feature_size)

        #compute returns        
        G = torch.zeros(self.b,dtype=torch.float32,device=self.device)
        returns = torch.zeros((self.T,self.b),dtype=torch.float32,device=self.device)
        for t in reversed(range(T)):
            G = rewards[t].squeeze(0).squeeze(1)+gamma*G
            returns[t] = G
        returns = returns.view(-1, 1)
        
        #solve system of equations (i.e. fit) using least squares
        A = torch.matmul(features.t(), returns)
        B = torch.matmul(features.t(), features) 
        for _ in range(5):
            try:
                sol = torch.linalg.lstsq(A, B + reg_coeff * self.eye)
                
                if torch.isnan(sol[0]).any() or torch.isinf(sol[0]).any():
                    raise RuntimeError('NANs/Infs encountered')
                
                break
            except RuntimeError:
                reg_coeff *= 10
        else:
             raise RuntimeError('Unable to find a solution')
        
        #set weights vector
        self.linear.weight.data = sol[0].data.t()
        
    def forward(self, states):
        features = torch.cat([states, states **2, self.timestep, self.timestep**2, self.timestep**3, self.ones],dim=2)
        return torch.matmul(features,self.linear.weight.data)
        # return self.linear(features)
    
    
#--------
# TRPO
#--------

def conjugate_gradients(Avp_f, b, rdotr_tol=1e-10, nsteps=10):
    """
    nsteps = max_iterations
    rdotr = residual
    """
    x = torch.zeros_like(b).float()
    r = b.clone().detach()
    p = b.clone().detach()
    rdotr = torch.dot(r, r)
    
    for i in range(nsteps):
        Avp = Avp_f(p).detach()
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


def line_search(policy, prev_loss, prev_pis, value_net, gamma, T, b, D_dashes, Ds, step, prev_params, max_grad_kl, max_backtracks=10, zeta=0.5):
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
            loss=-(advantages*log_probs.sum(dim=2)).mean()
            theta_dash=policy.update_params(loss, alpha) 
            
            with torch.no_grad():  

                states, actions, rewards = D_dash
                advantages = compute_advantages(states,rewards,value_net,gamma)
                pi=policy(states,params=theta_dash)
                
                loss = - (advantages * torch.exp((pi.log_prob(actions)-prev_pi.log_prob(actions)).sum(dim=2))).mean()
                losses.append(loss)
                
                kl=kl_divergence(pi,prev_pi).mean()
                kls.append(kl)
            
        loss=torch.mean(torch.stack(losses, dim=0))
        kl=torch.mean(torch.stack(kls, dim=0))
        
        #check improvement
        actual_improve = loss - prev_loss
        if (actual_improve.item() < 0.0) and (kl.item() < max_grad_kl):
            break
    else:
        vector_to_parameters(prev_params, policy.parameters())
        

HVP = lambda v: parameters_to_vector(torch.autograd.grad(torch.dot(grad_kl, v),policy.parameters(),retain_graph=True)) + damping * v

# %% Inputs
#model / policy
n=2 #no. of NN layers
h=100 #size of hidden layers

#optimizer
alpha=0.1 #adaptation step size / learning rate
# beta=0.001 #meta step size / learning rate #TIP: controlled and adapted by TRPO (in maml step) [to guarantee monotonic improvement, etc]

#general
K=30 #no. of trials
tr_eps=20 #30 #200 #no. of training episodes/iterations
te_eps=1 #testing episodes
test=False
log_ival=1 #logging interval
# r=1 #no. of rollouts
b=32 #1 #batch size: Number of trajectories to sample from each task
meta_b=30 #30 #number of tasks sampled
log_ival = 1

#VPG
gamma = 0.95

#TRPO
max_grad_kl=0.01 #0.001
max_backtracks=10
accept_ratio=0.1 #0.0
zeta=0.8 #0.5
rdotr_tol=1e-10
nsteps=10
damping=0.01 

# %% Environment
env_names=['cartpole_custom-v1','halfcheetah_custom-v1']
env_name=env_names[1]
env=gym.make(env_name)
T=200 #300 #env._max_episode_steps #task horizon
ds=env.observation_space.shape[0] #state dims
da=env.action_space.shape[0] #action dims

# %% Initializations
Rollout = namedtuple('Rollout', ('states', 'actions', 'rewards'))
in_size=ds
out_size=da
device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
policy = PolicyNetwork(in_size,n,h,out_size,device).to(device) #dynamics model
value_net=ValueNetwork(in_size,T,b,device).to(device) 
plot_rewards=[]
rewards_tr_ep=[]
best_reward=-1e6
theta_dashes=[]
D_dashes=[]
Ds=[]

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

# [T,b,item_size]

set_seed(0,env)

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
        loss=-(advantages*log_probs.sum(dim=2)).mean()
        
        #compute adapted params (via: GD)
        theta_dash=policy.update_params(loss, alpha) 
        theta_dashes.append(theta_dash)
        
        #sample b trajectories/rollouts using f_theta'
        D_dash=collect_rollout_batch(env, policy, T, b, device, params=theta_dash) 
        D_dashes.append(D_dash)
    
    #update meta-params (via: TRPO) 
    #compute surrogate loss
    kls, losses, pis=[], [], []
    for theta_dash, D_dash in zip(theta_dashes,D_dashes):
        #unpack
        states, actions, rewards = D_dash
        
        advantages = compute_advantages(states,rewards,value_net,gamma)
        
        pi=policy(states,params=theta_dash)
        pi_fixed=Normal(loc=pi.loc.detach(),scale=pi.scale.detach())
        pis.append(pi_fixed)
        prev_pi=pi_fixed
        loss = - (advantages * torch.exp((pi.log_prob(actions)-prev_pi.log_prob(actions)).sum(dim=2))).mean()
        losses.append(loss)
        
        kl=kl_divergence(pi,prev_pi).mean()
        kls.append(kl)

    prev_loss=torch.mean(torch.stack(losses, dim=0))
    grads = parameters_to_vector(torch.autograd.grad(prev_loss, policy.parameters(),retain_graph=True))
    kl=torch.mean(torch.stack(kls, dim=0))
    grad_kl=parameters_to_vector(torch.autograd.grad(kl, policy.parameters(),create_graph=True))
    search_step_dir=conjugate_gradients(HVP, grads)
    max_length=torch.sqrt(2 * max_grad_kl / torch.dot(search_step_dir, HVP(search_step_dir)))
    full_step=search_step_dir*max_length
    
    prev_params = parameters_to_vector(policy.parameters())
    line_search(policy, prev_loss, pis, value_net, gamma, T, b, D_dashes, Ds, full_step, prev_params, max_grad_kl, max_backtracks, zeta)

    reward_ep = (torch.mean(torch.stack([torch.mean(torch.sum(rewards, dim=0)) for rewards in rewards_tr_ep], dim=0))).item()    #sum over T, mean over b, stack horiz one reward per task, mean of tasks
    #save best running model [params]
    if reward_ep>best_reward: 
        best_reward=reward_ep
        torch.save(policy.state_dict(), "saved_models/"+env_name+".pt")
    #log iteration results & statistics
    plot_rewards.append(reward_ep)
    if episode % log_ival == 0:
        log_msg="Rewards Sum: {:.2f}".format(reward_ep)
        episodes.set_description(desc=log_msg); episodes.refresh()

#%% Results & Plot
title="Training Rewards (Learning Curve)"
plt.figure(figsize=(16,8))
plt.grid(1)
plt.plot(plot_rewards)
plt.title(title)
plt.show()