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
# TIP: retain_graph defaults to the value of create_graph (default: False) 

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
from collections import OrderedDict #, namedtuple

#ML
import torch
torch.cuda.empty_cache()
from torch import nn
# from torch.optim import Adam
from torch.distributions import Normal, Independent #, MultivariateNormal
from torch.distributions.kl import kl_divergence
from torch.nn.utils.convert_parameters import parameters_to_vector, vector_to_parameters

#multiprocessing
import multiprocessing as mp
# import os
# os.environ["OMP_NUM_THREADS"] = "1"
import queue as Q


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

def collect_rollout_batch(envs, ds, da, policy, T, b, n_workers, queue, device, params=None): # a batch of rollouts
    states=[[] for _ in range(b)]
    rewards = [[] for _ in range(b)]
    actions = [[] for _ in range(b)]
    
    states_mat=np.zeros((T,b,ds),dtype=np.float32)
    actions_mat=np.zeros((T,b,da),dtype=np.float32)
    rewards_mat=np.zeros((T,b),dtype=np.float32)
    
    
    for rollout_idx in range(b):
        queue.put(rollout_idx)
    for _ in range(n_workers):
        queue.put(None)
    
    #each rollout in the batch is the history of stepping through the environment once till termination
    s, rollout_idxs=envs.reset()
    dones=[False]
    
    while (not all(dones)) or (not queue.empty()):
        with torch.no_grad():
            state=torch.from_numpy(s).to(device)
            dist=policy(state,params)
            a=dist.sample().cpu().numpy()
        s_dash, r, dones, rollout_idxs_new, _ = envs.step(a)
        #append to batch
        for state, action, reward, rollout_idx in zip(s,a,r,rollout_idxs):
            if rollout_idx is not None:
                states[rollout_idx].append(state.astype(np.float32))
                actions[rollout_idx].append(action.astype(np.float32))
                rewards[rollout_idx].append(reward.astype(np.float32))
        #reset
        s, rollout_idxs = s_dash, rollout_idxs_new
    
    for rollout_idx in range(b):
        states_mat[:,rollout_idx]= np.stack(states[rollout_idx])
        actions_mat[:,rollout_idx]= np.stack(actions[rollout_idx])
        rewards_mat[:,rollout_idx]= np.stack(rewards[rollout_idx])
    
    states=torch.from_numpy(states_mat).to(device)
    actions=torch.from_numpy(actions_mat).to(device)
    rewards=torch.from_numpy(rewards_mat).to(device)
    D=[states, actions, rewards]
    
    return D

def compute_advantages(states,rewards,value_net,gamma):
    
    values = value_net(states).squeeze(2).detach()
    values = torch.nn.functional.pad(values,(0,0,0,1))
    
    deltas = rewards + gamma * values[1:] - values[:-1] #delta = r + gamma * v - v' #TD error
    advantages = torch.zeros_like(deltas, dtype=torch.float32)
    advantage = torch.zeros_like(deltas[0], dtype=torch.float32)
    
    for t in reversed(range(value_net.T)):
        advantage = advantage * gamma + deltas[t]
        advantages[t] = advantage
    
    #Normalize advantages to improve: learning, numerical stability & convergence
    advantages = (advantages - advantages.mean()) / (advantages.std()+np.finfo(np.float32).eps)
    
    return advantages


def reset_tasks(envs,tasks):
    return all(envs.reset_task(tasks))

#--------
# Models
#--------

class PolicyNetwork(nn.Module):
    def __init__(self, in_size, n, h, out_size, device):
        super().__init__()
        
        self.device=device
        
        self.logvar = nn.Parameter(np.log(1.0)*torch.ones(1,out_size,device=device, dtype=torch.float32)) #???
        
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
        inputs=self.nonlinearity(nn.functional.linear(inputs,weight=params['layer3.weight'],bias= params['layer3.bias']))
        mean=nn.functional.linear(inputs,weight=params['layer4.weight'],bias= params['layer4.bias'])
        
        var = torch.exp(torch.clamp(params['logvar'], min=1e-6)) #???: how come this [i.e logvar] is not the output of the network (even if it is still updated with GD)
        
        # return Normal(mean,var) #???: how is this normal and not multivariate normal distribution (w/ iid vars)?
        # return MultivariateNormal(mean,var)
        return Independent(Normal(mean,var),1)
        

class ValueNetwork(nn.Module):
    def __init__(self, in_size, T, b, gamma, device, reg_coeff=1e-5):
        super().__init__()
        
        self.reg_coeff=reg_coeff
        self.device = device
        self.T = T
        self.b = b
        self.gamma=gamma
        
        self.ones = torch.ones(self.T,self.b,1,device=self.device)
        self.timestep= torch.cumsum(self.ones, dim=0) / 100. #torch.arange(self.T).view(-1, 1, 1) * self.ones / 100.0
        self.feature_size=2*in_size + 4
        self.eye=torch.eye(self.feature_size,dtype=torch.float32,device=self.device)
        
        # self.linear = nn.Linear(self.feature_size,1,bias=False)
        # self.linear.weight.data.zero_()
        self.w=nn.Parameter(torch.zeros((self.feature_size,1), dtype=torch.float32), requires_grad=False)
        
    def fit_params(self, states, rewards):
        
        reg_coeff = self.reg_coeff
        
        #create features
        features = torch.cat([states, states **2, self.timestep, self.timestep**2, self.timestep**3, self.ones],dim=2)
        features=features.view(-1, self.feature_size)

        #compute returns        
        G = torch.zeros(self.b,dtype=torch.float32,device=self.device)
        returns = torch.zeros((self.T,self.b),dtype=torch.float32,device=self.device)
        for t in reversed(range(self.T)):
            G = rewards[t]+self.gamma*G
            returns[t] = G
        returns = returns.view(-1, 1)
        
        #solve system of equations (i.e. fit) using least squares
        A = torch.matmul(features.t(), returns)
        B = torch.matmul(features.t(), features) 
        for _ in range(5):
            try:
                #FIXME: which of the following versions is the correct one?
                # sol = torch.linalg.lstsq(A, B + reg_coeff * self.eye)[0]
                sol = torch.linalg.lstsq(B + reg_coeff * self.eye, A)[0] #???: this version learns better
                
                if torch.isnan(sol).any() or torch.isinf(sol).any():
                    raise RuntimeError('NANs/Infs encountered in baseline fitting')
                
                break
            except RuntimeError:
                reg_coeff *= 10
        else:
             raise RuntimeError('Unable to find a solution')
        
        #set weights vector
        # self.linear.weight.data = sol.data.t()
        self.w.data=sol#.t()
        
    def forward(self, states):
        features = torch.cat([states, states **2, self.timestep, self.timestep**2, self.timestep**3, self.ones],dim=2)
        # return torch.matmul(features,self.linear.weight.data)
        # return self.linear(features)
        return torch.matmul(features,self.w.data)
    
    
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


def line_search(policy, prev_loss, prev_pis, value_net, alpha, gamma, T, b, D_dashes, Ds, step, prev_params, max_grad_kl, advs, max_backtracks=10, zeta=0.5):
    """backtracking line search"""
    
    for step_frac in [zeta**x for x in range(max_backtracks)]:
        vector_to_parameters(prev_params - step_frac * step, policy.parameters())
        
        #surrogate loss: compute new kl and loss
        kls, losses, it =[], [], 0
        for D, D_dash, prev_pi in zip(Ds,D_dashes,prev_pis):
            
            #unpack
            states, actions, rewards = D
            
            #compute new adapted params 
            value_net.fit_params(states,rewards)
            advantages=compute_advantages(states,rewards,value_net,gamma)
            pi=policy(states)
            log_probs=pi.log_prob(actions)
            # loss=-(advantages*log_probs.sum(dim=2)).mean()
            loss=-(log_probs*advantages).mean()
            theta_dash=policy.update_params(loss, alpha) 
            
            with torch.no_grad():  

                states, actions, rewards = D_dash
                advantages = advs[it] #compute_advantages(states,rewards,value_net,gamma)
                pi=policy(states,params=theta_dash)
                
                # loss = - (advantages * torch.exp((pi.log_prob(actions)-prev_pi.log_prob(actions)).sum(dim=2))).mean()
                loss = - (advantages * torch.exp(pi.log_prob(actions)-prev_pi.log_prob(actions))).mean()
                losses.append(loss)
                
                kl=kl_divergence(pi,prev_pi).mean()
                kls.append(kl)
            
            it+=1
            
        loss=torch.mean(torch.stack(losses, dim=0))
        kl=torch.mean(torch.stack(kls, dim=0))
        
        #check improvement
        actual_improve = loss - prev_loss
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            raise RuntimeError('NANs/Infs encountered in line search')
        if (actual_improve.item() < 0.0) and (kl.item() < max_grad_kl):
            break
    else:
        vector_to_parameters(prev_params, policy.parameters()) #???: why is this reached a lot here but never in the other codes?
        
def HVP(grad_kl,policy,damping):
    return lambda v, retain_graph: parameters_to_vector(torch.autograd.grad(torch.dot(grad_kl, v),policy.parameters(),retain_graph=retain_graph)) + damping * v

#---

flatten_params = lambda params: torch.cat([param.view(-1) for param in params])

def get_flat_grad(inputs, params, retain_graph=False, create_graph=False):
    """get grad of inputs wrt params, then flatten its elements"""
    if create_graph:
        retain_graph = True

    grads = torch.autograd.grad(inputs, params, retain_graph=retain_graph,      create_graph=create_graph)
    flat_grad = torch.cat([param_grad.view(-1) for param_grad in grads])
    return flat_grad


def update_actor_params(policy,flat_params): #make it update
    n = 0 #prev_index
    for p in policy.parameters(): #p=parameter
        numel = p.numel() #total no. of elements in p #flat_size
        g = flat_params[n:n + numel].view(p.shape)
        p.data += g
        n += numel



#----------------
# Multiprocessing
#----------------

class SubprocVecEnv(gym.Env):
    def __init__(self,env_funcs,ds,da,queue,lock):
        
        self.parent_conns, self.child_conns = zip(*[mp.Pipe() for _ in env_funcs])
        self.workers = [EnvWorker(child_conn, env_func, queue, lock) for (child_conn, env_func) in zip(self.child_conns, env_funcs)]
        
        for worker in self.workers:
            worker.daemon = True #making child processes daemonic to not continue running when master process exists
            worker.start()
        for child_conn in self.child_conns:
            child_conn.close()
        
        self.waiting = False
        self.closed = False
        
    def step(self, actions):
        
        #step through each env asynchronously
        for parent_conn, action in zip(self.parent_conns, actions):
            parent_conn.send(('step',action))
        self.waiting = True
        
        #wait for all envs to finish stepping and then collect results
        results = [parent_conn.recv() for parent_conn in self.parent_conns]
        self.waiting = False
        states, rewards, dones, rollouts_idxs, infos = zip(*results)
        
        return np.stack(states), np.stack(rewards), np.stack(dones), rollouts_idxs, infos
    
    def reset_task(self, tasks):
        for parent_conn, task in zip(self.parent_conns,tasks):
            parent_conn.send(('reset_task',task))
        return np.stack([parent_conn.recv() for parent_conn in self.parent_conns])
    
    def reset(self):
        for parent_conn in self.parent_conns:
            parent_conn.send(('reset',None))
        results = [parent_conn.recv() for parent_conn in self.parent_conns]
        states, rollouts_idxs = zip(*results)
        return np.stack(states), rollouts_idxs
    
    def close(self):
        if self.closed:
            return
        if self.waiting:
            for parent_conn in self.parent_conns:
                parent_conn.recv()
        for parent_conn in self.parent_conns:
            parent_conn.send(('close',None))
        for worker in self.workers:
            worker.join()
        self.closed = True
        
        
class EnvWorker(mp.Process):
    def __init__(self, child_conn, env_func, queue, lock):
        super().__init__()
        
        self.child_conn = child_conn
        self.env = env_func()
        self.queue = queue
        self.lock = lock
        self.rollout_idx = None
        self.done = False
        self.ds = self.env.observation_space.shape
    
    def try_reset(self):
        with self.lock:
            try:
                self.rollout_idx = self.queue.get()
                self.done = (self.rollout_idx is None)
            except Q.Empty:
                self.done = True #!!!: [seems to be] unreached
        if self.done:
            state = np.zeros(self.ds, dtype=np.float32) #!!!: [seems to be] unreached
        else:
            state = self.env.reset()
        return state
    
    def run(self):
        while True:
            func, arg = self.child_conn.recv()
            
            if func == 'step':
                if self.done:
                    state, reward, done, info = np.zeros(self.ds, dtype=np.float32), 0.0, True, {} #!!!: [seems to be] unreached
                else:
                    state, reward, done, info = self.env.step(arg)
                if done and not self.done:
                    state = self.try_reset() #!!!: [seems to be] unreached #!!!: is self.done needed?
                self.child_conn.send((state,reward,done,self.rollout_idx,info))
                
            elif func == 'reset':
                state = self.try_reset()
                self.child_conn.send((state,self.rollout_idx))
              
            elif func == 'reset_task':
                self.env.reset_task(arg)
                self.child_conn.send(True)
                
            elif func == 'close':
                self.child_conn.close()
                break

def make_env(env_name,seed=None):
    def _make_env():
        env = gym.make(env_name)
        env.seed(seed)
        return env
    return _make_env

#%% Main Func
def main():
    # %% Inputs
    #model / policy
    n=2 #no. of NN layers
    h=64 #100 #size of hidden layers
    
    #optimizer
    alpha=0.5 #0.1 #0.5 #adaptation step size / learning rate
    # beta=0.001 #meta step size / learning rate #TIP: controlled and adapted by TRPO (in maml step) [to guarantee monotonic improvement, etc]
    
    #general
    # K=30 #no. of trials
    tr_eps=20 #20 #200 #no. of training episodes/iterations
    log_ival=1 #logging interval
    b=16 #1 #batch size: Number of trajectories (rollouts) to sample from each task
    meta_b=15 #30 #number of tasks sampled
    log_ival = 1
    
    #VPG
    gamma = 0.95 #0.99
    
    #TRPO
    max_grad_kl=0.01 #0.001
    max_backtracks=10 #15
    accept_ratio=0.1 #0.0
    zeta=0.8 #0.5
    rdotr_tol=1e-10
    nsteps=10
    damping=0.01 
    
    #multiprocessing
    n_workers = mp.cpu_count() - 1
    
    # %% Initializations
    #common
    # Rollout = namedtuple('Rollout', ('states', 'actions', 'rewards'))
    theta_dashes=[]
    D_dashes=[]
    Ds=[]
    seed=0
    
    #multiprocessing
    queue = mp.Queue()
    lock = mp.Lock()
    
    #environment
    env_names=['cartpole_custom-v1','halfcheetah_custom-v1']
    env_name=env_names[1]
    env=gym.make(env_name)
    T=env._max_episode_steps #200 #task horizon
    ds=env.observation_space.shape[0] #state dims
    da=env.action_space.shape[0] #action dims
    # env_funcs=[lambda : gym.make(env_name)] * n_workers #!!!: elements should be functions whose value is gym.make(env_name)
    env_funcs=[make_env(env_name,seed)] * n_workers
    envs=SubprocVecEnv(env_funcs, ds, da, queue, lock)
    
    #models
    in_size=ds
    out_size=da
    device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    policy = PolicyNetwork(in_size,n,h,out_size,device).to(device) #dynamics model
    value_net=ValueNetwork(in_size,T,b,gamma,device).to(device)
    
    #results 
    plot_tr_rewards=[]
    plot_val_rewards=[]
    rewards_tr_ep=[]
    rewards_val_ep=[]
    best_reward=-1e6
    
    set_seed(seed,env)
    
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
    
    episodes=progress(tr_eps)
    for episode in episodes:
        
        #sample batch of tasks
        tasks = env.sample_tasks(meta_b) 
        kls, losses, pis, advs =[], [], [], []
        
        for task in tasks:
            
            #set env task to current task
            reset_tasks(envs,[task] * n_workers)
            # envs.reset_task([task] * n_workers) 
            
            #sample b trajectories/rollouts using f_theta
            D=collect_rollout_batch(envs, ds, da, policy, T, b, n_workers, queue, device)
            Ds.append(D)
            states, actions, rewards = D
            rewards_tr_ep.append(rewards)
            
            #compute loss (via: VPG w/ baseline)
            value_net.fit_params(states,rewards)
            advantages=compute_advantages(states,rewards,value_net,gamma)
            pi=policy(states)
            log_probs=pi.log_prob(actions)
            # loss=-(log_probs.sum(dim=2)*advantages).mean()
            loss=-(log_probs*advantages).mean()
            
            #compute adapted params (via: GD) --> perform 1 gradient step update
            theta_dash=policy.update_params(loss, alpha) 
            # theta_dashes.append(theta_dash)
            
            #sample b trajectories/rollouts using f_theta'
            D_dash=collect_rollout_batch(envs, ds, da, policy, T, b, n_workers, queue, device, params=theta_dash)
            D_dashes.append(D_dash)
            states, actions, rewards = D_dash
            rewards_val_ep.append(rewards)
            
            #update meta-params (via: TRPO) 
            #compute surrogate loss
            advantages = compute_advantages(states,rewards,value_net,gamma)
            advs.append(advantages)
            
            pi=policy(states,params=theta_dash)
            # pi_fixed=Normal(loc=pi.loc.detach(),scale=pi.scale.detach())
            pi_fixed=Independent(Normal(loc=pi.base_dist.loc.detach(),scale=pi.base_dist.scale.detach()),pi.reinterpreted_batch_ndims)
            pis.append(pi_fixed)
            prev_pi=pi_fixed
            # loss = - (advantages * torch.exp((pi.log_prob(actions)-prev_pi.log_prob(actions)).sum(dim=2))).mean()
            loss = - (advantages * torch.exp(pi.log_prob(actions)-prev_pi.log_prob(actions))).mean()
            losses.append(loss)
            
            kl=kl_divergence(pi,prev_pi).mean()
            kls.append(kl)
    
        prev_loss=torch.mean(torch.stack(losses, dim=0))
        grads = parameters_to_vector(torch.autograd.grad(prev_loss, policy.parameters(),retain_graph=True))
        kl=torch.mean(torch.stack(kls, dim=0)) #???: why is it always zero?
        grad_kl=parameters_to_vector(torch.autograd.grad(kl, policy.parameters(),create_graph=True))
        hvp=HVP(grad_kl,policy,damping)
        search_step_dir=conjugate_gradients(hvp, grads)
        max_length=torch.sqrt(2.0 * max_grad_kl / torch.dot(search_step_dir, hvp(search_step_dir,retain_graph=False)))
        full_step=search_step_dir*max_length
        
        # print(search_step_dir,full_step,"\n")
        
        prev_params = parameters_to_vector(policy.parameters())
        line_search(policy, prev_loss, pis, value_net, alpha, gamma, T, b, D_dashes, Ds, full_step, prev_params, max_grad_kl, advs, max_backtracks, zeta)
    
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

#%%
if __name__ == '__main__':
    main()


# %% Meta-Testing
# te_eps=1 #testing episodes
# test=False
# if test:
    
#     #load best model
#     model = PE(B,in_size,n,h,out_size,device).to(device)
#     model.load_state_dict(torch.load("saved_models/"+env_name+".pt", map_location=device))
#     model.eval()
#     policy.model=model
    
#     ep_rewards=[]
    
#     episodes=progress(te_eps)
#     for episode in episodes:
#         o=env.reset() #observation
#         O, A, rewards, done= [o], [], [], False
#         policy.reset()
#         for _ in range(100):
#             a=policy.act(o) #first optimal action in sequence (initially random)
    
#             o, r, done, _ = env.step(a) #execute first action from optimal actions
            
#             A.append(a)
#             O.append(o)
#             rewards.append(r)
            
#             env.render()
            
#             if done:
#                 break
        
#         ep_rewards.append(np.sum(rewards))
#         if episode % log_ival == 0:
#             log_msg="Rewards Sum: {:.2f}".format(np.sum(rewards))
#             episodes.set_description(desc=log_msg); episodes.refresh()
        
# # Results & Plots
#     title="Testing Control Actions"
#     plt.figure(figsize=(16,8))
#     plt.grid(1)
#     plt.plot(A)
#     plt.title(title)
#     plt.show()
        
#     title="Testing Rewards"
#     plt.figure(figsize=(16,8))
#     plt.grid(1)
#     plt.plot(ep_rewards)
#     plt.title(title)
#     plt.show()
    
#     env.close()
    