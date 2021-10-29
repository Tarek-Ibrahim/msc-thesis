# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 18:56:33 2021

@author: TIB001
"""

# %% TODOs
# TODO: include [custom] environments in the main repo (and remove the need for the gym-custom repo) 

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
import copy

#ML
import torch
torch.cuda.empty_cache()
from torch import nn
from torch.distributions import Normal, Independent , MultivariateNormal
from torch.distributions.kl import kl_divergence
from torch.nn.utils.convert_parameters import parameters_to_vector, _check_param_device #vector_to_parameters

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
        

def weighted_mean(tensor, lengths=None):
    if lengths is None:
        return torch.mean(tensor)
    for i, length in enumerate(lengths):
        tensor[length:, i].fill_(0.)

    extra_dims = (1,) * (tensor.dim() - 2)
    lengths = torch.as_tensor(lengths, dtype=torch.float32)

    out = torch.sum(tensor, dim=0)
    out.div_(lengths.view(-1, *extra_dims))

    return out


def weighted_normalize(tensor, lengths=None, epsilon=1e-8):
    mean = weighted_mean(tensor, lengths=lengths)
    out = tensor - mean.mean()
    if lengths is not None: 
        for i, length in enumerate(lengths):
            out[length:, i].fill_(0.)

    std = torch.sqrt(weighted_mean(out ** 2, lengths=lengths).mean())
    out.div_(std + epsilon)

    return out

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
            s=s.astype(np.float32)
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
    
    values = value_net(states)
    values = values.squeeze(2).detach() if values.dim()>2 else values.detach()
    values = torch.nn.functional.pad(values,(0,0,0,1))
    
    deltas = rewards + gamma * values[1:] - values[:-1] #delta = r + gamma * v - v' #TD error
    advantages = torch.zeros_like(deltas, dtype=torch.float32)
    advantage = torch.zeros_like(deltas[0], dtype=torch.float32)
    
    for t in range(value_net.T-1,-1,-1):
        advantage = advantage * gamma + deltas[t]
        advantages[t] = advantage
    
    #Normalize advantages to improve: learning, numerical stability & convergence
    # advantages = (advantages - advantages.mean()) / (advantages.std()+np.finfo(np.float32).eps)
    advantages = weighted_normalize(advantages)
    
    return advantages


def reset_tasks(envs,tasks):
    return all(envs.reset_task(tasks))


def adapt(D,value_net,policy,alpha):
    #unpack
    states, actions, rewards = D
    
    #compute new adapted params 
    value_net.fit_params(states,rewards)
    # with torch.set_grad_enabled(True):
    advantages=compute_advantages(states,rewards,value_net,value_net.gamma)
    pi=policy(states)
    log_probs=pi.log_prob(actions)
    # loss=-(advantages*log_probs.sum(dim=2)).mean() if log_probs.dim()>2 else -(log_probs*advantages).mean()
    loss=-weighted_mean(log_probs.sum(dim=2)*advantages) if log_probs.dim()>2 else -weighted_mean(log_probs*advantages)
    theta_dash=policy.update_params(loss, alpha)
    
    return theta_dash     

#--------
# Models
#--------

class PolicyNetwork(nn.Module):
    def __init__(self, in_size, n, h, out_size, device):
        super().__init__()
        
        self.device=device
                
        self.layer1=nn.Linear(in_size, h)
        self.layer2=nn.Linear(h, h)
        # self.layer3=nn.Linear(h, h)
        self.layer4=nn.Linear(h, out_size)
        
        self.logstd = nn.Parameter(np.log(1.0)*torch.ones(1,out_size,device=device, dtype=torch.float32),requires_grad=True)
        
        self.nonlinearity=nn.functional.relu #nn.ReLU()
        
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
        # inputs=self.nonlinearity(nn.functional.linear(inputs,weight=params['layer3.weight'],bias= params['layer3.bias']))
        mean=nn.functional.linear(inputs,weight=params['layer4.weight'],bias= params['layer4.bias'])
        
        std = torch.exp(torch.clamp(params['logstd'], min=np.log(1e-6))) #???: how come this [i.e logvar] is not the output of the network (even if it is still updated with GD)
        
        #TIP: MVN=Indep(Normal(),1) --> mainly useful (compared to Normal) for changing the shape og the result of log_prob
        # return Normal(mean,std)
        # return MultivariateNormal(mean,torch.diag(std[0]))
        return Independent(Normal(mean,std),1)
        

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
        for t in range(self.T-1,-1,-1):
            G = rewards[t]+self.gamma*G
            returns[t] = G
        returns = returns.view(-1, 1)
        
        #solve system of equations (i.e. fit) using least squares
        A = torch.matmul(features.t(), features).detach().cpu().numpy()
        B = torch.matmul(features.t(), returns).detach().cpu().numpy()
        for _ in range(5):
            try:
                #TIP: which of the following versions is the correct one? --> numpy solves ax=b, torch solves bx=a (but why they give same result for bx=a and not for ax=b???)
                # sol = torch.linalg.lstsq(B, A + reg_coeff * self.eye)[0]
                sol = np.linalg.lstsq(A + reg_coeff * self.eye.detach().cpu().numpy(), B, rcond=-1)[0]
                
                if np.any(np.isnan(sol)):
                # if torch.isnan(sol).any() or torch.isinf(sol).any():
                    raise RuntimeError('NANs/Infs encountered in baseline fitting')
                
                break
            except RuntimeError:
                reg_coeff *= 10
        else:
             raise RuntimeError('Unable to find a solution')
        
        #set weights vector
        # self.w.copy_(sol.t())
        self.w.copy_(torch.as_tensor(sol))
        
    def forward(self, states):
        features = torch.cat([states, states **2, self.timestep, self.timestep**2, self.timestep**3, self.ones],dim=2)
        return torch.matmul(features,self.w)
    

def detach_dist(pi):
    # pi_fixed=Normal(loc=pi.loc.detach(),scale=pi.scale.detach())
    # pi_fixed=MultivariateNormal(pi.loc.detach(),pi.covariance_matrix.detach())
    pi_fixed=Independent(Normal(loc=pi.base_dist.loc.detach(),scale=pi.base_dist.scale.detach()),pi.reinterpreted_batch_ndims)
    return pi_fixed

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


def surrogate_loss(Ds, D_dashes,policy,value_net,gamma,alpha,T,prev_pis=None):
    
    kls, losses, pis =[], [], []
    if prev_pis is None:
        prev_pis = [None] * T
        
    for D, D_dash, prev_pi in zip(Ds,D_dashes,prev_pis):
        
        theta_dash=adapt(D,value_net,policy,alpha)
        
        with torch.set_grad_enabled(prev_pi is None):  

            states, actions, rewards = D_dash
            
            pi=policy(states,params=theta_dash)
            pis.append(detach_dist(pi))
            
            if prev_pi is None:
                prev_pi = detach_dist(pi)
            
            advantages=compute_advantages(states,rewards,value_net,gamma)
            
            ratio=pi.log_prob(actions)-prev_pi.log_prob(actions)
            # loss = - (advantages * torch.exp(ratio.sum(dim=2))).mean() if ratio.dim()>2 else - (advantages * torch.exp(ratio)).mean()
            loss = - weighted_mean(advantages * torch.exp(ratio.sum(dim=2))) if ratio.dim()>2 else - weighted_mean(advantages * torch.exp(ratio))
            losses.append(loss)
            
            #???: which version is correct?
            # kl=weighted_mean(kl_divergence(pi,prev_pi))
            kl=kl_divergence(prev_pi,pi).mean()
            
            kls.append(kl)
    
    prev_loss=torch.mean(torch.stack(losses, dim=0))
    kl=torch.mean(torch.stack(kls, dim=0)) #???: why is it always zero?
    
    return prev_loss, kl, pis


def line_search(policy, prev_loss, prev_pis, value_net, alpha, gamma, T, b, D_dashes, Ds, step, prev_params, max_grad_kl, max_backtracks=10, zeta=0.5):
    """backtracking line search"""
    
    for step_frac in [zeta**x for x in range(max_backtracks)]:
        vector_to_parameters(prev_params - step_frac * step, policy.parameters())
        
        loss, kl, _ = surrogate_loss(Ds, D_dashes,policy,value_net,gamma,alpha,T,prev_pis=prev_pis)
        
        #check improvement
        actual_improve = loss - prev_loss
        if not np.isfinite(loss.item()): #torch.isnan(loss).any() or torch.isinf(loss).any():
            raise RuntimeError('NANs/Infs encountered in line search')
        if (actual_improve.item() < 0.0) and (kl.item() < max_grad_kl):
            break
    else:
        vector_to_parameters(prev_params, policy.parameters())
        
def HVP(Ds,D_dashes,policy,damping,value_net,alpha):
    
    def _HVP(v):
        kl=kl_div(Ds,D_dashes,value_net,policy,alpha)
        grad_kl=parameters_to_vector(torch.autograd.grad(kl, policy.parameters(),create_graph=True))
        return parameters_to_vector(torch.autograd.grad(torch.dot(grad_kl, v),policy.parameters())) + damping * v
    return _HVP


def vector_to_parameters(vector, parameters):
    param_device = None

    pointer = 0
    for param in parameters:
        param_device = _check_param_device(param, param_device)

        num_param = param.numel()
        param.data.copy_(vector[pointer:pointer + num_param].view_as(param).data)

        pointer += num_param


def kl_div(Ds,D_dashes,value_net,policy,alpha):
    kls = []
    for D, D_dash in zip(Ds, D_dashes):
        theta_dash=adapt(D,value_net,policy,alpha)
        
        states, actions, rewards = D_dash
        pi=policy(states,params=theta_dash)
        
        prev_pi = detach_dist(pi)
        
        #???: which version is correct?
        kl = weighted_mean(kl_divergence(prev_pi,pi))
        # kl = weighted_mean(kl_divergence(pi,prev_pi))
        
        kls.append(kl)
    
    return torch.mean(torch.stack(kls, dim=0))

#---

# flatten_params = lambda params: torch.cat([param.view(-1) for param in params])

# def get_flat_grad(inputs, params, retain_graph=False, create_graph=False):
#     """get grad of inputs wrt params, then flatten its elements"""
#     if create_graph:
#         retain_graph = True

#     grads = torch.autograd.grad(inputs, params, retain_graph=retain_graph, create_graph=create_graph)
#     flat_grad = torch.cat([param_grad.view(-1) for param_grad in grads])
#     return flat_grad


# def update_policy_params(policy,flat_params): #make it update
#     n = 0 #prev_index
#     for p in policy.parameters(): #p=parameter
#         numel = p.numel() #total no. of elements in p #flat_size
#         g = flat_params[n:n + numel].view(p.shape)
#         p.data += g
#         n += numel



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
                self.done = True
        if self.done:
            state = np.zeros(self.ds, dtype=np.float32)
        else:
            state = self.env.reset()
        return state
    
    def run(self):
        while True:
            func, arg = self.child_conn.recv()
            
            if func == 'step':
                if self.done:
                    state, reward, done, info = np.zeros(self.ds, dtype=np.float32), 0.0, True, {}
                else:
                    state, reward, done, info = self.env.step(arg)
                if done and not self.done:
                    state = self.try_reset()
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
        # env.seed(seed)
        return env
    return _make_env

#%% Main Func
def main():
    
    # %% Inputs
    #model / policy
    n=2 #no. of NN layers
    h=100 #100 #size of hidden layers
    
    #optimizer
    alpha=0.1 #0.1 #adaptation step size / learning rate
    
    #general
    # K=30 #no. of trials
    tr_eps=500 #200 #no. of training episodes/iterations
    log_ival=1 #logging interval
    b=20 #16 #32 #batch size: Number of trajectories (rollouts) to sample from each task
    meta_b=40 #30 #number of tasks sampled
    
    #VPG
    gamma = 0.99
    
    #TRPO
    max_grad_kl=0.01 #considered to be beta: meta step size / learning rate #but actually its the stepfrac controlled by TRPO [to guarantee monotonic improvement, etc]
    max_backtracks=15
    accept_ratio=0.1
    zeta=0.8 #0.5
    rdotr_tol=1e-10
    nsteps=10
    damping=1e-5
    
    #multiprocessing
    n_workers = mp.cpu_count() - 1
    
    # %% Initializations
    #common
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
    best_reward=-1e6
    
    # set_seed(seed,env)
    
    # %% Implementation
        
    episodes=progress(tr_eps)
    for episode in episodes:
        
        #sample batch of tasks
        tasks = env.sample_tasks(meta_b)
        
        rewards_tr_ep, rewards_val_ep = [], []
        Ds, D_dashes=[], []
        
        for task in tasks:
            
            #set env task to current task
            reset_tasks(envs,[task] * n_workers)
            
            #sample b trajectories/rollouts using f_theta
            D=collect_rollout_batch(envs, ds, da, policy, T, b, n_workers, queue, device)
            Ds.append(D)
            _, _, rewards = D
            rewards_tr_ep.append(rewards)
            
            #compute adapted params (via: GD) --> perform 1 gradient step update
            theta_dash=adapt(D,value_net,policy,alpha)
            
            #sample b trajectories/rollouts using f_theta'
            D_dash=collect_rollout_batch(envs, ds, da, policy, T, b, n_workers, queue, device, params=theta_dash)
            D_dashes.append(D_dash)
            _, _, rewards = D_dash
            rewards_val_ep.append(rewards)
        
        #update meta-params (via: TRPO) 
        #compute surrogate loss
        prev_loss, _, prev_pis = surrogate_loss(Ds, D_dashes,policy,value_net,gamma,alpha,T)
        grads = parameters_to_vector(torch.autograd.grad(prev_loss, policy.parameters()))
        hvp=HVP(Ds,D_dashes,policy,damping,value_net,alpha)
        search_step_dir=conjugate_gradients(hvp, grads)
        max_length=torch.sqrt(2.0 * max_grad_kl / torch.dot(search_step_dir, hvp(search_step_dir)))
        full_step=search_step_dir*max_length        
        prev_params = parameters_to_vector(policy.parameters())
        line_search(policy, prev_loss.detach(), prev_pis, value_net, alpha, gamma, T, b, D_dashes, Ds, full_step, prev_params, max_grad_kl, max_backtracks, zeta)
    
        #compute & log results
        # compute rewards
        reward_ep = (torch.mean(torch.stack([torch.mean(torch.sum(rewards, dim=0)) for rewards in rewards_tr_ep], dim=0))).item()    #sum over T, mean over b, stack horiz one reward per task, mean of tasks
        reward_val = (torch.mean(torch.stack([torch.mean(torch.sum(rewards, dim=0)) for rewards in rewards_val_ep], dim=0))).item()
        #save best running model [params]
        if reward_val>best_reward: 
            best_reward=reward_val
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
    plt.savefig('plots/mtr_tr.png')
    # plt.show()
    
    title="Meta-Training Testing Rewards (Learning Curve)"
    plt.figure(figsize=(16,8))
    plt.grid(1)
    plt.plot(plot_val_rewards)
    plt.title(title)
    plt.savefig('plots/mtr_ts.png')
    # plt.show()

#%%
if __name__ == '__main__':
    main()
    