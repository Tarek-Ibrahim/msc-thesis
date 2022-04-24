#%% Imports
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from baselines.common.vec_env import VecEnv, CloudpickleWrapper
import tqdm
import gym
#------only for spyder IDE
for env in gym.envs.registration.registry.env_specs.copy():
     if 'custom' in env:
         print('Remove {} from registry'.format(env))
         del gym.envs.registration.registry.env_specs[env]
#------
import gym_custom
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import Normal, Independent, MultivariateNormal 
import timeit
import queue as Q
import random


#%% General
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device="cpu"
torch.set_default_tensor_type(torch.FloatTensor)
torch.autograd.set_detect_anomaly(True)

#%% Utils

progress=lambda x: tqdm.trange(x, leave=True) #for visualizing/monitoring training progress

def set_seed(seed):
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        

def weighted_mean(tensor, axis=None, weights=None):
    if weights is None:
        out = tensor.mean()
    if axis is None:
        out = (tensor * weights).sum() / weights.sum()
    else:
        mean_dim = (tensor * weights).sum(axis) / (weights).sum(axis)
        out = mean_dim.mean()
    return out


def weighted_normalize(tensor, axis=None, weights=None, epsilon=1e-8):
    mean = weighted_mean(tensor, axis=axis, weights=weights)
    num = tensor * (1 if weights is None else weights) - mean
    std = torch.sqrt(weighted_mean(num ** 2, axis=axis, weights=weights))
    out = num/(std + epsilon)
    return out


def detach_dist(pi):
    if isinstance(pi, Normal):
        return Normal(loc=pi.loc.detach(),scale=pi.scale.detach())
    elif isinstance(pi, MultivariateNormal):
        return MultivariateNormal(pi.loc.detach(),pi.covariance_matrix.detach())
    elif isinstance(pi, Independent):
        return Independent(Normal(loc=pi.base_dist.loc.detach(),scale=pi.base_dist.scale.detach()),pi.reinterpreted_batch_ndims)
    else:
        raise RuntimeError('Distribution not supported')
        

# class ReplayBuffer(object):
#     def __init__(self, max_size=1e6):
#         self.storage = []
#         self.max_size = int(max_size)
#         self.next_idx = 0

#     # Expects tuples of (state, next_state, action, reward, done)
#     def add(self, data):
#         if self.next_idx >= len(self.storage):
#             self.storage.append(data)
#         else:
#             self.storage[self.next_idx] = data

#         self.next_idx = (self.next_idx + 1) % self.max_size

#     def sample(self, batch_size):
#         ind = np.random.randint(0, len(self.storage), size=batch_size)
#         x, y, u, r, d = [], [], [], [], []

#         for i in ind:
#             X, Y, U, R, D = self.storage[i]
#             x.append(np.array(X, copy=False))
#             y.append(np.array(Y, copy=False))
#             u.append(np.array(U, copy=False))
#             r.append(np.array(R, copy=False))
#             d.append(np.array(D, copy=False))

#         return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)


def collect_rollout_batch(envs, ds, da, policy, T, b, n_workers, queue): # a batch of rollouts
    states=[[] for _ in range(b)]
    rewards = [[] for _ in range(b)]
    actions = [[] for _ in range(b)]
    next_states = [[] for _ in range(b)]
    
    for rollout_idx in range(b):
        queue.put(rollout_idx)
    for _ in range(n_workers):
        queue.put(None)
    
    #each rollout in the batch is the history of stepping through the environment once till termination
    s, rollout_idxs=envs.reset()
    dones=[False]
    
    while (not all(dones)) or (not queue.empty()):
        state=torch.from_numpy(s).float().to(device)
        dist=policy(state)
        a=dist.sample().cpu().numpy()
        s_dash, r, dones, rollout_idxs_new, _ = envs.step(a)
        #append to batch
        for state, next_state, action, reward, rollout_idx in zip(s,s_dash,a,r,rollout_idxs):
            if rollout_idx is not None:
                states[rollout_idx].append(state.astype(np.float32))
                next_states[rollout_idx].append(next_state.astype(np.float32))
                actions[rollout_idx].append(action.astype(np.float32))
                rewards[rollout_idx].append(reward.astype(np.float32))

        #reset
        s, rollout_idxs = s_dash, rollout_idxs_new
    
    T_max=max(map(len,rewards))
    states_mat=np.zeros((T_max,b,ds),dtype=np.float32)
    actions_mat=np.zeros((T_max,b,da),dtype=np.float32)
    rewards_mat=np.zeros((T_max,b),dtype=np.float32)
    masks_mat=np.zeros((T_max,b),dtype=np.float32)
    
    for rollout_idx in range(b):
        T_rollout=len(rewards[rollout_idx])
        states_mat[:T_rollout,rollout_idx]= np.stack(states[rollout_idx])
        actions_mat[:T_rollout,rollout_idx]= np.stack(actions[rollout_idx])
        rewards_mat[:T_rollout,rollout_idx]= np.stack(rewards[rollout_idx])
        masks_mat[:T_rollout,rollout_idx]=1.0
    
    D=[torch.from_numpy(states_mat).to(device), torch.from_numpy(actions_mat).to(device), torch.from_numpy(rewards_mat).to(device), torch.from_numpy(masks_mat).to(device)]
    # D=[states_mat, actions_mat, np.expand_dims(rewards_mat,-1), np.expand_dims(masks_mat,-1)]
    
    return D

#%% Environment Workers

class SubprocVecEnv(gym.Env):
    def __init__(self,env_funcs,ds,da,queue,lock):
        
        self.parent_conns, self.child_conns = zip(*[mp.Pipe() for _ in env_funcs])
        # self.workers = [EnvWorker(child_conn, env_func, queue, lock) for (child_conn, env_func) in zip(self.child_conns, env_funcs)]
        self.workers = [mp.Process(target=envworker,args=(child_conn, parent_conn, CloudpickleWrapper(env_func),queue,lock)) for (child_conn, parent_conn, env_func) in zip(self.child_conns, self.parent_conns, env_funcs)]
        
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
    
    def randomize(self, randomized_values):
        for parent_conn, val in zip(self.parent_conns, randomized_values):
            parent_conn.send(('randomize', val))
        results = [parent_conn.recv() for parent_conn in self.parent_conns]
        self.waiting = False
    
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
        

def envworker(child_conn, parent_conn, env_func, queue, lock):
    parent_conn.close()
    env = env_func.x()
    done=False
    rollout_idx = None
    ds=env.observation_space.shape[0]
    
    def try_reset(lock):
        with lock:
            try:
                rollout_idx = queue.get()
                done = (rollout_idx is None)
            except Q.Empty:
                done = True
        if done:
            state = np.zeros(ds, dtype=np.float32)
        else:
            state = env.reset()
        return state, rollout_idx, done
    
    while True:
        func, arg = child_conn.recv()
        
        if func == 'step':
            if done:
                state, reward, done_env, info = np.zeros(ds, dtype=np.float32), 0.0, True, {}
            else:
                state, reward, done_env, info = env.step(arg)
            if done_env and not done:
                state, rollout_idx, done = try_reset(lock)
            child_conn.send((state,reward,done_env,rollout_idx,info))
        elif func == 'reset':
            state, rollout_idx, done = try_reset(lock)
            child_conn.send((state,rollout_idx))
        elif func == 'reset_task':
            env.reset_task(arg)
            child_conn.send(True)
        elif func == 'close':
            child_conn.close()
            break
        elif func == 'randomize':
            env.randomize(arg)
            child_conn.send(None)


def make_env(env_name,seed=None, rank=None):
    def _make_env():
        env = gym.make(env_name)
        if seed is not None and rank is not None:
            env.seed(seed+rank)
        return env
    return _make_env


def make_vec_envs(env_name, seed, n_workers, ds, da, queue, lock):
    envs=[make_env(env_name,seed,rank) for rank in range(n_workers)]
    envs=SubprocVecEnv(envs, ds, da, queue, lock)
    return envs

#%% RL Agent/Policy (PPO)

class Actor(nn.Module):
    def __init__(self, in_size, h, out_size):
        super().__init__()
        
        self.log_std = nn.Parameter(np.log(1.0)*torch.ones(1,out_size,device=device, dtype=torch.float32),requires_grad=True)
        
        self.actor = nn.Sequential(nn.Linear(in_size, h),
                      nn.ReLU(),
                      nn.Linear(h, h),
                      nn.ReLU(),
                      nn.Linear(h, out_size))
        
    def forward(self,x):
        # x=torch.from_numpy(x).float().to(device)
        mean=self.actor(x)
        std = torch.exp(torch.clamp(self.log_std, min=np.log(1e-6)))
        return Independent(Normal(mean,std),1)
        

# class Critic(nn.Module):
#     def __init__(self, in_size, h):
#         super().__init__()
        
#         self.critic = nn.Sequential(nn.Linear(in_size, h),
#                        nn.ReLU(),
#                        nn.Linear(h, h),
#                        nn.ReLU(),
#                        nn.Linear(h, 1))
        
#     def forward(self,x):
#         value=self.critic(x)
#         return value
    

class ValueNetwork(nn.Module):
    def __init__(self, in_size, gamma, device, reg_coeff=1e-5):
        super().__init__()
        
        self.reg_coeff=reg_coeff
        self.gamma=gamma
        self.device=device
        
        self.feature_size=2*in_size + 4
        self.eye=torch.eye(self.feature_size,dtype=torch.float32,device=self.device)
        
        self.w=nn.Parameter(torch.zeros((self.feature_size,1), dtype=torch.float32), requires_grad=False)
        
    def fit_params(self, states, rewards, masks):
        
        T_max=states.shape[0]
        b=states.shape[1]
        ones = masks.unsqueeze(2)
        states = states * ones
        timestep= torch.cumsum(ones, dim=0) / 100.
        
        reg_coeff = self.reg_coeff
        
        #create features
        features = torch.cat([states, states **2, timestep, timestep**2, timestep**3, ones],dim=2)
        features=features.view(-1, self.feature_size)

        #compute returns        
        G = torch.zeros(b,dtype=torch.float32,device=self.device)
        returns = torch.zeros((T_max,b),dtype=torch.float32,device=self.device)
        for t in range(T_max-1,-1,-1):
            G = rewards[t]+self.gamma*G
            returns[t] = G
        returns = returns.view(-1, 1)
        
        #solve system of equations (i.e. fit) using least squares
        A = torch.matmul(features.t(), features).detach().cpu().numpy()
        B = torch.matmul(features.t(), returns).detach().cpu().numpy()
        for _ in range(5):
            try:
                sol = np.linalg.lstsq(A + reg_coeff * self.eye.detach().cpu().numpy(), B, rcond=-1)[0]                
                
                if np.any(np.isnan(sol)):
                    raise RuntimeError('NANs/Infs encountered in baseline fitting')
                
                break
            except RuntimeError:
                reg_coeff *= 10
        else:
             raise RuntimeError('Unable to find a solution')
        
        #set weights vector
        self.w.copy_(torch.as_tensor(sol))
        # self.w.assign(tf.transpose(sol))
        
    def forward(self, states, masks):
        
        ones = masks.unsqueeze(2)
        states = states * ones
        timestep= torch.cumsum(ones, dim=0) / 100.
        
        features = torch.cat([states, states **2, timestep, timestep**2, timestep**3, ones],dim=2)
        
        return torch.matmul(features,self.w)


def compute_advantages(states,rewards,critic,gamma,masks):
    
    T_max=states.shape[0]
    
    values = critic(states,masks)
    if len(list(values.shape))>2: values = values.squeeze(2)
    values = torch.nn.functional.pad(values.detach()*masks,(0,0,0,1))
    
    deltas = rewards + gamma * values[1:] - values[:-1] #delta = r + gamma * v - v' #TD error
    advantages = torch.zeros_like(deltas, dtype=torch.float32)
    advantage = torch.zeros_like(deltas[0], dtype=torch.float32)
    
    for t in range(T_max - 1, -1, -1): #reversed(range(-1,T -1 )):
        advantage = advantage * gamma * masks[t] + deltas[t]
        advantages[t] = advantage
        
    #Normalize advantages to improve: learning, numerical stability & convergence
    advantages = weighted_normalize(advantages,weights=masks)
    
    return advantages


def surrogate_loss(D_dashes,policy,value_net,gamma,clip=0.2,ent_coeff=0.):
    
    losses =[] 
        
    for D_dash in D_dashes:

        states, actions, rewards, masks = D_dash
        
        value_net.fit_params(states,rewards,masks)
        pi=policy(states)
        
        prev_pi = detach_dist(pi)
        
        advantages=compute_advantages(states,rewards,value_net,gamma,masks)
        
        ratio=pi.log_prob(actions)-prev_pi.log_prob(actions)
        if len(ratio.shape) > 2:
            ratio = ratio.sum(2)
        ratio = torch.exp(ratio)
        surr1= ratio * advantages
        surr2=torch.clip(ratio,1.-clip,1.+clip)*advantages
        actor_loss = - weighted_mean(torch.minimum(surr1,surr2),axis=0,weights=masks) + ent_coeff * weighted_mean(pi.entropy(),axis=0,weights=masks)  #+ critic_loss - 0.01 * weighted_mean(pi.entropy(),axis=0,weights=masks)
        # total_loss = - weighted_mean(tf.exp(ratio)*advantages,axis=0,weights=masks)
        losses.append(actor_loss)
    
    loss=torch.stack(losses,0).mean()
    
    return loss


# def surrogate_loss(D_dash,policy,value_net,gamma,clip=0.2):

#     states, actions, rewards, masks = D_dash

#     value_net.fit_params(states,rewards,masks)
#     pi=policy(states)
    
#     prev_pi = detach_dist(pi)
    
#     advantages=compute_advantages(states,rewards,value_net,gamma,masks)
    
#     ratio=pi.log_prob(actions)-prev_pi.log_prob(actions)
#     if len(ratio.shape) > 2:
#         ratio = ratio.sum(2)
#     ratio = torch.exp(ratio)
#     surr1= ratio * advantages
#     surr2=torch.clip(ratio,1-clip,1+clip)*advantages
#     loss = - weighted_mean(torch.minimum(surr1,surr2),axis=0,weights=masks)
    
#     return loss

#%% Implementation (ADR algorithm)
if __name__ == '__main__':
    
    queue = mp.Queue()
    lock = mp.Lock()
    
    seed = 1
    set_seed(seed)
    
    lr=3e-4 #0.001 #3e-4
    gamma=0.99
    h=256 #64 #100
    # tau=0.95 #GAE lambda
    thr_high=6 #7 #8 #7 #2 #6 #20 #high threshold for no. of *consecutive* successes
    thr_low=2 #3 #4 #2 #1 #10 #low threshold for no. of *consecutive* successes
    clip=0.2
    ent_coeff=0. #0.01
    # vf_coeff=1.0
    # l2_reg_weight=1e-6
    epochs=1 #5 #7 #agnet training epochs
    # batch_size=128
    adr_delta=0.15
    pb=0.5 #boundary sampling probability
    
    env_names=['halfcheetah_custom_rand-v2','halfcheetah_custom_rand-v1','lunarlander_custom_820_rand-v0','cartpole_custom-v1','hopper_custom_rand-v2']
    env_name=env_names[0]
    env=gym.make(env_name)
    T_env=env._max_episode_steps #task horizon / max env timesteps
    ds=env.observation_space.shape[0] #state dims
    da=env.action_space.shape[0] #action dims
    dr=env.unwrapped.randomization_space.shape[0]
    n_workers=10 #3 #W
    b=n_workers
    thr_r= 300. #1200. #200. #env.spec.reward_threshold / 50. #8. #define a success in an episode to mean reaching this threshold
    m=20 #30 #3 #240 #length of performance buffer
    meta_b=20 #1 #3 #5
    
    assert thr_low < thr_high <= n_workers
    
    envs=make_vec_envs(env_name, seed, n_workers, ds, da, queue, lock)
    
    policy=Actor(ds,h,da).to(device)
    value_net=ValueNetwork(ds, gamma, device).to(device)
    optimizer=Adam(policy.parameters(),lr=lr)
    
    # RB=ReplayBuffer() #Rollout/Training Buffer
    D={str(env.unwrapped.dimensions[dim].name):{"low":[],"high":[]} for dim in range(dr)} #Performance buffers/queues
    
    phis={str(env.unwrapped.dimensions[dim].name):{"low":env.unwrapped.dimensions[dim].default_value,"high":env.unwrapped.dimensions[dim].default_value} for dim in range(dr)}
    
    phis_plot={str(env.unwrapped.dimensions[dim].name):{"low":[env.unwrapped.dimensions[dim].default_value],"high":[env.unwrapped.dimensions[dim].default_value]} for dim in range(dr)}
    
    lambda_vec=np.zeros(dr)
    
    tr_eps=250 #int(1e6) #1000
    plot_tr_rewards_mean=[]
    sampled_regions = [[] for _ in range(dr)]
    rand_step=0.15 #for discretizing the sampled regions plot
    common_name="_autodr"
    verbose = 1 #0 #1 
    plot_freq=1 #50 #how often to plot
    best_reward=-1e6
    # p_bar=0.
    bounds_reached={str(env.unwrapped.dimensions[dim].name):{"low":0,"high":0} for dim in range(dr)}
    
    episodes=progress(tr_eps) if not verbose else range(tr_eps)
    for episode in episodes:
        
        D_dashes=[]
        lambda_norms=[]
        
        for _ in range(meta_b):
            #lambda ~ P_phi
            lambda_norm = np.zeros_like(lambda_vec)
            for i in range(len(lambda_vec)):
                dim_name=env.unwrapped.dimensions[i].name
                low=env.unwrapped.dimensions[i].range_min
                high=env.unwrapped.dimensions[i].range_max
                # upper_limit=phis[dim_name]["low"] if phis[dim_name]["low"] > phis[dim_name]["high"] else phis[dim_name]["high"]
                lambda_vec[i]=np.random.uniform(phis[dim_name]["low"],phis[dim_name]["high"])
                lambda_norm[i]=(lambda_vec[i]-low)/(high-low)
                
            #execute algorithm 1 (adr eval worker)
            if np.random.rand() < pb: 
                
                i = np.random.randint(0,dr)
                if np.random.rand() < 0.5:
                    boundary="low"
                    other_boundary="high"
                else:
                    boundary="high"
                    other_boundary="low"
                dim_name=env.unwrapped.dimensions[i].name
                lambda_vec[i]=phis[dim_name][boundary]
                
                low=env.unwrapped.dimensions[i].range_min
                high=env.unwrapped.dimensions[i].range_max
                default=env.unwrapped.dimensions[i].default_value
                lambda_norm[i] = (lambda_vec[i]-low)/(high-low)
                envs.randomize(np.tile(lambda_norm,(n_workers,1)))
                
                trajs=collect_rollout_batch(envs, ds, da, policy, T_env, b, n_workers, queue)
                _,_,rewards,_=trajs
                
                p=(rewards.sum(0).detach().cpu().numpy()>=thr_r).astype(int).sum()
                # p=(rewards.sum(0)>=thr_r).astype(int).sum()
                # RB.add(trajs) #add trajs to rollout training buffer
                D[dim_name][boundary].append(p)
                
                if len(D[dim_name][boundary])>=m:
                    p_bar=np.mean(D[dim_name][boundary])
                    D[dim_name][boundary]=[]
                    if p_bar>=thr_high: #expand bounds
                        if boundary=="low": 
                            phis[dim_name][boundary]=max(phis[dim_name][boundary] - adr_delta, low) #decrease lower bound
                            if phis[dim_name]["low"] <= low:
                                bounds_reached[dim_name]["low"]+=1
                        else:
                            phis[dim_name][boundary]=min(phis[dim_name][boundary] + adr_delta, high) #increase upper bound
                            if phis[dim_name]["high"] >= high:
                                bounds_reached[dim_name]["high"]+=1
                        phis_plot[dim_name][boundary].append(phis[dim_name][boundary])
                        phis_plot[dim_name][other_boundary].append(phis[dim_name][other_boundary])

                    elif p_bar<=thr_low: #tighten bounds
                        if boundary=="high":    
                            phis[dim_name][boundary]=max(phis[dim_name][boundary] - adr_delta, default) #decrease upper bound
                        else:
                            phis[dim_name][boundary]=min(phis[dim_name][boundary] + adr_delta, default) #increase lower bound
                        phis_plot[dim_name][boundary].append(phis[dim_name][boundary])
                        phis_plot[dim_name][other_boundary].append(phis[dim_name][other_boundary])
                    # phis[dim_name][boundary] = np.clip(phis[dim_name][boundary],low,high)
                
            #execute algorithm 2 (rollout worker)    
            else:
                envs.randomize(np.tile(lambda_norm,(n_workers,1)))
    
                trajs=collect_rollout_batch(envs, ds, da, policy, T_env, b, n_workers, queue)
                _,_,rewards,_=trajs
            
            plot_tr_rewards_mean.append(rewards.sum(0).mean().detach().cpu().item())
            # plot_tr_rewards_mean.append(rewards.sum(0).mean())
            D_dashes.append(trajs)
            lambda_norms.append(lambda_norm)
            
            # RB.add(trajs) #add trajs to rollout training buffer
        
        #check stop condition (once you've covered the whole range of all randomized dims)
        n_covered_dims=0
        for dim in range(dr):
            dim_name=env.unwrapped.dimensions[i].name
            if bounds_reached[dim_name]["low"] > 1 and bounds_reached[dim_name]["high"] > 1:
                n_covered_dims += 1
        
        if n_covered_dims == dr:
            break
        
        #optimize RL agent/policy
        #sample from rollout/training buffer
        for epoch in range(epochs):
            loss=surrogate_loss(D_dashes,policy,value_net,gamma,clip,ent_coeff)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # for epoch in epochs:
        #     random.shuffle(D_dashes)
        #     loss=surrogate_loss(D_dashes,policy,value_net,gamma,clip,ent_coeff)
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()
        
        # D_dashes=np.concatenate([np.concatenate(D_dash,-1) for D_dash in D_dashes])
        # for epoch in range(epochs):
        #     idxs=np.random.randint(0, D_dashes.shape[0], size=int(D_dashes.shape[0]))
        #     n_batches=D_dashes.shape[0] // batch_size
        #     for batch_num in range(n_batches): 
        #         D_dashes_batch_idxs=idxs[batch_num * batch_size : (batch_num + 1) * batch_size]
        #         D_dashes_batch=D_dashes[D_dashes_batch_idxs]
        #         sn,an,rn,mn=D_dashes_batch[:,:,:ds],D_dashes_batch[:,:,ds:ds+da],D_dashes_batch[:,:,ds+da:ds+da+1],D_dashes_batch[:,:,ds+da+1:]
        #         D_dashes_batch=[[torch.from_numpy(sn).to(device), torch.from_numpy(an).to(device), torch.from_numpy(rn).squeeze().to(device), torch.from_numpy(mn).squeeze().to(device)]]
                
        #         loss=surrogate_loss(D_dashes_batch,policy,value_net,gamma,clip,ent_coeff)
        #         optimizer.zero_grad()
        #         loss.backward()
        #         optimizer.step()
        
        #plot sampled regions
        if episode % plot_freq == 0:
            for dim in range(dr):
                dim_name=env.unwrapped.dimensions[dim].name
                low=env.unwrapped.dimensions[dim].range_min
                high=env.unwrapped.dimensions[dim].range_max
                x=np.arange(low,high+rand_step,rand_step)
                linspace_x=np.arange(min(x),max(x)+2*rand_step,rand_step)
                
                scaled_instances=low + (high-low) * np.array(lambda_norms)
                sampled_regions[dim]=np.concatenate([sampled_regions[dim],scaled_instances.flatten()])
                  
                title=f"Sampled Regions for Randomization Dim = {dim_name} {env.rand} at Episode = {episode}"
                plt.figure(figsize=(16,8))
                plt.grid(1)
                plt.hist(sampled_regions[dim], linspace_x, histtype='barstacked')
                plt.xlim(min(x), max(x)+rand_step)
                plt.title(title)
                plt.savefig(f'plots/sampled_regions_dim_{dim_name}_{env.rand}{common_name}.png')
                plt.close()
                
                title=f"Boundaries Change for Randomization Dim = {dim_name} {env.rand} at Episode = {episode}"
                plt.figure(figsize=(16,8))
                plt.grid(1)
                plt.plot(phis_plot[dim_name]["low"],label="Lower Bound")
                plt.plot(phis_plot[dim_name]["high"],label="Upper Bound")
                plt.title(title)
                plt.legend()
                plt.savefig(f'plots/boundaries_change_dim_{dim_name}_{env.rand}{common_name}.png')
                plt.close()
            
            title="Training Rewards"
            plt.figure(figsize=(16,8))
            plt.grid(1)
            plt.plot(plot_tr_rewards_mean)
            plt.title(title)
            # plt.show()
            plt.savefig(f'plots/tr{common_name}.png')
            plt.close()
        
        #save best running model [params]
        eval_rewards=np.mean(plot_tr_rewards_mean[-meta_b:])
        if eval_rewards>best_reward: 
            best_reward=eval_rewards
            torch.save(policy.state_dict(), f"saved_models/model{common_name}.pt")
        
        log_msg="Rewards Agent: {:.2f}, Avg Performace: {:.2f}".format(rewards.sum(0).mean(), p_bar)
        if verbose:
            print(log_msg+f" Episode: {episode}")
        else:
            episodes.set_description(desc=log_msg); episodes.refresh()
            
    #%% Results & Plots

    # title="Training Rewards"
    # plt.figure(figsize=(16,8))
    # plt.grid(1)
    # plt.plot(plot_tr_rewards_mean)
    # plt.title(title)
    # # plt.show()
    # plt.savefig(f'plots/tr{common_name}.png')
    # plt.close()
