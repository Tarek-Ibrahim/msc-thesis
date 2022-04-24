#%% TODOs

# TODO: check if the weighted mean and normalize usages are correct 
# TODO: check which is the correct version of kl_divergence

#%% Imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.convert_parameters import parameters_to_vector as params2vec, _check_param_device, vector_to_parameters as vec2params
from torch.optim import Adam
from torch.distributions import Normal, Independent , MultivariateNormal
from torch.distributions.kl import kl_divergence
import tqdm
from collections import OrderedDict
from scipy.spatial.distance import squareform, pdist
from baselines.common.vec_env import CloudpickleWrapper
import queue as Q
import gym
#------only for spyder IDE
for env in gym.envs.registration.registry.env_specs.copy():
      if 'custom' in env:
          print('Remove {} from registry'.format(env))
          del gym.envs.registration.registry.env_specs[env]
#------
import gym_custom
import multiprocessing as mp

#%% Common

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def progress(x): #for visualizing/monitoring training progress
    return tqdm.trange(x, leave=True)

def set_seed(seed,env):
    import random
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    env.seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class ReplayBuffer(object):
    def __init__(self, max_size=1e6):
        self.storage = []
        self.max_size = int(max_size)
        self.next_idx = 0

    # Expects tuples of (state, next_state, action, reward, done)
    def add(self, data):
        if self.next_idx >= len(self.storage):
            self.storage.append(data)
        else:
            self.storage[self.next_idx] = data

        self.next_idx = (self.next_idx + 1) % self.max_size

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d = [], [], [], [], []

        for i in ind:
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)


def weighted_mean(tensor, axis=None, weights=None):
    if weights is None:
        out = torch.mean(tensor)
    if axis is None:
        out = torch.sum(tensor * weights) / torch.sum(weights)
    else:
        mean_dim = torch.sum(tensor * weights, dim=axis) / (torch.sum(weights, dim=axis))
        out = torch.mean(mean_dim)
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

def parameters_to_vector(parameters, grad=False, both=False):
    # Flag for the device where the parameter is located
    param_device = None

    if not both:
        if not grad:
            return params2vec(parameters)
        else:
            vec = []
            for param in parameters:
                param_device = _check_param_device(param, param_device)
                vec.append(param.grad.data.view(-1))
                return torch.cat(vec)
    else:
        vec_params, vec_grads = [], []
        for param in parameters:
            param_device = _check_param_device(param, param_device)
            vec_params.append(param.data.view(-1))
            vec_grads.append(param.grad.data.view(-1))
        return torch.cat(vec_params), torch.cat(vec_grads)


def vector_to_parameters(vec, parameters, grad=True):
    param_device = None

    # Pointer for slicing the vector for each parameter
    pointer = 0
    if grad:
        for param in parameters:
            # Ensure the parameters are located in the same device
            param_device = _check_param_device(param, param_device)
            # The length of the parameter
            num_param = torch.prod(torch.LongTensor(list(param.size())))
            param.grad.data = vec[pointer:pointer + num_param].view(param.size())
            # Increment the pointer
            pointer = pointer + num_param
    else:
        vec2params(vec, parameters)


def collect_rollout_batch(envs, ds, da, policy, T, b, n_workers, queue, params=None,add_noise=False,noise_scale=0.1): # a batch of rollouts
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
        s=s.astype(np.float32)
        state=torch.from_numpy(s).to(device)
        dist=policy(state,params)
        a=dist.sample().detach().cpu().numpy()
        if add_noise:
            a = a + np.random.normal(0, noise_scale, size=a.shape)
            # a = a.clip(-1, 1)
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
    
    # D=[states_mat, actions_mat, rewards_mat, masks_mat]
    D=[torch.from_numpy(states_mat).to(device), torch.from_numpy(actions_mat).to(device), torch.from_numpy(rewards_mat).to(device), torch.from_numpy(masks_mat).to(device)]
    
    #concatenate rollouts
    trajs = []
    for rollout_idx in range(b):
        trajs.append(np.concatenate(
            [
                np.array(states[rollout_idx]),
                np.array(actions[rollout_idx]),
                np.array(next_states[rollout_idx])
            ], axis=-1))
    
    return D, trajs


#%% Envs

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

def reset_tasks(envs,tasks):
    return all(envs.reset_task(tasks))

#%% DDPG

class Actor(nn.Module):
    def __init__(self, in_size, h1, h2, out_size, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(in_size, h1)
        self.l2 = nn.Linear(h1, h2)
        self.l3 = nn.Linear(h2, out_size)

        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.max_action * torch.tanh(self.l3(x)) if self.max_action > 0 else self.l3(x)
        return x


class Critic(nn.Module):
    def __init__(self, in_size, h1, h2):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(in_size, h1)
        self.l2 = nn.Linear(h1, h2)
        self.l3 = nn.Linear(h2, 1)


    def forward(self, x, u):
        x = F.relu(self.l1(torch.cat([x, u], 1)))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x
                
class DDPG(object):
    def __init__(self, in_size, out_size, h1, h2, lr_agent, batch_size, epochs, a_max=1., tau=0.005, gamma=0.99):
        
        self.gamma = gamma
        self.epochs = epochs
        self.batch_size = batch_size
        self.tau=tau
        self.a_max = a_max
        
        self.dr=in_size
        self.last_states = np.random.uniform(0, 1, (self.dr,))
        self.timesteps = 0
        
        self.actor = Actor(in_size=in_size, h1=h1, h2=h2, out_size=out_size, max_action=a_max).to(device)
        self.actor_target = Actor(in_size=in_size, h1=h1, h2=h2, out_size=out_size, max_action=a_max).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = Adam(self.actor.parameters(),lr=lr_agent)

        self.critic = Critic(in_size=in_size+out_size, h1=h1, h2=h2).to(device)
        self.critic_target = Critic(in_size=in_size+out_size, h1=h1, h2=h2).to(device) 
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = Adam(self.critic.parameters(),lr=lr_agent*10.)
    
    def select_action(self, state):
        state = torch.from_numpy(state,dtype=torch.float32).to(device)
        return self.actor(state).cpu().data.numpy()
    
    def step(self,RB, T_svpg, T_init, H_svpg):
        
        self.simulation_instance = np.zeros((T_svpg, self.dr))
        RB_samples=[]
        #reset
        current_sim_params = self.last_states
        done=False
        # add_to_buffer=True

        for t in range(T_svpg):
            self.simulation_instance[t] = current_sim_params

            if len(RB.storage) > T_init:
                action = self.select_action(current_sim_params)
            elif self.a_max > 0:
                action = self.a_max * np.random.uniform(-1, 1, (self.dr,))
            else:
                action = np.random.uniform(-1, 1, (self.dr,))
            
            #step
            next_params = current_sim_params + action
            reward = 1.
            done=True if next_params < 0 or next_params > 1 else False
            # next_params = np.clip(next_params,0,1)
            done_bool = 0 if self.timesteps + 1 == H_svpg else float(done)
            
            # if add_to_buffer:
            RB_samples.append([current_sim_params, next_params, action, reward, done_bool])
            
            if done_bool:
                current_sim_params = np.random.uniform(0, 1, (self.dr,))                
                self.timesteps = 0
                # add_to_buffer=False
                # break
            else:
                current_sim_params = next_params
                self.timesteps += 1

        self.last_states = current_sim_params

        return np.array(self.simulation_instance), RB_samples

    def train(self, RB):
        for epoch in range(self.epochs):
            # Sample replay buffer 
            x, y, u, r, d = RB.sample(self.batch_size)
            state = torch.FloatTensor(x).to(device) 
            action = torch.FloatTensor(u).to(device) 
            next_state = torch.FloatTensor(y).to(device) 
            done = torch.FloatTensor(1-d).to(device) 
            reward = torch.FloatTensor(r).to(device)  

            # Compute the target Q value
            target_Q = self.critic_target(next_state, self.actor_target(next_state))
            target_Q = reward + (done * self.gamma * target_Q).detach()

            # Get current Q estimate
            current_Q = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q, target_Q)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Compute actor loss
            actor_loss = -self.critic(state, self.actor(state)).mean()
            
            # Optimize the actor 
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


#%% Discriminator

class Disc_MLP(nn.Module):
    def __init__(self, in_size, h1, h2):
        super(Disc_MLP, self).__init__()
        
        self.l1 = nn.Linear(in_size, h1)
        self.l2 = nn.Linear(h1, h2)
        self.l3 = nn.Linear(h2, 1)

    # Tuple of S-A-S'
    def forward(self, x):
        x = torch.tanh(self.l1(x))
        x = torch.tanh(self.l2(x))
        x = torch.sigmoid(self.l3(x))
        return x


class Discriminator(object): #basically: a binary classifier 
    def __init__(self, ds, da, h, b_disc, r_disc_scale, lr_disc):
        self.discriminator = Disc_MLP(in_size=ds+da+ds, h1=h, h2=h).to(device)  #Input: state-action-state' transition; Output: probability that it was from a reference trajectory

        self.disc_loss_func = nn.BCELoss()
        self.disc_optimizer = Adam(self.discriminator.parameters(),lr=lr_disc)
        self.reward_scale = r_disc_scale
        self.batch_size = b_disc 

    def calculate_rewards(self, randomized_trajectory):
        """
        We want to use the negative of the adversarial calculation (Normally, -log(D)). We want to *reward*
        our simulator for making it easier to discriminate between the reference env + randomized onea
        """
        traj_tensor = torch.from_numpy(randomized_trajectory).float().to(device)

        with torch.no_grad():
            score = (self.discriminator(traj_tensor).cpu().detach().numpy()+1e-8).mean()
        
        reward = np.log(score) - np.log(0.5)

        return self.reward_scale * reward, score

    def train(self, ref_traj, rand_traj, eps):
        """Trains discriminator to distinguish between reference and randomized state action tuples"""
        for _ in range(eps):
            randind = np.random.randint(0, len(rand_traj), size=int(self.batch_size))
            refind = np.random.randint(0, len(ref_traj), size=int(self.batch_size))

            rand_batch = torch.from_numpy(rand_traj[randind]).float().to(device)
            ref_batch = torch.from_numpy(ref_traj[refind]).float().to(device)

            g_o = self.discriminator(rand_batch)
            e_o = self.discriminator(ref_batch)
            
            disc_loss = self.disc_loss_func(g_o, torch.ones((len(rand_batch), 1), device=device)) + self.disc_loss_func(e_o,torch.zeros((len(ref_batch), 1), device=device)) #ref label = 0; rand label = 1
            
            self.disc_optimizer.zero_grad()
            disc_loss.backward()
            self.disc_optimizer.step()

#%% Active DR

def map_rewards(map_type,reward_scale,rand_reward,ref_reward):
    
    if "neg" in map_type:
        simulator_reward = -rand_reward
    elif "delta" in map_type:
        simulator_reward = ref_reward-rand_reward #assumes thr_r >0.
    
    return reward_scale * simulator_reward

#%% MAML

def adapt(D,value_net,policy,alpha):
    
    #unpack
    states, actions, rewards, masks = D
    
    value_net.fit_params(states,rewards,masks)
    
    advantages=compute_advantages(states,rewards,value_net,value_net.gamma,masks)
    pi=policy(states)
    log_probs=pi.log_prob(actions)
    if len(log_probs.shape) > 2:
        log_probs = torch.sum(log_probs,2)
    loss=-weighted_mean(log_probs*advantages,axis=0,weights=masks)
                
    #compute adapted params (via: GD) --> perform 1 gradient step update
    grads = torch.autograd.grad(loss, policy.parameters())
    theta_dash=policy.update_params(grads, alpha) 
    
    return theta_dash 

#%% TRPO (& PPO & VPG)

class PolicyNetwork(nn.Module):
    def __init__(self, in_size, h, out_size):
        super().__init__()
        
        self.layer1=nn.Linear(in_size, h)
        self.layer2=nn.Linear(h, h)
        self.layer3=nn.Linear(h, out_size)
        
        self.logstd = nn.Parameter(np.log(1.0)*torch.ones(1,out_size,device=device, dtype=torch.float32),requires_grad=True)
        
        self.nonlinearity=F.relu
        
        self.apply(PolicyNetwork.initialize)
    
    @staticmethod
    def initialize(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias.data)
            
    def update_params(self, grads, alpha):
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
        
        std = torch.exp(torch.clamp(params['logstd'], min=np.log(1e-6)))
        
        #TIP: MVN=Indep(Normal(),1) --> mainly useful (compared to Normal) for changing the shape og the result of log_prob
        # return Normal(mean,std)
        # return MultivariateNormal(mean,torch.diag(std[0]))
        return Independent(Normal(mean,std),1)
        

class ValueNetwork(nn.Module):
    def __init__(self, in_size, gamma, reg_coeff=1e-5):
        super().__init__()
        
        self.reg_coeff=reg_coeff
        self.gamma=gamma
        
        self.feature_size=2*in_size + 4
        self.eye=torch.eye(self.feature_size,dtype=torch.float32,device=device)
        
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
        G = torch.zeros(b,dtype=torch.float32,device=device)
        returns = torch.zeros((T_max,b),dtype=torch.float32,device=device)
        for t in range(T_max - 1, -1, -1):
            G = rewards[t]*masks[t]+self.gamma*G
            returns[t] = G
        returns = returns.view(-1, 1)
        
        #solve system of equations (i.e. fit) using least squares
        A = torch.matmul(features.t(), features).detach().cpu().numpy()
        B = torch.matmul(features.t(), returns).detach().cpu().numpy()
        for _ in range(5):
            try:
                sol=np.linalg.lstsq(A+reg_coeff * self.eye.detach().cpu().numpy(), B,rcond=-1)[0]                
                
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


def compute_advantages(states,rewards,value_net,gamma,masks,returns=False):
    
    T_max=states.shape[0]
    
    values = value_net(states,masks)
    values = values.squeeze(2).detach() if values.dim()>2 else values.detach()
    values = F.pad(values*masks,(0,0,0,1))
    
    deltas = rewards + gamma * values[1:] - values[:-1] #delta = r + gamma * v - v' #TD error
    # advantages = tf.zeros_like(deltas, dtype=torch.float32)
    advantages = torch.zeros_like(deltas, dtype=torch.float32)
    advantage = torch.zeros_like(deltas[0], dtype=torch.float32)
    
    for t in range(T_max - 1, -1, -1): #reversed(range(-1,T -1 )):
        
        advantage = advantage * gamma + deltas[t] if not returns else rewards[t]+gamma*advantage*masks[t]
        # advantages = advantages.write(t, advantage)
        advantages[t] = advantage
    
    # advantages = advantages.stack()
    
    #Normalize advantages to improve: learning, numerical stability & convergence
    advantages = weighted_normalize(advantages,weights=masks)
    
    return advantages


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


def surrogate_loss(D_dashes,policy,value_net,gamma,alpha=None,prev_pis=None, Ds=None, alg="trpo", clip=0.2):
    
    kls, losses, pis =[], [], []
    if Ds is None:
        Ds = [None] * len(D_dashes)
        
    if prev_pis is None:
        prev_pis = [None] * len(D_dashes)
        
    for D, D_dash, prev_pi in zip(Ds,D_dashes,prev_pis):

        states, actions, rewards, masks = D_dash
        
        if D is not None:
            theta_dash=adapt(D,value_net,policy,alpha)
            pi=policy(states,params=theta_dash)
        else:
            value_net.fit_params(states,rewards,masks)
            pi=policy(states)
        
        pis.append(detach_dist(pi))
        
        if prev_pi is None:
            prev_pi = detach_dist(pi)
        
        advantages=compute_advantages(states,rewards,value_net,gamma,masks)
        
        ratio=pi.log_prob(actions)-prev_pi.log_prob(actions)
        if len(ratio.shape) > 2:
            ratio = torch.sum(ratio, dim=2)
        ratio = torch.exp(ratio)
        surr1= ratio * advantages
        if alg=="trpo":
            loss = surr1
        elif alg=="ppo":
            surr2=torch.clip(ratio,1-clip,1+clip)*advantages
            loss = torch.minimum(surr1,surr2)
            
        loss = - weighted_mean(loss,axis=0,weights=masks)
        losses.append(loss)
        
        if len(actions.shape) > 2:
            masks = masks.unsqueeze(2)
        #???: which version is correct?
        # kl=tf.math.reduce_mean(pi.kl_divergence(prev_pi))
        # kl=tf.math.reduce_mean(prev_pi.kl_divergence(pi))
        # kl= weighted_mean(kl_divergence(prev_pi,pi),axis=0,weights=masks)
        kl= kl_divergence(prev_pi,pi).mean()
        # kl= kl_divergence(pi,prev_pi).mean()
        
        kls.append(kl)
    
    prev_loss=torch.mean(torch.stack(losses, dim=0))
    kl=torch.mean(torch.stack(kls, dim=0)) #???: why is it always zero?
    
    return prev_loss, kl, pis


def line_search(policy, prev_loss, prev_pis, value_net, gamma, b, D_dashes, step, prev_params, max_grad_kl, max_backtracks=10, zeta=0.5, alpha=None, Ds=None):
    """backtracking line search"""
    
    for step_frac in [zeta**x for x in range(max_backtracks)]:
        vector_to_parameters(prev_params - step_frac * step, policy.parameters(), grad=False)
        
        loss, kl, _ = surrogate_loss(D_dashes,policy,value_net,gamma,alpha=alpha,prev_pis=prev_pis,Ds=Ds)
        
        #check improvement
        actual_improve = loss - prev_loss
        if not np.isfinite(loss.item()):
            raise RuntimeError('NANs/Infs encountered in line search')
        if (actual_improve.item() < 0.0) and (kl.item() < max_grad_kl):
            break
    else:
        vector_to_parameters(prev_params, policy.parameters(), grad=False)


def kl_div(D_dashes,value_net,policy,alpha=None,Ds=None):
    kls=[]
    
    if Ds is None:
        Ds = [None] * len(D_dashes)
                          
    for D, D_dash in zip(Ds,D_dashes):
        
        states, actions, _, masks = D_dash
        
        if D is not None:
            theta_dash=adapt(D,value_net,policy,alpha)
        else:
            theta_dash=None
        
        pi=policy(states,params=theta_dash)
        
        prev_pi = detach_dist(pi)
        
        if len(actions.shape) > 2:
            masks = masks.unsqueeze(2)
        #???: which version is correct?
        # kl=tf.math.reduce_mean(pi.kl_divergence(prev_pi))
        # kl=tf.math.reduce_mean(prev_pi.kl_divergence(pi))
        kl= kl_divergence(prev_pi,pi).mean() #weighted_mean(kl_divergence(prev_pi,pi),axis=0,weights=masks)
        # kl= kl_divergence(pi,prev_pi).mean()
        
        kls.append(kl)
    
    return torch.mean(torch.stack(kls, dim=0))


def HVP(D_dashes,policy,value_net,damping,alpha=None,Ds=None):
    def _HVP(v):
        kl = kl_div(D_dashes,value_net,policy,alpha,Ds)
        grad_kl=parameters_to_vector(torch.autograd.grad(kl, policy.parameters(),create_graph=True))
        return parameters_to_vector(torch.autograd.grad(torch.dot(grad_kl, v),policy.parameters())) + damping * v           
    return _HVP

#%% SAC

class SAC_MLP(nn.Module):
    def __init__(self, in_size, h, out_size):
        super(SAC_MLP, self).__init__()
        
        self.mlp=nn.Sequential(
            nn.Linear(in_size, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Linear(h, out_size)
            )
        
    def forward(self,*inputs):
        inputs=torch.cat(inputs,1)
        return self.mlp(inputs)


class SAC_Policy(SAC_MLP):
    def __init__(self, in_size, h, out_size, a_max, a_min):
        super().__init__(in_size, h, out_size)
        
        self.action_scale=torch.FloatTensor((a_max - a_min)/2.).to(device)
        self.action_bias=torch.FloatTensor((a_max + a_min)/2.).to(device)
    
    def forward(self, state):
        output = self.mlp(state)
        mean = output[..., :output.shape[-1] // 2]
        log_std=output[..., output.shape[-1] // 2:]
        log_std = torch.clip(log_std, -20, 2)
        
        std = torch.exp(log_std)
        dist = Normal(mean,std)
         
        x_t = dist.rsample() # for reparameterization trick (mean + std * N(0,1))     
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = dist.log_prob(x_t) - torch.log(self.action_scale * (1. - y_t.pow(2)) + 1e-6)
        log_prob = torch.sum(log_prob,1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        
        return action, log_prob, mean


class SAC(object):
    def __init__(self,in_size, out_size,h, lr,batch_size,epochs,alpha,delta_max,entropy_tuning_method, T_alpha_init,tau=0.005, gamma=0.99, alpha_min=0.0001, alpha_discount=0.1):
        
        self.batch_size=batch_size
        self.epochs=epochs
        self.tau=tau
        self.gamma=gamma
        self.delta_max=delta_max
        self.dr = in_size
        
        self.alpha=alpha
        self.entropy_tuning_method=entropy_tuning_method
        self.alpha_min=alpha_min
        self.alpha_discount=alpha_discount
        self.T_alpha_init=T_alpha_init
        
        self.last_states = np.random.uniform(0, 1, (self.dr,))
        self.timesteps = 0
        
        a_max=np.array([delta_max])
        a_min=np.array([-delta_max])
        
        #Q-Functions
        self.q1=SAC_MLP(in_size=in_size+out_size,h=h,out_size=1).to(device)
        self.q2=SAC_MLP(in_size=in_size+out_size,h=h,out_size=1).to(device)
        
        self.q1_target=SAC_MLP(in_size=in_size+out_size,h=h,out_size=1).to(device)
        self.q2_target=SAC_MLP(in_size=in_size+out_size,h=h,out_size=1).to(device)
        
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        
        self.q1_optimizer = Adam(self.q1.parameters(),lr=lr*10.)
        self.q2_optimizer = Adam(self.q2.parameters(),lr=lr*10.)
        
        #Policy
        self.pi=SAC_Policy(in_size, h, 2*out_size, a_max, a_min).to(device)
        self.pi_optimizer=Adam(self.pi.parameters(),lr=lr)
        
        #Temperature / Entropy (Automatic Tuning)
        if self.entropy_tuning_method=="learn":
            self.target_entropy = -torch.prod(torch.Tensor(self.dr).to(device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = Adam(self.log_alpha.parameters(),lr=lr)
        
    def select_action(self, state):
        state = torch.FloatTensor(state).to(device).unsqueeze(0)
        action, _, _ = self.pi(state)
        return action.detach().cpu().numpy()[0]
    
    def step(self,RB,T_svpg,T_init,H_svpg):
        
        self.simulation_instance = np.zeros((T_svpg, self.dr))
        #reset
        current_sim_params = self.last_states
        # done=False
        # add_to_buffer=True
        RB_samples=[]

        for t in range(T_svpg):
            self.simulation_instance[t] = current_sim_params

            action = self.select_action(current_sim_params) if len(RB.storage) > T_init else self.delta_max * np.random.uniform(-1, 1, (self.dr,))
            
            #step
            next_params = current_sim_params + action
            reward = 1.
            done=True if next_params < 0. or next_params > 1. else False
            # next_params = np.clip(next_params,0,1)
            done_bool = 0 if self.timesteps + 1 == H_svpg else float(done)
            
            RB_samples.append([current_sim_params, next_params, action, reward, done_bool])
            
            if done_bool:
                current_sim_params = np.random.uniform(0, 1, (self.dr,))                
                self.timesteps = 0
                # add_to_buffer=False
                # break
            else:
                current_sim_params = next_params
                self.timesteps += 1

        self.last_states = current_sim_params

        return np.array(self.simulation_instance), RB_samples
    
    
    def train(self,RB):
        
        if len(RB.storage) > self.T_alpha_init and self.entropy_tuning_method=="anneal":
            self.alpha *= self.alpha_discount
            if self.alpha < self.alpha_min:
                self.alpha = self.alpha_min
                    
        for epoch in range(self.epochs):
           
            # Sample replay buffer 
            x, y, u, r, d = RB.sample(self.batch_size)
            state = torch.FloatTensor(x).to(device)
            action = torch.FloatTensor(u).to(device)
            next_state = torch.FloatTensor(y).to(device)
            done = torch.FloatTensor(1 - d).to(device)
            reward = torch.FloatTensor(r).to(device)
            
            with torch.no_grad():
                next_state_action, next_state_log_pi, _ = self.pi(next_state)
                q1_next_target = self.q1_target(next_state, next_state_action)
                q2_next_target = self.q2_target(next_state, next_state_action)
                min_q_next_target = torch.min(q1_next_target, q2_next_target) - self.alpha * next_state_log_pi
                next_q_value = reward + done * self.gamma * (min_q_next_target)
            
            q1_value=self.q1(state,action)
            q1_loss = F.mse_loss(q1_value, next_q_value)
                
            self.q1_optimizer.zero_grad()
            q1_loss.backward()
            self.q1_optimizer.step()
            
            q2_value=self.q2(state,action)
            q2_loss = F.mse_loss(q2_value, next_q_value)
            
            self.q2_optimizer.zero_grad()
            q2_loss.backward()
            self.q2_optimizer.step()
            
            actions_new, log_pi, _ = self.pi(state)
            q_actions_new = torch.min(self.q1(state,actions_new),self.q2(state,actions_new))
            policy_loss = ((self.alpha* log_pi) - q_actions_new).mean()
    
            self.pi_optimizer.zero_grad()
            policy_loss.backward()
            self.pi_optimizer.step()
            
            if self.entropy_tuning_method=="learn":
                alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
    
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
    
                self.alpha = self.log_alpha.exp()
            
            # Update the frozen target models
            for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

#%% SVPG (A2C- & DDPG-based)

class SVPGParticleCritic(nn.Module):
    def __init__(self, in_size, h):
        super(SVPGParticleCritic, self).__init__()
        
        self.critic = nn.Sequential(
            nn.Linear(in_size, h),
            nn.Tanh(),
            nn.Linear(h, h),
            nn.Tanh(),
            nn.Linear(h, 1)
        )

    def forward(self, x):
        return self.critic(x)


class SVPGParticleActor(nn.Module):
    def __init__(self, in_size, h,out_size):
        super(SVPGParticleActor, self).__init__()
        
        self.actor_base = nn.Sequential(
            nn.Linear(in_size, h),
            nn.Tanh(),
            nn.Linear(h, h),
            nn.Tanh(),
            nn.Linear(h, out_size),
            nn.Tanh()
        )
        
        # self.logstd=nn.Parameter(torch.zeros((1,output_dim)))
        self.logstd=nn.Parameter(np.log(1.0)*torch.ones((out_size,1))-1.)
        
    def forward(self, x):
        mean= self.actor_base(x)
        
        std= torch.exp(torch.maximum(self.logstd, torch.from_numpy(np.log(np.array([1e-6]))).to(device)))
        
        dist=Normal(mean,std)
        
        return dist     
    

class SVPG:
    """
    Input: current randomization settings
    Output: either a direction to move in (Discrete - for 1D/2D) or a delta across all parameters (Continuous)
    """
    def __init__(self, n_particles, dr, h, delta_max, T_svpg, H_svpg, temperature, lr_svpg, svpg_kernel_mode,gamma_svpg=0.99, tau=0.005, epochs=30, batch_size=256, T_init=100,base_alg="a2c"):
        
        self.svpg_kernel_mode=svpg_kernel_mode
        self.temperature = temperature 
        self.n_particles = n_particles
        self.T_svpg = T_svpg
        self.dr = dr
        self.H_svpg = H_svpg
        self.delta_max = delta_max
        self.gamma = gamma_svpg
        self.tau=tau
        self.base_alg=base_alg
        
        self.last_states = np.random.uniform(0, 1, (self.n_particles, self.dr))
        self.timesteps = np.zeros(self.n_particles)
        
        self.epochs = epochs
        self.batch_size = batch_size
        self.T_init = T_init
        
        self.particles_actor = []
        self.particles_critic = []
        self.particles_log_probs = [[] for _ in range(n_particles)]
        
        self.particles_actor_target = []
        self.particles_critic_target = []
        
        self.optimizers_actor = []
        self.optimizers_critic = []
        

        for i in range(self.n_particles):
            
            # Initialize each of the individual particles
            actor = SVPGParticleActor(in_size=dr, h=h,out_size=dr).to(device) if base_alg=="a2c" else Actor(in_size=dr, h1=h, h2=h, out_size=dr, max_action=delta_max).to(device)
            actor_optimizer = Adam(actor.parameters(),lr=lr_svpg)
            critic = SVPGParticleCritic(in_size=dr, h=h).to(device) if base_alg=="a2c" else Critic(in_size=2*dr, h1=h, h2=h).to(device)
            critic_optimizer = Adam(critic.parameters(),lr=lr_svpg*10.)
            
            self.particles_actor.append(actor)
            self.particles_critic.append(critic)
            self.optimizers_actor.append(actor_optimizer)
            self.optimizers_critic.append(critic_optimizer)
            
            if base_alg=="ddpg":
                actor_target = Actor(in_size=dr, h1=h, h2=h, out_size=dr, max_action=delta_max).to(device)
                actor_target.load_state_dict(actor.state_dict())
                critic_target = Critic(in_size=2*dr, h1=h, h2=h).to(device) 
                critic_target.load_state_dict(critic.state_dict())
                
                self.particles_actor_target.append(actor_target)
                self.particles_critic_target.append(critic_target)

    def compute_kernel(self, X):
        """
        Computes covariance matrix K(X,X) and its gradient w.r.t. X
        for RBF kernel with design matrix X, as in the second term in eqn (8)
        of reference SVGD paper.

        Args:
            X (Tensor): (S, P), design matrix of samples, where S is num of
                samples, P is the dim of each sample which stacks all params
                into a (1, P) row. Thus P could be 1 millions.
        """

        X_np = X.cpu().data.numpy()
        pairwise_dists = squareform(pdist(X_np))**2

        # Median trick
        h = np.median(pairwise_dists)  
        h = np.sqrt(0.5 * h / np.log(self.n_particles+1))

        # Compute RBF Kernel
        k = torch.exp(-torch.from_numpy(pairwise_dists).to(device).float() / h**2 / 2)

        # Compute kernel gradient
        grad_k = -(k).matmul(X)
        sum_k = torch.sum(k,1)
        
        if self.svpg_kernel_mode==1:
            grad_k= grad_k + torch.matmul(sum_k.unsqueeze(0),X)
        elif self.svpg_kernel_mode==2:
            grad_k= grad_k + torch.multiply(X,sum_k.unsqueeze(1))
        else:
            RuntimeError("unknown svpg kernel mode")
        grad_k = grad_k / (h ** 2)

        return k, grad_k

    def select_action(self, policy_idx, state):
        
        if self.base_alg=="a2c":
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
            dist = self.particles_actor[policy_idx](state)
        
        
            value = self.particles_critic[policy_idx](state)
    
            action = dist.sample()
            self.particles_log_probs[policy_idx].append(dist.log_prob(action))
        
            action = action.item() if self.dr==1 else action.squeeze().cpu().detach().numpy()
            
            return action, value
            
        elif self.base_alg=="ddpg":
            state=torch.FloatTensor(state).to(device)
            action=self.particles_actor[policy_idx](state).cpu().data.numpy()

            return action

    def compute_returns(self, next_value, rewards, masks):
        return_ = next_value 
        returns = []
        for step in reversed(range(len(rewards))):
            # Eq. 80: https://arxiv.org/abs/1704.06440
            return_ = self.gamma * masks[step] * return_ + rewards[step]
            returns.insert(0, return_)

        return returns

    def step(self,RB=None):
        """Rollout trajectories, starting from random initializations of randomization settings (i.e. current_sim_params), each of T_svpg size
        Then, send it to agent for further training and reward calculation
        """
        self.simulation_instances = np.zeros((self.n_particles, self.T_svpg, self.dr))

        # Store the values of each state - for advantage estimation
        self.values=[torch.zeros((self.T_svpg, 1)).float().to(device) for _ in range(self.n_particles)]
        RB_samples=[[] for i in range(self.n_particles)] if RB is not None else None
        # Store the last states for each particle (calculating rewards)
        self.masks = np.ones((self.n_particles, self.T_svpg))

        for i in range(self.n_particles):
                
            self.particles_log_probs[i] = []
            current_sim_params = self.last_states[i]
            
            for t in range(self.T_svpg):
                self.simulation_instances[i][t] = current_sim_params
                
                if self.base_alg=="a2c":
                    action, value = self.select_action(i, current_sim_params)
                    action = self.delta_max * np.array(np.clip(action, -1, 1))
                    next_params = np.clip(current_sim_params + action, 0, 1) #step
                    
                    self.values[i][t] = value
                    
                    done_bool = True if np.array_equal(next_params, current_sim_params) or self.timesteps[i] + 1 == self.H_svpg else False
                    
                elif self.base_alg=="ddpg":
                    if len(RB.storage) > self.T_init:
                        action = self.select_action(i, current_sim_params)
                    elif self.delta_max > 0:
                        action = self.delta_max * np.random.uniform(-1, 1, (self.dr,))
                    else:
                        action = np.random.uniform(-1, 1, (self.dr,))
                    
                    next_params = current_sim_params + action
                    
                    reward = 1.
                    done=True if next_params < 0. or next_params > 1. else False
                    done_bool = 0 if self.timesteps[i] + 1 == self.H_svpg else float(done)
                    
                    RB_samples[i].append([current_sim_params, next_params, action, reward, done_bool])

                if done_bool:
                    current_sim_params = np.random.uniform(0, 1, (self.dr,))                
                    self.timesteps[i] = 0
                else:
                    current_sim_params = next_params
                    self.timesteps[i] += 1

            self.last_states[i] = current_sim_params

        return np.array(self.simulation_instances), RB_samples
    
    
    def train(self,arg):
        if self.base_alg=="a2c":
            return self._train_a2c(arg)
        elif self.base_alg=="ddpg":
            return self._train_ddpg(arg)
    
    def _train_a2c(self, simulator_rewards):
        
        policy_grads = []
        parameters = []
        
        for i in range(self.n_particles):
            
            # policy_grad_particle = []
            
            # Calculate the value of last state - for Return Computation
            _, next_value = self.select_action(i, self.last_states[i]) 

            particle_rewards = torch.from_numpy(simulator_rewards[i]).float().to(device)
            masks = torch.from_numpy(self.masks[i]).float().to(device)
            
            # Calculate entropy-augmented returns, advantages
            returns = self.compute_returns(next_value, particle_rewards, masks)
            returns = torch.cat(returns).detach()
            
            advantages = returns - self.values[i]
            
            # Compute value loss, update critic
            critic_loss = 0.5 * advantages.pow(2).mean()
            
            self.optimizers_critic[i].zero_grad()
            critic_loss.backward()
            self.optimizers_critic[i].step()
            
            # Store policy gradients for SVPG update
            policy_grad = torch.mean(torch.cat(self.particles_log_probs[i][:-1]) * advantages.detach())
            
            self.optimizers_actor[i].zero_grad()
            policy_grad.backward()
            
            # Vectorize parameters and PGs
            vec_param, vec_policy_grad = parameters_to_vector(list(self.particles_actor[i].parameters()), both=True)

            policy_grads.append(vec_policy_grad.unsqueeze(0))
            parameters.append(vec_param.unsqueeze(0))

        # calculating the kernel matrix and its gradients
        parameters = torch.cat(parameters)
        k, grad_k = self.compute_kernel(parameters)

        policy_grads = 1. / self.temperature * torch.cat(policy_grads,0)
        grad_logp = torch.mm(k, policy_grads)

        grad_theta = - (grad_logp + grad_k) / self.n_particles

        # update param gradients
        for i in range(self.n_particles):
            vector_to_parameters(grad_theta[i], list(self.particles_actor[i].parameters()), grad=True)
            self.optimizers_actor[i].step()

    def _train_ddpg(self, RB):
        
        for _ in range(self.epochs):
    
            policy_grads = []
            parameters = []
    
            for i in range(self.n_particles):
                
                x, y, u, r, d = RB.sample(self.batch_size)
                state = torch.FloatTensor(x).to(device)
                action = torch.FloatTensor(u).to(device)
                next_state = torch.FloatTensor(y).to(device)
                done = torch.FloatTensor(1 - d).to(device)
                reward = torch.FloatTensor(r).to(device)
                
                # Compute the target Q value
                target_Q = self.particles_critic_target[i](next_state, self.particles_actor_target[i](next_state))
                target_Q = reward + (done * self.gamma * target_Q).detach()
    
                # Get current Q estimate
                current_Q = self.particles_critic[i](state, action)
    
                # Compute critic loss
                critic_loss = F.mse_loss(current_Q, target_Q)
                    
                self.optimizers_critic[i].zero_grad()
                critic_loss.backward()
                self.optimizers_critic[i].step()
                
                # Compute actor loss
                policy_grad = torch.mean(self.particles_critic[i](state, self.particles_actor[i](state)))
                
                # Optimize the actor
                self.optimizers_actor[i].zero_grad()
                policy_grad.backward()
                
                # Vectorize parameters and PGs
                vec_param, vec_policy_grad = parameters_to_vector(list(self.particles_actor[i].parameters()), both=True)
    
                policy_grads.append(vec_policy_grad.unsqueeze(0))
                parameters.append(vec_param.unsqueeze(0))
    
            # calculating the kernel matrix and its gradients
            parameters = torch.cat(parameters)
            k, grad_k = self.compute_kernel(parameters)
    
            policy_grads = 1. / self.temperature * torch.cat(policy_grads)
            grad_logp = torch.mm(k, policy_grads)
    
            grad_theta = - (grad_logp + grad_k) / self.n_particles
    
            # update param gradients
            for i in range(self.n_particles):
                vector_to_parameters(grad_theta[i], list(self.particles_actor[i].parameters()), grad=True)
                self.optimizers_actor[i].step()
    
                # Update the frozen target models
                for param, target_param in zip(self.particles_critic[i].parameters(), self.particles_critic_target[i].parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
                for param, target_param in zip(self.particles_actor[i].parameters(), self.particles_actor_target[i].parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

