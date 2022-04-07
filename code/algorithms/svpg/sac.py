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
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Distribution, Normal, Independent
import random

#%% General

seed = 101

torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device="cpu"
torch.set_default_tensor_type(torch.FloatTensor)
torch.autograd.set_detect_anomaly(True)


#%% Utils

def set_seed(seed):
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
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


def rollout(n_particles,env,policy_agent,RB,T_env,add_noise=False,noise_scale=0.1): 
    
    states = [[] for _ in range(n_particles)]
    actions = [[] for _ in range(n_particles)]
    next_states = [[] for _ in range(n_particles)]
    rewards = [[] for _ in range(n_particles)]
    ep_rewards = []

    rewards_sum = np.zeros(n_particles)
    state = env.reset()

    done = [False] * n_particles
    add_to_buffer = [True] * n_particles
    t_env = 0 #env timestep

    while not all(done) and t_env <= T_env:
        action = policy_agent.select_action(np.array(state))

        if add_noise:
            action = action + np.random.normal(0, noise_scale, size=action.shape)
            # action = action.clip(-1, 1)

        next_state, reward, done, info = env.step(action)

        #Add samples to replay buffer
        for i, st in enumerate(state):
            if add_to_buffer[i]:
                states[i].append(st)
                actions[i].append(action[i])
                next_states[i].append(next_state[i])
                rewards[i].append(reward[i])
                rewards_sum[i] += reward[i]

                if RB is not None:
                    done_bool = 0 if t_env + 1 == T_env else float(done[i])
                    RB.add((state[i], next_state[i], action[i], reward[i], done_bool))

            if done[i]:
                # Avoid duplicates
                add_to_buffer[i] = False

        state = next_state
        t_env += 1

    ep_rewards.append(rewards_sum)

    return np.array(ep_rewards)

#%% Environment workers

def envworker(child_conn, parent_conn, env_func):
    parent_conn.close()
    env = env_func.x()
    while True:
        func, arg = child_conn.recv()
        
        if func == 'step':
            ob, reward, done, info = env.step(arg)
            child_conn.send((ob, reward, done, info))
        elif func == 'reset':
            ob = env.reset()
            child_conn.send(ob)
        elif func == 'close':
            child_conn.close()
            break
        elif func == 'randomize':
            randomized_val = arg
            env.randomize(randomized_val)
            child_conn.send(None)

class SubprocVecEnv(VecEnv):
    def __init__(self,env_funcs,ds,da):
        
        self.parent_conns, self.child_conns = zip(*[mp.Pipe() for _ in env_funcs])
        self.workers = [mp.Process(target=envworker,args=(child_conn, parent_conn, CloudpickleWrapper(env_func))) for (child_conn, parent_conn, env_func) in zip(self.child_conns, self.parent_conns, env_funcs)]
        
        for worker in self.workers:
            worker.daemon = True #making child processes daemonic to not continue running when master process exists
            worker.start()
        for child_conn in self.child_conns:
            child_conn.close()
        
        self.waiting = False
        self.closed = False
        
        VecEnv.__init__(self, len(env_funcs), ds, da)
        
    def step_async(self, actions):
        #step through each env asynchronously
        for parent_conn, action in zip(self.parent_conns, actions):
            parent_conn.send(('step',action))
        self.waiting = True
        
    def step_wait(self):
        #wait for all envs to finish stepping and then collect results
        results = [parent_conn.recv() for parent_conn in self.parent_conns]
        self.waiting = False
        states, rewards, dones, infos = zip(*results)
        
        return np.stack(states), np.stack(rewards), np.stack(dones), infos
    
    def randomize(self, randomized_values):
        for parent_conn, val in zip(self.parent_conns, randomized_values):
            parent_conn.send(('randomize', val))
        results = [parent_conn.recv() for parent_conn in self.parent_conns]
        self.waiting = False
    
    def reset(self):
        for parent_conn in self.parent_conns:
            parent_conn.send(('reset',None))
        results = [parent_conn.recv() for parent_conn in self.parent_conns]
        return np.stack(results)
    
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

def make_env(env_name,seed=None, rank=None):
    def _make_env():
        env = gym.make(env_name)
        if seed is not None and rank is not None:
            env.seed(seed+rank)
        return env
    return _make_env


def make_vec_envs(env_name, seed, n_workers):
    envs=[make_env(env_name,seed,rank) for rank in range(n_workers)]
    envs=SubprocVecEnv(envs, ds, da)
    return envs

#%% SAC

class MLP(nn.Module):
    def __init__(self,in_size,h,out_size):
        super().__init__()
        
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


class Policy(MLP):
    def __init__(self, in_size, h, out_size, a_max, a_min):
        super().__init__(in_size, h, 2 * out_size)
        
        self.action_scale=torch.FloatTensor((a_max - a_min)/2.).to(device)
        self.action_bias=torch.FloatTensor((a_max + a_min)/2.).to(device)
    
    def forward(self, state):
        output = self.mlp(state)
        mean = output[..., :output.shape[-1] // 2]
        log_std=output[..., output.shape[-1] // 2:]
        log_std = torch.clamp(log_std, min=-20, max=2)
        
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        
        x_t = dist.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = dist.log_prob(x_t) - torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        
        return action, log_prob, mean


class SAC(object):
    def __init__(self,ds,da,h,lr_q,lr_pi,a_max,a_min,batch_size,epochs,T_init,tau,alpha,gamma,lr_alpha):
        
        self.batch_size=batch_size
        self.epochs=epochs
        self.T_init=T_init
        self.tau=tau
        self.alpha=alpha
        self.gamma=gamma
        
        #Q-Functions
        self.q1=MLP(in_size=ds+da,h=h,out_size=1).to(device)
        self.q2=MLP(in_size=ds+da,h=h,out_size=1).to(device)
        
        self.q1_target=MLP(in_size=ds+da,h=h,out_size=1).to(device)
        self.q2_target=MLP(in_size=ds+da,h=h,out_size=1).to(device)
        
        for target_param, param in zip(self.q1_target.parameters(), self.q1.parameters()):
            target_param.data.copy_(param.data)
        
        for target_param, param in zip(self.q2_target.parameters(), self.q2.parameters()):
            target_param.data.copy_(param.data)
        
        self.q1_optimizer = Adam(self.q1.parameters(),lr=lr_q)
        self.q2_optimizer = Adam(self.q2.parameters(),lr=lr_q)
        
        #Policy
        self.pi=Policy(ds, h, da, a_max, a_min).to(device)
        self.pi_optimizer=Adam(self.pi.parameters(), lr=lr_pi)
        
        #Temperature / Entropy (Automatic Tuning)
        self.target_entropy = -torch.prod(torch.Tensor(da).to(device)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = Adam([self.log_alpha], lr=lr_alpha)
        
    def select_action(self, state):
        state = torch.FloatTensor(state).to(device).unsqueeze(0)
        action, _, _ = self.pi(state)
        return action.detach().cpu().numpy()[0]
    
    def train(self,RB):
        if len(RB.storage) > self.T_init: 
            for _ in range(self.epochs):
               
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
                q2_value=self.q2(state,action)
                
                q1_loss = F.mse_loss(q1_value, next_q_value)
                q2_loss = F.mse_loss(q2_value, next_q_value)
                
                self.q1_optimizer.zero_grad()
                q1_loss.backward()
                self.q1_optimizer.step()
        
                self.q2_optimizer.zero_grad()
                q2_loss.backward()
                self.q2_optimizer.step()
                
                actions_new, log_pi, _ = self.pi(state)
                q_actions_new = torch.min(self.q1(state,actions_new),self.q2(state,actions_new))
                policy_loss = ((self.alpha* log_pi) - q_actions_new).mean()
        
                self.pi_optimizer.zero_grad()
                policy_loss.backward()
                self.pi_optimizer.step()
                
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
                

#%% Implementation 
if __name__ == '__main__':
    
    env_names=['halfcheetah_custom_norm-v1','halfcheetah_custom_rand-v1','lunarlander_custom_820_rand-v0','cartpole_custom-v1']
    env_name=env_names[-1]
    env=gym.make(env_name)
    T_env=env._max_episode_steps #task horizon / max env timesteps
    ds=env.observation_space.shape[0] #state dims
    da=env.action_space.shape[0] #action dims
    n_workers=3
    
    envs=make_vec_envs(env_name, seed, n_workers)
    
    h=64
    lr_q=3e-4
    lr_pi=3e-4
    lr_alpha=3e-4
    a_max=env.action_space.high
    a_min=env.action_space.low
    gamma=0.99
    alpha=0.1 #temperature
    tau=0.005
    epochs=30
    batch_size=256
    T_init=0 #300
    
    policy=SAC(ds,da,h,lr_q,lr_pi,a_max,a_min,batch_size,epochs,T_init,tau,alpha,gamma,lr_alpha)
    
    set_seed(seed)
    RB=ReplayBuffer()
    
    T_eps=1000 #number of training episodes
    plot_tr_rewards_mean=[]
    common_name="_sac"
    t_eps=0
    
    with tqdm.tqdm(total=T_eps) as pbar:
       while t_eps < T_eps:
           
           ep_rewards = rollout(n_workers,envs,policy,RB,T_env,add_noise=True,noise_scale=0.1)
           mean_rewards=np.mean(ep_rewards)
           plot_tr_rewards_mean.append(mean_rewards)
           
           policy.train(RB)
           
           #log episode results
           log_msg="Reward: {:.2f}, Episode: {}".format(mean_rewards, t_eps)
           pbar.update(); pbar.set_description(desc=log_msg); pbar.refresh()
           t_eps+=1

    #%% Results & Plots

    title="Training Rewards"
    plt.figure(figsize=(16,8))
    plt.grid(1)
    plt.plot(plot_tr_rewards_mean)
    plt.title(title)
    plt.show()
    