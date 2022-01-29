# -*- coding: utf-8 -*-
"""
Created on Sat Jan  1 01:15:23 2022

@author: TIB001
"""

#%% Imports
import numpy as np
import matplotlib.pyplot as plt
import tqdm

import multiprocessing as mp
from baselines.common.vec_env import VecEnv, CloudpickleWrapper
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
from tensorboardX import SummaryWriter

from stable_baselines3 import DDPG as sb_DDPG
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv as vecenv
from stable_baselines3.common.callbacks import BaseCallback, EventCallback

#%% General
seeds=[None,1,2,3,4,5]
seed = seeds[1]

torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type(torch.FloatTensor)
torch.autograd.set_detect_anomaly(True)

# epsilon = np.finfo(np.float32).eps.item()

LUNAR_LANDER_SOLVED_SCORE = 200.0
check_solved = lambda r: np.median(r) > LUNAR_LANDER_SOLVED_SCORE
check_new_best= lambda new, current: new > current
n_eval_points=50 #number of eval points (eval randomization discretization)
eval_eps=3

#%% Utils

progress=lambda x: tqdm.trange(x, leave=True) #for visualizing/monitoring training progress

def set_seed(seed):
    import random
    
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


def rollout(envs,n_workers,T,policy_agent,RB,T_agent_init,b_agent,gamma_agent,t_agent,t_eval,freeze_agent=True,add_noise=False,noise_scale=0.1):

    states=envs.reset()
    ep_rewards = np.zeros(n_workers)
    dones = [False] * n_workers
    add_to_buffer = [True] * n_workers
    t=0
    iters=0
    
    while not all(dones) and t <= T:
        actions=policy_agent.select_action(np.array(states))
        
        if add_noise:
            actions += np.random.normal(0, noise_scale, size=actions.shape)
            actions = actions.clip(-1, 1)
        
        next_states, rewards, dones, _ = envs.step(actions) #this steps through the envs even after they are done, but it doesn't matter here since data from an env is added to buffer only up until the point that the done signal is recieved from that env (and it is an off-policy algorithm)
        # ep_rewards+=np.sum(rewards)
        
        for i, st in enumerate(states):
            if add_to_buffer[i]:
                iters += 1
                t_eval+=1
                t_agent+=1
                ep_rewards[i] += rewards[i]
                
                if RB is not None:
                    done_bool = 0 if t + 1 == T else float(dones[i])
                    RB.add((states[i], next_states[i], actions[i], rewards[i], done_bool))
        
            if dones[i]:
                add_to_buffer[i] = False
                
        states = next_states
        t+=1
    
    if not freeze_agent and len(RB.storage) > T_agent_init: #if it has enough samples
        eps_agent=iters #t*n_rollouts
        policy_agent.train(RB=RB, eps=eps_agent,batch_size=b_agent,gamma=gamma_agent)
            
    return ep_rewards, t_eval, t_agent

#%% Environments

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
            env.randomize(arg)
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

#%% Policy
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
        x = self.max_action * torch.tanh(self.l3(x))
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
    def __init__(self, ds, da, h1, h2, lr_agent, a_max=1.):
        self.actor = Actor(in_size=ds, h1=h1, h2=h2, out_size=da, max_action=a_max).to(device)
        self.actor_target = Actor(in_size=ds, h1=h1, h2=h2, out_size=da, max_action=a_max).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = Adam(self.actor.parameters(),lr=lr_agent)

        self.critic = Critic(in_size=ds+da, h1=h1, h2=h2).to(device)
        self.critic_target = Critic(in_size=ds+da, h1=h1, h2=h2).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = Adam(self.critic.parameters(),lr=lr_agent)
    
    def select_action(self, state):
        state = torch.FloatTensor(state).to(device)
        return self.actor(state).cpu().data.numpy()

    def train(self, RB, eps, batch_size, gamma, tau=0.005):
        for _ in range(eps):
            # Sample replay buffer 
            x, y, u, r, d = RB.sample(batch_size)
            state = torch.FloatTensor(x).to(device)
            action = torch.FloatTensor(u).to(device)
            next_state = torch.FloatTensor(y).to(device)
            done = torch.FloatTensor(1 - d).to(device)
            reward = torch.FloatTensor(r).to(device)

            # Compute the target Q value
            with torch.no_grad():
                target_Q = self.critic_target(next_state, self.actor_target(next_state))
                target_Q = reward + (done * gamma * target_Q)

            # Get current Q estimate
            current_Q = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q, target_Q)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            
            for p in self.critic.parameters():
                p.requires_grad = False

            # Compute actor loss
            actor_loss = -self.critic(state, self.actor(state)).mean()
            
            # Optimize the actor 
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            for p in self.critic.parameters():
                p.requires_grad = True

            # Update the frozen target models (soft update)
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


lr_agent=0.001 #0.001 #learning rate
h1_agent=400 #64 #400
h2_agent=300 #64 #300
gamma_agent=0.99 #discount factor
T_agent_init=1000 #1000 #number of timesteps before any updates
b_agent=1000 #1000 #batch size
T_agent=int(1e6)

#%% Env
env_names=["cartpole_custom-v1","halfcheetah_custom-v1","HalfCheetah-v2",'halfcheetah_custom_norm-v1','halfcheetah_custom_rand-v1','lunarlander_custom_default_rand-v0']
env_name=env_names[-1]
env=gym.make(env_name)
T=env._max_episode_steps #task horizon / max env timesteps
ds=env.observation_space.shape[0] #state dims
da=env.action_space.shape[0] #action dims
a_max=env.action_space.high[0]
dr=env.unwrapped.randomization_space.shape[0] #N_rand (no. of randomization params)

#%% Stable Baselines

def SBCallback(BaseCallback):
    def __init__(self,dr):
         super(SBCallback, self).__init__()
         self.dr=dr
    def _on_step(self):
        return True
    def _on_rollout_start(self):
        self.training_env.randomize(["random"]*self.dr)
        
callback=SBCallback(dr)

policy_kwargs=dict(net_arch=[h1_agent,h2_agent]) #dict(net_arch=[dict(pi=[h1_agent,h2_agent], vf=[h1_agent,h2_agent])]) #None
policy_sb=sb_DDPG("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, buffer_size=int(1e6), learning_rate=lr_agent, tau=0.005, gamma=gamma_agent, learning_starts=T_agent_init, seed=seed, batch_size=b_agent, device=device)
policy_sb.learn(total_timesteps=500000, log_interval=10,callback=callback)

#%% Algorithm

if __name__ == '__main__':
    
    
    n_workers= 5 # mp.cpu_count() - 1 #=n_rollouts
    envs=make_vec_envs(env_name, seed, n_workers)
    
    # hard_envs=make_vec_envs('lunarlander_10_rand-v0', seed, n_workers)
    eval_freq = T * n_workers
    t_eval=0 # agent timesteps since eval 
    
    policy_agent=DDPG(ds, da, h1_agent, h2_agent, lr_agent, a_max)
    
    set_seed(seed)
    env.seed(seed)
    RB=ReplayBuffer()
    plot_rewards=[]
    plot_eval_rewards=[]
    eps=1000
    t_agent=0
    eval_rewards_mean=0
    
    with tqdm.tqdm(total=T_agent) as pbar:
        while t_agent < T_agent:
            
            envs.randomize([["random"]*dr]*n_workers)
            
            ep_rewards,t_eval,t_agent=rollout(envs,n_workers,T,policy_agent,RB,T_agent_init,b_agent,gamma_agent,t_agent,t_eval,freeze_agent=False,add_noise=True)
            
            #evaluate
            with torch.no_grad():
                if t_eval>eval_freq:
                    t_eval %= eval_freq
                    eval_rewards = []
                    for _ in range(eval_eps):
                        envs.randomize([["random"]*dr]*n_workers)
                        ep_eval_rewards,_,_=rollout(envs,n_workers,T,policy_agent,None,T_agent_init,b_agent,gamma_agent,t_agent,t_eval)
                        eval_rewards.append(ep_eval_rewards)
                    
                    eval_rewards_mean=np.mean(np.array(eval_rewards).flatten())
                    plot_eval_rewards.append(eval_rewards_mean)
            
            #log iteration results & statistics
            plot_rewards.append(np.mean(ep_rewards))
            log_msg="Rewards Tr: {:.2f}, Rewards Eval: {:.2f}".format(np.mean(ep_rewards),eval_rewards_mean)
            pbar.update(); pbar.set_description(desc=log_msg); pbar.refresh()
            
            # solved condition for lunar lander
            # if check_solved(ep_rewards) and check_solved(np.array(eval_rewards).flatten()):
            #     print("Solved!")
            #     break

    #%% Results & Plot
    title="Training Rewards (Learning Curve)"
    plt.figure(figsize=(16,8))
    plt.grid(1)
    plt.plot(plot_rewards)
    plt.title(title)
    plt.show()
