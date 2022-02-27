# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 16:41:21 2021

@author: TIB001
"""

#%% Imports
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from baselines.common.vec_env import VecEnv, CloudpickleWrapper
import tqdm
from scipy.spatial.distance import squareform, pdist
# import os
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
from torch.nn.utils.convert_parameters import parameters_to_vector as params2vec, _check_param_device, vector_to_parameters as vec2params
from torch.optim import Adam
from torch.distributions.kl import kl_divergence
import pandas as pd
import timeit

#%% General
seeds=[None,1,2,3,4,5]
seed = 101 #seeds[1]

# os.environ["OMP_NUM_THREADS"] = "1"

torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device="cpu"
torch.set_default_tensor_type(torch.FloatTensor)
torch.autograd.set_detect_anomaly(True)

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


# def orthogonal_init(module, gain=1):
#     nn.init.orthogonal_(module.weight.data, gain=gain)
#     nn.init.constant_(module.bias.data, 0)
#     return module

def orthogonal_init(x, gain=1):
    return x

# class MLP(nn.Module):
#     def __init__(self, in_size, h1, h2, h_nonlinearity, out_nonlinearity=None, out_size=None, init_func = lambda x, y: x, out_scale=1, out_init_scale=False):
#         super(MLP, self).__init__()
        
#         nonlinearities={"relu": F.relu, "tanh": nn.Tanh(), "sigmoid": torch.sigmoid, None: lambda x: x}
        
#         self.l1 = init_func(nn.Linear(in_size, h1), np.sqrt(2))
#         self.l2 = init_func(nn.Linear(h1, h2), np.sqrt(2))
#         self.h_nonlinearity= nonlinearities[h_nonlinearity]
#         self.out_size=out_size
#         if out_size:  
#             self.out_nonlinearity= nonlinearities[out_nonlinearity]
#             self.out_scale=out_scale
#             self.l3 = init_func(nn.Linear(h2, out_size), np.sqrt(2))
#             # if out_init_scale:
#             #     self.l3.weight.data.mul_(0.1)
#             #     self.l3.bias.data.mul_(0.0)

#     def forward(self, x, u=None):
#         if u is not None:
#             x=torch.cat([x, u], 1)
#         x = self.h_nonlinearity(self.l1(x))
#         x = self.h_nonlinearity(self.l2(x))
#         if self.out_size is not None:
#             x = self.out_scale * self.out_nonlinearity(self.l3(x)) 
#         return x

# class Actor(MLP):
#     def __init__(self, in_size, h1, h2, nonlinearity, out_size=None, init_func = lambda x, y: x,max_out=1):
#         super(Actor, self).__init__(in_size=in_size,h1=h1,h2=h2,h_nonlinearity=nonlinearity,out_nonlinearity="tanh",out_size=out_size,init_func=init_func,out_scale=max_out)


# class Critic(MLP):
#     def __init__(self, in_size, h1, h2, nonlinearity, init_func = lambda x, y: x):
#         super(Critic, self).__init__(in_size=in_size,h1=h1,h2=h2,h_nonlinearity=nonlinearity,out_size=1,init_func=init_func)


def rollout(n_particles,env,policy_agent,RB,eps_rollout_agent,T_env,T_agent_init,b_agent,gamma_agent,freeze_agent=True,add_noise=False,noise_scale=0.1): 
    
    states = [[] for _ in range(n_particles)]
    actions = [[] for _ in range(n_particles)]
    next_states = [[] for _ in range(n_particles)]
    rewards = [[] for _ in range(n_particles)]
    ep_rewards = []

    for ep in range(eps_rollout_agent):
        rewards_sum = np.zeros(n_particles)
        state = env.reset()

        done = [False] * n_particles
        add_to_buffer = [True] * n_particles
        t_env = 0 #env timestep
        training_iters = 0

        while not all(done) and t_env <= T_env:
            action = policy_agent.select_action(np.array(state))

            if add_noise:
                action = action + np.random.normal(0, noise_scale, size=action.shape)
                action = action.clip(-1, 1)

            next_state, reward, done, info = env.step(action)

            #Add samples to replay buffer
            for i, st in enumerate(state):
                if add_to_buffer[i]:
                    states[i].append(st)
                    actions[i].append(action[i])
                    next_states[i].append(next_state[i])
                    rewards[i].append(reward[i])
                    rewards_sum[i] += reward[i]
                    training_iters += 1

                    if RB is not None:
                        done_bool = 0 if t_env + 1 == T_env else float(done[i])
                        RB.add((state[i], next_state[i], action[i], reward[i], done_bool))

                if done[i]:
                    # Avoid duplicates
                    add_to_buffer[i] = False

            state = next_state
            t_env += 1

        # Train agent policy
        if not freeze_agent and len(RB.storage) > T_agent_init: #if it has enough samples
            policy_agent.train(RB=RB, eps=training_iters,batch_size=b_agent,gamma=gamma_agent)

        ep_rewards.append(rewards_sum)

    #concatenate rollouts
    trajs = []
    for i in range(n_particles):
        trajs.append(np.concatenate(
            [
                np.array(states[i]),
                np.array(actions[i]),
                np.array(next_states[i])
            ], axis=-1))

    return trajs, np.array(ep_rewards)

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

#%% Agent's Policy (any model-free RL algorithm. here: DDPG)

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
    def __init__(self, ds, da, h1, h2, a_max=1.):
        # self.actor = Actor(in_size=ds, h1=h1, h2=h2, nonlinearity="relu", out_size=da, max_out=a_max).to(device)
        # self.actor_target = Actor(in_size=ds, h1=h1, h2=h2, nonlinearity="relu", out_size=da, max_out=a_max).to(device)
        self.actor = Actor(in_size=ds, h1=h1, h2=h2, out_size=da, max_action=a_max).to(device)
        self.actor_target = Actor(in_size=ds, h1=h1, h2=h2, out_size=da, max_action=a_max).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = Adam(self.actor.parameters())

        # self.critic = Critic(in_size=ds+da, h1=h1, h2=h2, nonlinearity="relu").to(device)
        # self.critic_target = Critic(in_size=ds+da, h1=h1, h2=h2, nonlinearity="relu").to(device)
        self.critic = Critic(in_size=ds+da, h1=h1, h2=h2).to(device)
        self.critic_target = Critic(in_size=ds+da, h1=h1, h2=h2).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = Adam(self.critic.parameters())
    
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
            target_Q = self.critic_target(next_state, self.actor_target(next_state))
            target_Q = reward + (done * gamma * target_Q).detach()

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
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


lr_agent=0.001 #0.01 #0.001 #learning rate
h1_agent=400 #64 #400
h2_agent=300 #64 #300
gamma_agent=0.99 #discount factor
T_agent_init=1000 #number of timesteps before any updates
b_agent=1000 #100 #1000 #batch size

# policy_agent=DDPG(ds, da, h1_agent, h2_agent, a_max)

#%% Discriminator

class MLP(nn.Module):
    def __init__(self, in_size, h1, h2, out_size, out_scale=1):
        super(MLP, self).__init__()
                
        self.l1 = nn.Linear(in_size, h1)
        self.l2 = nn.Linear(h1, h2)
        self.l3 = nn.Linear(h2, out_size)
        
        self.nonlinearity= nn.Tanh()
        self.out_size=out_size
        self.out_scale=out_scale
        
        self.l3.weight.data.mul_(0.1)
        self.l3.bias.data.mul_(0.0)

    def forward(self, x, u=None):
        if u is not None:
            x=torch.cat([x, u], 1)
        x = self.nonlinearity(self.l1(x))
        x = self.nonlinearity(self.l2(x))
        x = self.out_scale * torch.sigmoid(self.l3(x)) 
        return x


class Discriminator(object):
    def __init__(self, ds, da, h, b_disc, r_disc_scale, lr_disc):
        self.discriminator = MLP(in_size=ds+da+ds, h1=h, h2=h, out_size=1).to(device) #Input: state-action-state' transition; Output: probability that it was from a reference trajectory

        self.disc_loss_func = nn.BCELoss()
        self.disc_optimizer = Adam(self.discriminator.parameters(), lr=lr_disc)
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

        return self.reward_scale * reward

    def train(self, ref_traj, rand_traj, eps):
        """Trains discriminator to distinguish between reference and randomized state action tuples"""
        for _ in range(eps):
            randind = np.random.randint(0, len(rand_traj[0]), size=int(self.batch_size))
            refind = np.random.randint(0, len(ref_traj[0]), size=int(self.batch_size))

            rand_batch = torch.from_numpy(rand_traj[randind]).float().to(device)
            ref_batch = torch.from_numpy(ref_traj[refind]).float().to(device)

            g_o = self.discriminator(rand_batch)
            e_o = self.discriminator(ref_batch)
            
            disc_loss = self.disc_loss_func(g_o, torch.ones((len(rand_batch), 1), device=device)) + self.disc_loss_func(e_o,torch.zeros((len(ref_batch), 1), device=device))

            self.disc_optimizer.zero_grad()
            disc_loss.backward()
            self.disc_optimizer.step()


r_disc_scale = 1. #reward scale
h_disc=128 #32 #128
lr_disc=0.002 #0.02 #0.002
b_disc=128
train_disc=True

# discriminator=Discriminator(ds, da, h_disc, b_disc, r_disc_scale, lr_disc)


#%% ADR Policy (or: ensemble of policies / SVPG particles)

#============================
# Adjust Torch Distributions
#============================

# Categorical
FixedCategorical = torch.distributions.Categorical

old_sample = FixedCategorical.sample
FixedCategorical.sample = lambda self: old_sample(self).unsqueeze(-1)

log_prob_cat = FixedCategorical.log_prob
FixedCategorical.log_probs = lambda self, actions: log_prob_cat(self, actions.squeeze(-1)).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

FixedCategorical.mode = lambda self: self.probs.argmax(dim=-1, keepdim=True)

class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()

        self.linear = orthogonal_init(nn.Linear(num_inputs, num_outputs),gain=0.01)

    def forward(self, x):
        x = self.linear(x)
        return FixedCategorical(logits=x)


# Normal
FixedNormal = torch.distributions.Normal

log_prob_normal = FixedNormal.log_prob
FixedNormal.log_probs = lambda self, actions: log_prob_normal(self, actions).sum(-1, keepdim=True)

normal_entropy = FixedNormal.entropy
FixedNormal.entropy = lambda self: normal_entropy(self).sum(-1)

FixedNormal.mode = lambda self: self.mean


class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias

class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DiagGaussian, self).__init__()

        self.fc_mean = orthogonal_init(nn.Linear(num_inputs, num_outputs))
        self.logstd = AddBias(torch.zeros(num_outputs))

    def forward(self, x):
        action_mean = self.fc_mean(x)

        #  An ugly hack for my KFAC implementation.
        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.cuda()

        action_logstd = self.logstd(zeros)
        return FixedNormal(action_mean, action_logstd.exp())
    
#====================
# Adjust Torch Utils
#====================

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
    # Ensure vec of type Variable
    # if not isinstance(vec, torch.cuda.FloatTensor):
    #     raise TypeError('expected torch.Tensor, but got: {}'.format(torch.typename(vec)))
    # Flag for the device where the parameter is located
    param_device = None

    # Pointer for slicing the vector for each parameter
    pointer = 0
    if grad:
        for param in parameters:
            # Ensure the parameters are located in the same device
            param_device = _check_param_device(param, param_device)
            # The length of the parameter
            num_param = torch.prod(torch.LongTensor(list(param.size())))
            param.grad.data = vec[pointer:pointer + num_param].view(param.size()) #BUG: is this the problematic inplace operation?
            # Increment the pointer
            pointer = pointer + num_param
    else:
        vec2params(vec, parameters)

#======
# SVPG
#======

class SVPGParticleCritic(nn.Module):
    def __init__(self, in_size, h):
        super(SVPGParticleCritic, self).__init__()

        self.critic = nn.Sequential(
            orthogonal_init(nn.Linear(in_size, h),np.sqrt(2)),
            nn.Tanh(),
            orthogonal_init(nn.Linear(h, h),np.sqrt(2)),
            nn.Tanh(),
            orthogonal_init(nn.Linear(h, 1),np.sqrt(2))
        )

    def forward(self, x):
        return self.critic(x)


class SVPGParticleActor(nn.Module):
    def __init__(self, in_size, h,out_size):
        super(SVPGParticleActor, self).__init__()

        self.actor = nn.Sequential(
            orthogonal_init(nn.Linear(in_size, h),np.sqrt(2)),
            nn.Tanh(),
            orthogonal_init(nn.Linear(h, h),np.sqrt(2)),
            nn.Tanh(),
            # orthogonal_init(nn.Linear(h, out_size),0.01)
        )

    def forward(self, x):
        return self.actor(x)


class SVPGParticle(nn.Module):
    """Implements a AC architecture for a Discrete A2C Policy, used inside of SVPG"""
    def __init__(self, in_size, out_size, h, type_particles, freeze=False):
        super(SVPGParticle, self).__init__()

        # self.critic = Critic(in_size=in_size, h1=h,h2=h,nonlinearity="tanh",init_func=orthogonal_init)
        # self.actor = Actor(in_size=in_size, h1=h,h2=h,nonlinearity="tanh",init_func=orthogonal_init)
        self.critic = SVPGParticleCritic(in_size=in_size, h=h)
        self.actor = SVPGParticleActor(in_size=in_size, h=h, out_size=out_size)
        self.dist = Categorical(h, out_size) if type_particles=="discrete" else DiagGaussian(h, out_size)
        # self.dist=torch.distributions.Categorical

        if freeze:
            for param in self.critic.parameters():
                param.requires_grad = False

            for param in self.actor.parameters():
                param.requires_grad = False
    
            # for param in self.dist.parameters():
            #     param.requires_grad = False

        self.reset()
        
    def reset(self):
        self.saved_log_probs = []
        self.saved_klds = []
        self.rewards = []

    def forward(self, x):
        actor = self.actor(x)
        # dist = self.dist(logits=actor)
        dist = self.dist(actor)
        value = self.critic(x)

        return dist, value        


class SVPG:
    """
    Input: current randomization settings
    Output: either a direction to move in (Discrete - for 1D/2D) or a delta across all parameters (Continuous)
    """
    def __init__(self, n_particles, dr, h, delta_max, T_svpg, T_svpg_reset, temp, kld_coeff, lr_svpg, gamma_svpg, type_particles):
        
        self.particles = []
        self.prior_particles = []
        self.optimizers = []
        
        self.delta_max = delta_max
        self.T_svpg = T_svpg
        self.T_svpg_reset = T_svpg_reset
        self.temp = temp
        self.n_particles = n_particles
        self.gamma = gamma_svpg

        self.dr = dr
        self.out_size = dr * 2 if type_particles=="discrete" else dr
        self.type_particles = type_particles
        self.kld_coeff = kld_coeff

        self.last_states = np.random.uniform(0, 1, (self.n_particles, self.dr))
        self.timesteps = np.zeros(self.n_particles)

        for i in range(self.n_particles):
            
            # Initialize each of the individual particles
            policy = SVPGParticle(in_size=self.dr, out_size=self.out_size, h=h, type_particles=type_particles).to(device)
            prior_policy = SVPGParticle(in_size=self.dr, out_size=self.out_size, h=h, type_particles=type_particles, freeze=True).to(device)
            optimizer = Adam(policy.parameters(), lr=lr_svpg)
            
            self.particles.append(policy)
            self.prior_particles.append(prior_policy)
            self.optimizers.append(optimizer)

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
        sum_k = k.sum(1)
        # grad_k_new=[]
        for i in range(X.shape[1]):
            grad_k[:, i] = grad_k[:, i] + X[:, i].matmul(sum_k)
            # grad_k_new.append(grad_k[:, i] + X[:, i].matmul(sum_k))
        # grad_k=torch.stack(grad_k_new,1)
        grad_k = grad_k / (h ** 2)

        return k, grad_k

    def select_action(self, policy_idx, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        policy = self.particles[policy_idx]
        prior_policy = self.prior_particles[policy_idx]
        dist, value = policy(state)
        prior_dist, _ = prior_policy(state)

        action = dist.sample()
        # ac = action               
            
        policy.saved_log_probs.append(dist.log_prob(action))
        policy.saved_klds.append(kl_divergence(dist, prior_dist))
        
        if self.dr == 1 or self.type_particles=="discrete":
            action = action.item()
        else:
            action = action.squeeze().cpu().detach().numpy()

        return action, value #, ac

    def compute_returns(self, next_value, rewards, masks, klds):
        R = next_value 
        returns = []
        for step in reversed(range(len(rewards))):
            # Eq. 80: https://arxiv.org/abs/1704.06440
            R = self.gamma * masks[step] * R + (rewards[step] - self.kld_coeff * klds[step])
            returns.insert(0, R)

        return returns

    def step(self):
        """Rollout trajectories, starting from random initializations of randomization settings (i.e. current_sim_params), each of T_svpg size
        Then, send it to agent for further training and reward calculation
        """
        self.simulation_instances = np.zeros((self.n_particles, self.T_svpg, self.dr))

        # Store the values of each state - for advantage estimation
        self.values = torch.zeros((self.n_particles, self.T_svpg, 1)).float().to(device)
        # self.values=[[] for _ in range(self.n_particles)]
        # Store the last states for each particle (calculating rewards)
        self.masks = np.ones((self.n_particles, self.T_svpg))

        for i in range(self.n_particles):
            self.particles[i].reset()
            current_sim_params = self.last_states[i]

            for t in range(self.T_svpg):
                self.simulation_instances[i][t] = current_sim_params

                action, value = self.select_action(i, current_sim_params)  
                self.values[i][t] = value
                # self.values[i].append(value)
                # p_grad=ac.backward()
                
                action = self._process_action(action) 
                clipped_action = action * self.delta_max
                next_params = np.clip(current_sim_params + clipped_action, 0, 1)

                if np.array_equal(next_params, current_sim_params) or self.timesteps[i] + 1 == self.T_svpg_reset:
                    next_params = np.random.uniform(0, 1, (self.dr,))
                    
                    self.masks[i][t] = 0 # done = True
                    self.timesteps[i] = 0

                current_sim_params = next_params
                self.timesteps[i] += 1

            self.last_states[i] = current_sim_params

        return np.array(self.simulation_instances)

    def train(self, simulator_rewards):
        policy_grads = []
        parameters = []

        for i in range(self.n_particles):
            policy_grad_particle = []
            
            # Calculate the value of last state - for Return Computation
            _, next_value = self.select_action(i, self.last_states[i]) 

            particle_rewards = torch.from_numpy(simulator_rewards[i]).float().to(device)
            masks = torch.from_numpy(self.masks[i]).float().to(device)
            
            # Calculate entropy-augmented returns, advantages
            returns = self.compute_returns(next_value, particle_rewards, masks, self.particles[i].saved_klds)
            returns = torch.cat(returns).detach()
            advantages = returns - self.values[i]
            # advantages = returns - torch.cat(self.values[i])

            # logprob * A = policy gradient (before backwards)
            for log_prob, advantage in zip(self.particles[i].saved_log_probs, advantages):
                policy_grad_particle.append(log_prob * advantage.detach())

            # Compute value loss, update critic
            self.optimizers[i].zero_grad()
            critic_loss = 0.5 * advantages.pow(2).mean()
            critic_loss.backward(retain_graph=True)
            # critic_loss.backward()
            self.optimizers[i].step()

            # Store policy gradients for SVPG update
            self.optimizers[i].zero_grad()
            policy_grad = -torch.cat(policy_grad_particle).mean()
            policy_grad.backward()

            # Vectorize parameters and PGs
            vec_param, vec_policy_grad = parameters_to_vector(self.particles[i].parameters(), both=True)

            policy_grads.append(vec_policy_grad.unsqueeze(0))
            parameters.append(vec_param.unsqueeze(0))

        # calculating the kernel matrix and its gradients
        parameters = torch.cat(parameters)
        k, grad_k = self.compute_kernel(parameters)

        policy_grads = 1 / self.temp * torch.cat(policy_grads)
        grad_logp = torch.mm(k, policy_grads)

        grad_theta = (grad_logp + grad_k) / self.n_particles

        # update param gradients
        for i in range(self.n_particles):
            vector_to_parameters(grad_theta[i], self.particles[i].parameters(), grad=True)
            self.optimizers[i].step()
            
    def _process_action(self, action):
        """Transform policy output into environment-action"""
        if self.type_particles=="discrete":
            if self.dr == 1:
                if action == 0:
                    action = [-1.]
                elif action == 1:
                    action = [1.]
            elif self.dr == 2:
                if action == 0:
                    action = [-1., 0]
                elif action == 1:
                    action = [1., 0]
                elif action == 2:
                    action = [0, -1.]
                elif action == 3:
                    action = [0, 1.]
        else:
            if isinstance(action, float):
                action = np.clip(action, -1, 1)
            else:
                action = action / np.linalg.norm(action, ord=2)

        return np.array(action)


n_particles=10 #3 #10
temp=10. #temperature
types_particles=["discrete","continuous"] #???: which one is better?
type_particles=types_particles[1]
kld_coeff=0. #kld = KL Divergence
T_svpg_reset=50 #25 #how often to fully reset svpg particles
delta_max=0.05 #maximum allowable change to env randomization params caused by svpg particles (If discrete, this is fixed, If continuous, this is max)
T_svpg_init=0 #number of svpg steps to take before updates
T_svpg=5 #2 #5 #length of one svpg particle rollout
lr_svpg=0.0003 #0.03 #0.0003
gamma_svpg=0.99
h_svpg=100 #16 #100

# svpg = SVPG(n_particles, dr, h_svpg, delta_max, T_svpg, T_svpg_reset, temp, kld_coeff, lr_svpg, gamma_svpg, type_particles)

#%% Implementation (ADR algorithm)
if __name__ == '__main__':
    
    env_names=['halfcheetah_custom_norm-v1','halfcheetah_custom_rand-v1','lunarlander_custom_820_rand-v0']
    env_name=env_names[-1]
    env=gym.make(env_name)
    T_env=env._max_episode_steps #task horizon / max env timesteps
    ds=env.observation_space.shape[0] #state dims
    da=env.action_space.shape[0] #action dims
    a_max=env.action_space.high[0]
    dr=env.unwrapped.randomization_space.shape[0] #N_rand (no. of randomization params)
    n_workers=n_particles
    
    env_ref=make_vec_envs(env_name, seed, n_workers)
    env_rand=make_vec_envs(env_name, seed, n_workers)
    
    policy_agent=DDPG(ds, da, h1_agent, h2_agent, a_max)
    discriminator=Discriminator(ds, da, h_disc, b_disc, r_disc_scale, lr_disc)
    svpg = SVPG(n_particles, dr, h_svpg, delta_max, T_svpg, T_svpg_reset, temp, kld_coeff, lr_svpg, gamma_svpg, type_particles)
    
    set_seed(seed)
    RB=ReplayBuffer()
    
    t_svpg=0 #SVPG timesteps
    t_agent=0
    T_agent=int(1e6) #max agent timesteps
    eps_rollout_agent=1 #number of episodes to rollout the agent for per simulation instance
    sampled_regions = [[] for _ in range(dr)]
    common_name="_ddpg_adr_torch"
    verbose=1
    
    start_time=timeit.default_timer()
    
    # with tqdm.tqdm(total=T_agent) as pbar:
    while t_agent < T_agent:
        #get sim instances from SVPG policy if current timestep is greater than the specified initial, o.w. create completely randomized env
        simulation_instances = svpg.step() if t_svpg >= T_svpg_init else -1 * np.ones((n_particles,T_svpg,dr))
        
        # Create placeholders
        rand_trajs = [[] for _ in range(n_particles)]
        ref_trajs = [[] for _ in range(n_particles)]
        rewards_disc = np.zeros(simulation_instances.shape[:2])
    
        # Reshape to work with vectorized environments
        simulation_instances = np.transpose(simulation_instances, (1, 0, 2))
        
        for t in range(T_svpg):
            t_agent_ep = 0 #agent timesteps in the current iteration/episode
            # create ref and randomized instances of the env, rollout the agent in both, and train the agent
            ref_traj, _=rollout(n_particles,env_ref,policy_agent,None,eps_rollout_agent,T_env,T_agent_init,b_agent,gamma_agent)
            env_rand.randomize(simulation_instances[t])
            rand_traj, rewards_agent =rollout(n_particles,env_rand,policy_agent,RB,eps_rollout_agent,T_env,T_agent_init,b_agent,gamma_agent,freeze_agent=False,add_noise=True)
            
            for i in range(n_particles):
                t_agent_ep += len(rand_traj[i])
                t_agent += len(rand_traj[i])
                
                #append trajs
                ref_trajs[i].append(ref_traj[i])
                rand_trajs[i].append(rand_traj[i])
                
                r_disc = discriminator.calculate_rewards(rand_trajs[i][t])
                rewards_disc[i][t]= r_disc
                
            #train discriminator (with set of all ref and rand trajs for all agents at the current svpg timestep)
            if train_disc:
                flattened_rand = [rand_trajs[i][t] for i in range(n_particles)]
                flattened_rand = np.concatenate(flattened_rand)
        
                flattened_ref= [ref_trajs[i][t] for i in range(n_particles)]
                flattened_ref = np.concatenate(flattened_ref)
                
                discriminator.train(ref_traj=flattened_ref, rand_traj=flattened_rand, eps=t_agent_ep)
                    
        #update svpg particles (ie. train their policies)
        if t_svpg >= T_svpg_init:
            svpg.train(rewards_disc)
            
            #log sampled regions only once svpg particles start training (i.e. once adr starts)
            for dim in range(dr):
                low=env.unwrapped.dimensions[dim].range_min
                high=env.unwrapped.dimensions[dim].range_max
                scaled_instances=low + (high-low) * simulation_instances[:, :, dim]
                sampled_regions[dim]=np.concatenate([sampled_regions[dim],scaled_instances.flatten()])
        
        #log progress
        if t_agent % 1 == 0:
            log_msg="Rewards Agent: {:.2f}, Rewards Disc: {:.2f}".format(rewards_agent.mean(),rewards_disc.sum(-1).mean())
            if verbose:
                print(log_msg+f" Timesteps: {t_agent} \n")
            # else:
            #     pbar.update(); pbar.set_description(desc=log_msg); pbar.refresh()
            
        #TODO: evaluate
        #TODO: save&load best running model(s)
        
        t_svpg += 1
    
    
#%% Results & Plots

    eps_step=int((T_agent-T_svpg_init)/4)
    rand_step=0.1
    # region_step=eps_step#*T_svpg*n_particles
    df2=pd.DataFrame()
    for dim, regions in enumerate(sampled_regions):
        
        region_step=int(len(regions)/4)
        
        low=env.unwrapped.dimensions[dim].range_min
        high=env.unwrapped.dimensions[dim].range_max
        
        dim_name=env.unwrapped.dimensions[dim].name
        
        # d = decimal.Decimal(str(low))
        # step_exp=d.as_tuple().exponent-1
        # step=10**step_exp

        x=np.arange(low,high+rand_step,rand_step)
        
        title=f"Sampled Regions for Randomization Dim = {dim_name} {env.rand} Over Time"
        plt.figure(figsize=(16,8))
        plt.grid(1)
        plt.hist((regions[region_step*0:region_step*1],regions[region_step*1:region_step*2],regions[region_step*2:region_step*3], regions[region_step*3:]), np.arange(min(x),max(x)+2*rand_step,rand_step), histtype='barstacked', label=[f'{eps_step*1} eps',f'{eps_step*2} eps', f'{eps_step*3} eps', f'{eps_step*4} eps'],color=["lightskyblue","blueviolet","hotpink","lightsalmon"])
        plt.xlim(min(x), max(x)+rand_step)
        plt.legend()
        plt.title(title)
        #save results
        plt.savefig(f'plots/sampled_regions_dim_{dim_name}_{env.rand}{common_name}.png')
        df2[f'Sampled_Regions_{dim_name}_{env.rand}'] = list(regions)
    
    df2.to_pickle(f"plots/sampled_regions{common_name}.pkl")
    
    
    #record elapsed time and close envs
    end_time=timeit.default_timer()
    print("Elapsed Time: {:.1f} minutes \n".format((end_time-start_time)/60.0))
    
    env.close()
    env_ref.close()
    env_rand.close()
