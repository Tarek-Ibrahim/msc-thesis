#%% Imports
import numpy as np
import matplotlib.pyplot as plt
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
import torch.nn.functional as F
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
        
class PrioritizedReplayBuffer(object):
    def __init__(self, max_size=1e6):
        self.storage = []
        self.max_size = int(max_size)
        self.next_idx = 0
        self.alpha=0.8
        self.reset = False

    # Expects tuples of (state, next_state, action, reward, done)
    def add(self, data):
        self.reset = True
        if self.next_idx >= len(self.storage):
            self.storage.append((data,1.))
        else:
            self.storage[self.next_idx] = (data,1.)

        self.next_idx = (self.next_idx + 1) % self.max_size

    def sample(self, batch_size):
        
        if self.reset:
            data, weights = zip(*self.storage)
            self.storage=list(zip(data,list(np.array(weights)*self.alpha)))
            self.reset=False
        
        data, weights = zip(*self.storage)
        samples = random.choices(data,weights,k=batch_size)
        # ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d = [], [], [], [], []

        for sample in samples:
            X, Y, U, R, D = sample
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)
        

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
        x = self.l3(x) #self.max_action * torch.tanh(self.l3(x))
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
    def __init__(self, dr, h1, h2, T_svpg, H_svpg, gamma, epochs, batch_size, lr, xp_type, T_init, a_max=0.005, tau=0.005):
        
        self.T_svpg = T_svpg
        self.dr = dr
        self.H_svpg = H_svpg
        self.gamma = gamma
        self.last_states = np.random.uniform(0, 1, (self.dr,))
        self.timesteps = 0
        self.epochs = epochs
        self.batch_size = batch_size
        self.tau=tau
        self.xp_type=xp_type
        self.T_init = T_init
        self.a_max = a_max
        # self.RB=ReplayBuffer()
        self.RB=PrioritizedReplayBuffer()

        self.actor = Actor(in_size=dr, h1=h1, h2=h2, out_size=dr, max_action=a_max).to(device)
        self.actor_target = Actor(in_size=dr, h1=h1, h2=h2, out_size=dr, max_action=a_max).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = Adam(self.actor.parameters(),lr=lr)

        self.critic = Critic(in_size=2*dr, h1=h1, h2=h2).to(device)
        self.critic_target = Critic(in_size=2*dr, h1=h1, h2=h2).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = Adam(self.critic.parameters(),lr=lr*10.)
    
    def select_action(self, state):
        state = torch.FloatTensor(state).to(device)
        return self.actor(state).cpu().data.numpy()
    
    def step(self):
        self.simulation_instance = np.zeros((self.T_svpg, self.dr))

        #reset
        current_sim_params = self.last_states
        done=False
        add_to_buffer=True

        for t in range(self.T_svpg):
            self.simulation_instance[t] = current_sim_params

            action = self.select_action(current_sim_params) if len(self.RB.storage) > self.T_init else self.a_max * np.random.uniform(-1, 1, (self.dr,))
            
            #step
            next_params = current_sim_params + action
            if self.xp_type=="peak":
                reward = 1. if next_params <= 0.6 and next_params >= 0.4 else -1.
                # reward = 1./(np.abs(0.5-next_params)+1e-8)
            elif self.xp_type=="valley":
                reward = -1. if next_params <= 0.6 and next_params >= 0.4 else 1.
                # reward = -1./(np.abs(0.5-next_params)+1e-8)
            done=True if next_params < 0. or next_params > 1. else False
            # next_params = np.clip(next_params,0,1)
            done_bool = 0 if self.timesteps + 1 == self.H_svpg else float(done)
            
            if add_to_buffer:
                self.RB.add((current_sim_params, next_params, action, reward, done_bool))
            
            if done_bool:
                current_sim_params = np.random.uniform(0, 1, (self.dr,))                
                self.timesteps = 0
                # add_to_buffer=False
                # break
            else:
                current_sim_params = next_params
                self.timesteps += 1

        self.last_states = current_sim_params

        return np.array(self.simulation_instance), self.last_states

    def train(self):
        if len(self.RB.storage) > self.T_init: 
            for _ in range(self.epochs):
                # Sample replay buffer 
                x, y, u, r, d = self.RB.sample(self.batch_size)
                state = torch.FloatTensor(x).to(device)
                action = torch.FloatTensor(u).to(device)
                next_state = torch.FloatTensor(y).to(device)
                done = torch.FloatTensor(1 - d).to(device)
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

#%% Implementation (ADR algorithm)
if __name__ == '__main__':
    
    lr_svpg=0.003 #0.0003
    gamma_svpg=0.99
    h1=64 #32 #100 #100 #hidden sizes
    h2=64 #32 #64
    T_svpg=20 #10 #50 #ddpg rollout length
    delta_max = 1. #0.5 #0.05 #0.005 #0.05 #maximum allowable change to svpg states (i.e. upper bound on the svpg action)
    H_svpg = 100 #25 #50 #ddpg rollout horizon
    batch_size=250 #for batch used for ddpg training 
    epochs=50
    xp_types=["peak","valley"] #experiment types
    xp_type=xp_types[0]
    rewards_scale=1.
    T_init = 100 #initial steps to take before svpg update or step
    
    env_names=['halfcheetah_custom_norm-v1','halfcheetah_custom_rand-v1','lunarlander_custom_820_rand-v0','cartpole_custom-v1']
    env_name=env_names[-2]
    env=gym.make(env_name)
    T_env=env._max_episode_steps #task horizon / max env timesteps
    ds=env.observation_space.shape[0] #state dims
    da=env.action_space.shape[0] #action dims
    dr=env.unwrapped.randomization_space.shape[0]
        
    svpg = DDPG(dr, h1, h2, T_svpg, H_svpg, gamma_svpg, epochs, batch_size, lr_svpg, xp_type, T_init, a_max=delta_max)
    
    set_seed(seed)
    
    T_eps=1000 #number of training episodes
    plot_tr_rewards_mean=[]
    sampled_regions = [[] for _ in range(dr)]
    rand_step=0.1 #for discretizing the sampled regions plot
    common_name="_ddpg_dr"
    t_eps=0
    
    with tqdm.tqdm(total=T_eps) as pbar:
        while t_eps < T_eps:
            
            #collect rollout batch with svpg particles (storing values)
            simulation_instances, next_instances = svpg.step()
            
            #calculate deterministic reward
            simulation_instances_mask = np.concatenate([simulation_instances[1:,0],next_instances])
            rewards = np.ones_like(simulation_instances_mask,dtype=np.float32)
            if xp_type =="peak":
                rewards[((simulation_instances_mask<=0.40).astype(int) + (simulation_instances_mask>=0.60).astype(int)).astype(bool)]=-1.
                # rewards *= 1./(np.abs(0.5-simulation_instances_mask)+1e-8)
            elif xp_type=="valley":
                rewards[((simulation_instances_mask>=0.40).astype(int) * (simulation_instances_mask<=0.60).astype(int)).astype(bool)]=-1.
                # rewards *= - 1./(np.abs(0.5-simulation_instances_mask)+1e-8)
                
            rewards = rewards * rewards_scale
                
            mean_rewards=rewards.sum(-1).mean()
            plot_tr_rewards_mean.append(mean_rewards)
            
            #train svpg 
            svpg.train()
            
            #plot sampled regions
            for dim in range(dr):
                dim_name=env.unwrapped.dimensions[dim].name
                low=env.unwrapped.dimensions[dim].range_min
                high=env.unwrapped.dimensions[dim].range_max
                x=np.arange(low,high+rand_step,rand_step)
                linspace_x=np.arange(min(x),max(x)+2*rand_step,rand_step)
                
                scaled_instances=low + (high-low) * simulation_instances[:, dim]
                sampled_regions[dim]=np.concatenate([sampled_regions[dim],scaled_instances.flatten()])
                  
                title=f"Sampled Regions for Randomization Dim = {dim_name} {env.rand} at Episode = {t_eps}"
                plt.figure(figsize=(16,8))
                plt.grid(1)
                plt.hist(sampled_regions[dim], linspace_x, histtype='barstacked')
                plt.xlim(min(x), max(x)+rand_step)
                plt.title(title)
                plt.savefig(f'plots/sampled_regions_dim_{dim_name}_{env.rand}{common_name}.png')
                plt.close()
                
                title=f"Value Function for Randomization Dim = {dim_name} {env.rand} at Episode = {t_eps}"
                plt.figure(figsize=(16,8))
                plt.grid(1)
                ls=np.linspace(0,1,len(linspace_x))
                a=svpg.actor(torch.from_numpy(ls).unsqueeze(1).float().to(device))
                v=svpg.critic(torch.from_numpy(ls).unsqueeze(1).float().to(device),a)
                plt.plot(linspace_x,v.detach().cpu().numpy())
                plt.xlim(min(x), max(x)+rand_step)
                plt.title(title)
                plt.savefig(f'plots/value_function_dim_{dim_name}_{env.rand}{common_name}.png')
                plt.close()
            
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
