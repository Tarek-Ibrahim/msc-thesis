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
    def __init__(self,dr,h,lr_q,lr_pi,a_max,a_min,batch_size,epochs,T_init,tau,alpha,gamma,lr_alpha,delta_max,T_svpg,H_svpg,xp_type,entropy_tuning=False,temp_anneal=False):
        
        self.batch_size=batch_size
        self.epochs=epochs
        self.T_init=T_init
        self.tau=tau
        self.alpha=alpha
        self.gamma=gamma
        self.delta_max=delta_max
        self.T_svpg = T_svpg
        self.dr = dr
        self.H_svpg = H_svpg
        self.xp_type=xp_type
        self.temp_anneal=temp_anneal
        self.alpha_min=0.0001
        self.alpha_discount=0.1
        
        self.last_states = np.random.uniform(0, 1, (self.dr,))
        self.timesteps = 0
        
        #Q-Functions
        self.q1=MLP(in_size=2*dr,h=h,out_size=1).to(device)
        self.q2=MLP(in_size=2*dr,h=h,out_size=1).to(device)
        
        self.q1_target=MLP(in_size=2*dr,h=h,out_size=1).to(device)
        self.q2_target=MLP(in_size=2*dr,h=h,out_size=1).to(device)
        
        for target_param, param in zip(self.q1_target.parameters(), self.q1.parameters()):
            target_param.data.copy_(param.data)
        
        for target_param, param in zip(self.q2_target.parameters(), self.q2.parameters()):
            target_param.data.copy_(param.data)
        
        self.q1_optimizer = Adam(self.q1.parameters(),lr=lr_q)
        self.q2_optimizer = Adam(self.q2.parameters(),lr=lr_q)
        
        #Policy
        self.pi=Policy(dr, h, dr, a_max, a_min).to(device)
        self.pi_optimizer=Adam(self.pi.parameters(), lr=lr_pi)
        
        #Temperature / Entropy (Automatic Tuning)
        if entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor(da).to(device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = Adam([self.log_alpha], lr=lr_alpha)
        
    def select_action(self, state):
        state = torch.FloatTensor(state).to(device).unsqueeze(0)
        action, _, _ = self.pi(state)
        return action.detach().cpu().numpy()[0]
    
    def step(self,RB):
        self.simulation_instance = np.zeros((self.T_svpg, self.dr))

        #reset
        current_sim_params = self.last_states
        done=False
        add_to_buffer=True

        for t in range(self.T_svpg):
            self.simulation_instance[t] = current_sim_params

            action = self.select_action(current_sim_params) if len(RB.storage) > self.T_init else self.delta_max * np.random.uniform(-1, 1, (self.dr,))
            
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
                RB.add((current_sim_params, next_params, action, reward, done_bool))
            
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
    
    def train(self,RB):
        if len(RB.storage) > self.T_init:
            if self.temp_anneal:
                self.alpha *= self.alpha_discount
                if self.alpha < self.alpha_min:
                    self.alpha = self.alpha_min
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
                
                if entropy_tuning:
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
    env_name=env_names[-2]
    env=gym.make(env_name)
    T_env=env._max_episode_steps #task horizon / max env timesteps
    ds=env.observation_space.shape[0] #state dims
    da=env.action_space.shape[0] #action dims
    dr=env.unwrapped.randomization_space.shape[0]
    
    h=64
    lr_q=0.03 #3e-4
    lr_pi=0.003 #3e-4
    lr_alpha=0.003 #3e-4
    delta_max=1.
    a_max=np.array([delta_max])
    a_min=np.array([-delta_max])
    gamma=0.99
    alpha=0.1 #0.0001 #temperature
    tau=0.005
    epochs=30
    batch_size=256
    T_init=100
    
    T_svpg=100
    H_svpg=100
    xp_types=["peak","valley"] #experiment types
    xp_type=xp_types[0]
    rewards_scale=1.
    entropy_tuning=False
    temp_anneal=True
    
    policy=SAC(dr,h,lr_q,lr_pi,a_max,a_min,batch_size,epochs,T_init,tau,alpha,gamma,lr_alpha,delta_max,T_svpg,H_svpg,xp_type,entropy_tuning,temp_anneal)
    
    set_seed(seed)
    RB=ReplayBuffer()
    
    T_eps=1000 #number of training episodes
    plot_tr_rewards_mean=[]
    common_name="_sac_dr"
    sampled_regions = [[] for _ in range(dr)]
    rand_step=0.1 #for discretizing the sampled regions plot
    t_eps=0
    
    with tqdm.tqdm(total=T_eps) as pbar:
       while t_eps < T_eps:
           
           #collect rollout batch with svpg particles (storing values)
            simulation_instances, next_instances = policy.step(RB)
            
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
            policy.train(RB)
            
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
                a,_,_=policy.pi(torch.from_numpy(ls).unsqueeze(1).float().to(device))
                v1=policy.q1(torch.from_numpy(ls).unsqueeze(1).float().to(device),a)
                v2=policy.q2(torch.from_numpy(ls).unsqueeze(1).float().to(device),a)
                plt.plot(linspace_x,v1.detach().cpu().numpy())
                plt.plot(linspace_x,v2.detach().cpu().numpy())
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
    