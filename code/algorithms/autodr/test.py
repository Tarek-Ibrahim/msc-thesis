#%% Imports
import numpy as np
import matplotlib.pyplot as plt
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
from torch.distributions import Normal, Independent, MultivariateNormal 

#%% General
torch.cuda.empty_cache()
device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device="cpu"
torch.set_default_tensor_type(torch.FloatTensor)
torch.autograd.set_detect_anomaly(True)

#%% Funcs

def set_seed(seed):
    import random
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

#%% Model
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


#%% Test

env_names=['halfcheetah_custom_rand-v2','halfcheetah_custom_rand-v1','lunarlander_custom_820_rand-v0','cartpole_custom-v1','hopper_custom_rand-v2']
env_name=env_names[2]
env=gym.make(env_name)
T_env=env._max_episode_steps #task horizon / max env timesteps
ds=env.observation_space.shape[0] #state dims
da=env.action_space.shape[0] #action dims
dr=env.unwrapped.randomization_space.shape[0]

seed = 1
set_seed(seed)

h=128 #64 #100
common_name="_autodr"

policy=Actor(ds,h,da).to(device)
policy.load_state_dict(torch.load(f"saved_models/model{common_name}.pt"))
policy.eval()

visualize=True
test_eps=3
test_random=True
test_reward=[]


rand_range=np.arange(0.0,1.1,0.1,dtype=np.float32) if test_random else range(1)

low=env.unwrapped.dimensions[0].range_min
high=env.unwrapped.dimensions[0].range_max
scaled_values=low + (high-low) * rand_range


for j, rand_value in enumerate(rand_range):
    rand_value_rewards=[]
    if visualize and test_random: print(f"For Rand Value = {scaled_values[j]}: \n")
    for test_ep in range(test_eps):
        if test_random: env.randomize([rand_value]*dr)
        
        s=env.reset()
        
        done=False
        R=0
        
        while not done:
            
            state=torch.from_numpy(s).float().to(device)
            dist=policy(state)
            a=dist.sample().squeeze().cpu().numpy()
                
            s, r, done, _ = env.step(a)
            
            if visualize and test_ep==0: env.render()
            
            R+=r
        rand_value_rewards.append(R)
    test_reward.append(np.array(rand_value_rewards).mean())
    # test_rewards_var[i].append(np.array(rand_value_rewards).std())
            
env.close()

#Plot
#rewards
title="Testing Rewards"
plt.figure(figsize=(16,8))
plt.grid(1)
plt.plot(scaled_values,test_reward)
    # plt.fill_between(scaled_values, np.array(test_reward) + np.array(test_rewards_var[i]), np.array(test_reward) - np.array(test_rewards_var[i]), alpha=0.2)
# plt.axhline(y = env.spec.reward_threshold, color = 'r', linestyle = '--',label='Solved')
plt.title(title)
# plt.legend(loc="upper right")
plt.show()
