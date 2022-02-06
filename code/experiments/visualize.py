#%% Imports
#general
import numpy as np
import pandas as pd
import yaml
from common import PolicyNetwork, ValueNetwork, adapt

#visualize
import matplotlib.pyplot as plt
import seaborn as sns

#env
import gym
#------only for spyder IDE
for env in gym.envs.registration.registry.env_specs.copy():
     if 'custom' in env:
         print('Remove {} from registry'.format(env))
         del gym.envs.registration.registry.env_specs[env]
#------
import gym_custom

#Utils
import decimal

#ML
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

#%% Models
#DDPG
# class DDPG(tf.keras.Model):
#     def __init__(self, in_size, h1, h2, out_size, max_action):
#         super(DDPG, self).__init__()
        
#         self.actor=tf.keras.models.Sequential(layers=[
#             tf.keras.layers.Input(in_size),
#             tf.keras.layers.Dense(h1,activation="relu"),
#             tf.keras.layers.Dense(h2,activation="relu"),
#             tf.keras.layers.Dense(out_size,activation="tanh")
#             ])

#         self.max_action = max_action

#     def call(self, x):
#         x=self.max_action * self.actor(x)
#         return x

#%% Inputs

modes=["debug_mode","run_mode"]
mode=modes[0]

with open("config.yaml", 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

config=config[mode]

tr_eps=config["tr_eps"]
T_svpg_init=config["T_svpg_init"]
T_svpg=config["T_svpg"]
n_particles=config["n_particles"]

h=config["h_maml"]
gamma=config["gamma_maml"]
alpha=config["lr_maml"]
# h1_agent=config["h1_ddpg"]
# h2_agent=config["h2_ddpg"]

visualize=False
test_eps=10
test_random=True #Whether to use randomized envs in testing vs default/reference env

env_name=config["env_name"]
env=gym.make(env_name)
ds=env.observation_space.shape[0] #state dims
da=env.action_space.shape[0] #action dims
dr=env.unwrapped.randomization_space.shape[0] #N_rand (no. of randomization params)
# a_max=env.action_space.high[0]

in_size=ds
out_size=da

setting=env_name.split("_")[0]+f" {env.rand}"

value_net = ValueNetwork(in_size,gamma)

policy_maml_adr = PolicyNetwork(in_size,h,out_size) 
policy_maml_udr = PolicyNetwork(in_size,h,out_size)
policy_trpo_adr = PolicyNetwork(in_size,h,out_size) 
policy_trpo_udr = PolicyNetwork(in_size,h,out_size)
policy_trpo = PolicyNetwork(in_size,h,out_size)

policies=[policy_maml_adr,policy_maml_udr,policy_trpo_adr,policy_trpo_udr,policy_trpo]  

# policy_ddpg_adr=DDPG(ds, h1_agent, h2_agent, da, a_max)
# policy_ddpg_udr=DDPG(ds, h1_agent, h2_agent, da, a_max)
# policy_ddpg=DDPG(ds, h1_agent, h2_agent, da, a_max)

# policies=[policy_maml_adr,policy_maml_udr,policy_ddpg_adr,policy_ddpg_udr,policy_maml,policy_ddpg]

test_rewards=[[] for _ in range(len(policies))]
test_rewards_var=[[] for _ in range(len(policies))]
control_actions=[[] for _ in range(len(policies))]

#%% Training Results

dfs=[]
dfs_sr=[]
filenames=["maml_trpo_adr_tf","maml_trpo_udr_tf","trpo_adr_tf","trpo_udr_tf","trpo_tf"]
labels=["MAML + ADR", "MAML + UDR", "TRPO + ADR","TRPO + UDR","TRPO"]
keys=['Rewards_Tr','Rewards_Val','Rewards_Eval','Rewards_Disc']

for i, file_name in enumerate(filenames):
    common_name = "_"+file_name+"_"+env_name
    dfs.append(pd.read_pickle(f"plots/results{common_name}.pkl"))
    policies[i].load_weights(f"saved_models/model{common_name}")
    if "adr" in file_name:
        dfs_sr.append(pd.read_pickle(f"plots/sampled_regions{common_name}.pkl"))

#plot results
for key in keys:
    
    key_mean=key+"_Mean"
    key_std=key+"_Std"
    
    title=f"{key} ({setting})"
    plt.figure(figsize=(16,8))
    plt.grid(1)
    
    for i, df in enumerate(dfs):
        if key_mean in list(df.keys()):
            plt.plot(df[key_mean],label=labels[i])
            plt.fill_between(range(df[key_mean].size), df[key_mean]+df[key_std], df[key_mean]-df[key_std],alpha=0.2)
    
    # plt.axhline(y = env.spec.reward_threshold, color = 'r', linestyle = '--',label='Solved')
    plt.legend()
    plt.title(title)
    plt.legend(loc="upper right")
    plt.show()


# #sampling efficiency
# title=f"MAML Sampling Efficiency ({setting})"
# plt.figure(figsize=(16,8))
# plt.grid(1)
# for i, df in enumerate(dfs):
#     if "maml" in filenames[i]:
#         plt.plot(df["Total_Timesteps"],df["Rewards_Eval"],label=labels[i])
#         plt.fill_between(df["Total_Timesteps"], df["Rewards_Eval_Max"], df["Rewards_Eval_Min"],alpha=0.2)
# # plt.axhline(y = env.spec.reward_threshold, color = 'r', linestyle = '--',label='Solved')
# plt.legend()
# plt.title(title)
# plt.legend(loc="upper right")
# plt.show()


# title=f"DDPG Sampling Efficiency ({setting})"
# plt.figure(figsize=(16,8))
# plt.grid(1)
# for i, df in enumerate(dfs):
#     if "ddpg" in filenames[i]:
#         plt.plot(df["Total_Timesteps"],df["Rewards_Eval"],label=labels[i])
#         plt.fill_between(df["Total_Timesteps"], df["Rewards_Eval_Max"], df["Rewards_Eval_Min"],alpha=0.2)
# # plt.axhline(y = env.spec.reward_threshold, color = 'r', linestyle = '--',label='Solved')
# plt.legend()
# plt.title(title)
# plt.legend(loc="upper right")
# plt.show()

#plot sampled regions
eps_step=int((tr_eps-T_svpg_init)/4)
region_step=eps_step*T_svpg*n_particles
labels_sr=[label_sr for label_sr in labels if "adr" in label_sr.lower()]

for j, df_sr in enumerate(dfs_sr): 
    sampled_regions=[list(df_sr.values[:,i]) for i in range(df_sr.values.shape[-1])]
    for dim, regions in enumerate(sampled_regions):
        
        low=env.unwrapped.dimensions[dim].range_min
        high=env.unwrapped.dimensions[dim].range_max
        
        dim_name=env.unwrapped.dimensions[dim].name
        
        d = decimal.Decimal(str(low))
        step_exp=d.as_tuple().exponent-1
        step=10**step_exp
    
        x=np.arange(low,high+step,step)
        
        title=f"Sampled Regions for Randomization Dim = {dim_name} {env.rand} Over Time ({labels_sr[j]})"
        plt.figure(figsize=(16,8))
        plt.grid(1)
        plt.hist((regions[region_step*0:region_step*1],regions[region_step*1:region_step*2],regions[region_step*2:region_step*3], regions[region_step*3:]), np.arange(min(x),max(x)+2*step,step), histtype='barstacked', label=[f'{eps_step*1} eps',f'{eps_step*2} eps', f'{eps_step*3} eps', f'{eps_step*4} eps'],color=["lightskyblue","blueviolet","hotpink","lightsalmon"])
        plt.xlim(min(x), max(x)+step)
        plt.legend()
        plt.title(title)
        plt.show()

#%% Testing

rand_range=np.arange(0.0,1.1,0.1,dtype=np.float32) if test_random else range(1)

low=env.unwrapped.dimensions[0].range_min
high=env.unwrapped.dimensions[0].range_max
scaled_values=low + (high-low) * rand_range

for i, policy in enumerate(policies):
    print(f"For {labels[i]}: \n")
    for j, rand_value in enumerate(rand_range):
        rand_value_rewards=[]
        if visualize and test_random: print(f"For Rand Value = {scaled_values[j]}: \n")
        for test_ep in range(test_eps):
            if test_random: env.randomize([rand_value]*dr)
            
            s=env.reset()
            
            if "maml" in filenames[i]:
                state=tf.expand_dims(tf.convert_to_tensor(s,dtype=tf.float32),0)
                dist=policy(state,params=None)
                a=tf.squeeze(dist.sample()).numpy()
                s, r, done, _ = env.step(a)
                R = r
                if test_ep == 0 and j==0:
                    act=a[0] if a.ndim >1 else a
                    act=np.clip(act,-1.0, 1.0)
                    control_actions[i].append(act)
            else:
                done=False
                R=0
            
            while not done:
                
                state=tf.expand_dims(tf.convert_to_tensor(s,dtype=tf.float32),0)
                
                if "maml" in filenames[i]:
                    states=tf.expand_dims(state,0)
                    actions=tf.expand_dims(tf.expand_dims(tf.convert_to_tensor(a),0),0)
                    rewards=tf.expand_dims(tf.expand_dims(tf.convert_to_tensor(np.array(r),dtype=np.float32),0),0)
                    masks=tf.expand_dims(tf.expand_dims(tf.convert_to_tensor(np.array(1.0),dtype=np.float32),0),0)
                    D=[states, actions, rewards, masks]
                                        
                    theta_dash=adapt(D,value_net,policy,alpha)
                    
                    dist=policy(state,params=theta_dash)
                    a=tf.squeeze(dist.sample()).numpy()
                elif "trpo" in filenames[i]:
                    dist=policy(state)
                    a=tf.squeeze(dist.sample()).numpy()
                else:
                    a=policy(state).numpy()
                
                if test_ep == 0 and j==0:
                    act=a[0] if a.ndim >1 else a
                    act=np.clip(act,-1.0, 1.0)
                    control_actions[i].append(act)
                    
                s, r, done, _ = env.step(a)
                
                if visualize and test_ep==0: env.render()
                
                R+=r
            rand_value_rewards.append(R)
        test_rewards[i].append(np.array(rand_value_rewards).mean())
        test_rewards_var[i].append(np.array(rand_value_rewards).std())
                
    env.close()

#Plot
#rewards
title=f"Testing Rewards ({setting})"
plt.figure(figsize=(16,8))
plt.grid(1)
for i, test_reward in enumerate(test_rewards):
    plt.plot(scaled_values,test_reward,label=labels[i])
    plt.fill_between(scaled_values, np.array(test_reward) + np.array(test_rewards_var[i]), np.array(test_reward) - np.array(test_rewards_var[i]), alpha=0.2)
# plt.axhline(y = env.spec.reward_threshold, color = 'r', linestyle = '--',label='Solved')
plt.title(title)
plt.legend(loc="upper right")
plt.show()


#control actions
for i, policy in enumerate(policies):
    title=f"Control Actions for {labels[i]} ({setting})"
    plt.figure(figsize=(16,8))
    plt.grid(1)
    plt.plot(control_actions[i],label=range(da))
    plt.title(title)
    plt.legend(loc="upper right")
    plt.show()
