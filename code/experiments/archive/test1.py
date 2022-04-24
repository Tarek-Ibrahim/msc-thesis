
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
import gym_custom

#Utils
import decimal

#ML
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

#%% Models
#DDPG
class DDPG(tf.keras.Model):
    def __init__(self, in_size, h1, h2, out_size, max_action):
        super(DDPG, self).__init__()
        
        self.actor=tf.keras.models.Sequential(layers=[
            tf.keras.layers.Input(in_size),
            tf.keras.layers.Dense(h1,activation="relu"),
            tf.keras.layers.Dense(h2,activation="relu"),
            tf.keras.layers.Dense(out_size,activation="tanh")
            ])

        self.max_action = max_action

    def call(self, x):
        x=self.max_action * self.actor(x)
        return x

#%% Inputs

modes=["debug_mode","run_mode"]
mode=modes[1]

with open("config.yaml", 'r') as f:
    config_file = yaml.load(f, Loader=yaml.FullLoader)

config=config_file[mode]

tr_eps=config["tr_eps"]
T_svpg_init=config["T_svpg_init"]
T_svpg=config["T_svpg"]
n_particles=config["n_particles"]

h=config["h_maml"] #256
gamma=config["gamma_maml"]
alpha=config["lr_maml"]
h1_agent=config["h1_ddpg"]
h2_agent=config["h2_ddpg"]

visualize=True
test_eps=3
test_random=True #Whether to use randomized envs in testing vs default/reference env

env_key='lunarlander' #'hopper_friction' #'halfcheetah_friction' #'hopper_friction'
env_name=config_file["env_names"][env_key] #config["env_name"]
env=gym.make(env_name)
ds=env.observation_space.shape[0] #state dims
da=env.action_space.shape[0] #action dims
dr=env.unwrapped.randomization_space.shape[0] #N_rand (no. of randomization params)
a_max=env.action_space.high[0]

in_size=ds
out_size=da

setting=env_name.split("_")[0]+f" {env.rand}"

value_net = ValueNetwork(in_size,gamma)

policy_maml = PolicyNetwork(in_size,h,out_size)
# policy_ddpg=DDPG(ds, h1_agent, h2_agent, da, a_max)
policies=[policy_maml]
# policies=[policy_ddpg]
test_rewards=[[] for _ in range(len(policies))]
control_actions=[[] for _ in range(len(policies))]
filenames=["trpo_udr"]
labels=["TRPO + UDR"]

for i, file_name in enumerate(filenames):
    common_name = "_"+file_name+"_"+env_key
    policies[i].load_weights(f"saved_models/model{common_name}")
    
#%% Testing

for i, policy in enumerate(policies):
    if visualize: print(f"For {labels[i]}: \n")
    for test_ep in range(test_eps):
        if test_random: env.randomize([0.0]*dr); print(env.unwrapped.dimensions[0].current_value)
        
        s=env.reset()
        
        if "maml" in filenames[i]:
            state=tf.expand_dims(tf.convert_to_tensor(s,dtype=tf.float32),0)
            dist=policy(state,params=None)
            a=tf.squeeze(dist.sample()).numpy()
            s, r, done, _ = env.step(a)
            R = r
            if test_ep == 0:
                act=a[0] if a.ndim >1 else a
                # act=np.clip(act,-1.0, 1.0)
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
            
            if test_ep == 0:
                act=a[0] if a.ndim >1 else a
                # act=np.clip(act,-1.0, 1.0)
                control_actions[i].append(act)
                
            s, r, done, _ = env.step(a)
            
            if visualize: env.render()
            
            R+=r
        test_rewards[i].append(R)
            
    env.close()

#Plot
#rewards
title=f"Testing Rewards ({setting})"
plt.figure(figsize=(16,8))
plt.grid(1)
for i, test_reward in enumerate(test_rewards):
    plt.plot(test_reward,label=labels[i])
# plt.axhline(y = env.spec.reward_threshold, color = 'r', linestyle = '--',label='Solved')
plt.title(title)
plt.legend(loc="upper right")
plt.show()


#control actions
# for i, policy in enumerate(policies):
#     title=f"Control Actions for {labels[i]} ({setting})"
#     plt.figure(figsize=(16,8))
#     plt.grid(1)
#     plt.plot(control_actions[i],label=range(da))
#     plt.title(title)
#     plt.legend(loc="upper right")
#     plt.show()
