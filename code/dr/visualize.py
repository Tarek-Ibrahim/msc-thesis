#%% Imports
#general
import numpy as np
import pandas as pd
import os

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
from collections import OrderedDict

#ML
import tensorflow as tf
import tensorflow_probability as tfp
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

#%% Models & Functions
#MAML
class PolicyNetwork(tf.keras.Model):
    def __init__(self, in_size, h, out_size):
        super().__init__()
        
        self.w1= tf.Variable(initial_value=tf.keras.initializers.glorot_uniform()(shape=(in_size, h)), dtype=tf.float32, name='layer1/weight')
        self.b1=tf.Variable(initial_value=tf.keras.initializers.Zeros()(shape=(h,)), dtype=tf.float32, name='layer1/bias')
        
        self.w2= tf.Variable(initial_value=tf.keras.initializers.glorot_uniform()(shape=(h, h)), dtype=tf.float32, name='layer2/weight')
        self.b2=tf.Variable(initial_value=tf.keras.initializers.Zeros()(shape=(h,)), dtype=tf.float32, name='layer2/bias')
        
        self.w3= tf.Variable(initial_value=tf.keras.initializers.glorot_uniform()(shape=(h, out_size)), dtype=tf.float32, name='layer3/weight')
        self.b3=tf.Variable(initial_value=tf.keras.initializers.Zeros()(shape=(out_size,)), dtype=tf.float32, name='layer3/bias')
        
        self.logstd = tf.Variable(initial_value=np.log(1.0)* tf.keras.initializers.Ones()(shape=[1,out_size]),dtype=tf.float32,name='logstd')
        
        self.nonlinearity=tf.nn.relu
            
    def update_params(self, grads, alpha):
        named_parameters = [(param.name, param) for param in self.trainable_variables]
        new_params = OrderedDict()
        for (name,param), grad in zip(named_parameters, grads):
            new_params[name]= param - alpha * grad
        return new_params
        
    def call(self, inputs, params=None):
        if params is None:
            params=OrderedDict((param.name, param) for param in self.trainable_variables)
 
        inputs=self.nonlinearity(tf.math.add(tf.linalg.matmul(inputs,params['layer1/weight:0']),params['layer1/bias:0']))
        inputs=self.nonlinearity(tf.math.add(tf.linalg.matmul(inputs,params['layer2/weight:0']),params['layer2/bias:0']))
        mean=tf.math.add(tf.linalg.matmul(inputs,params['layer3/weight:0']),params['layer3/bias:0'])
        
        std= tf.math.exp(tf.math.maximum(params["logstd:0"], np.log(1e-6)))

        dist=tfp.distributions.Normal(mean,std)
        #dist=tfp.distributions.MultivariateNormalDiag(mean,tf.linalg.diag(std))
        #dist=tfp.distributions.Independent(tfp.distributions.Normal(mean,std),1)

        return dist
        

class ValueNetwork(tf.keras.Model):
    def __init__(self, in_size, gamma, reg_coeff=1e-5):
        super().__init__()
        
        self.reg_coeff=reg_coeff
        self.gamma=gamma
        
        self.feature_size=2*in_size + 4
        self.eye=tf.eye(self.feature_size,dtype=tf.float32)
        
        self.w=tf.Variable(initial_value=tf.keras.initializers.Zeros()(shape=[self.feature_size,1]),dtype=tf.float32,trainable=False)
        
    def fit_params(self, states, rewards):
        
        T=states.shape[0]
        b=states.shape[1]
        ones = tf.ones([T,b,1],dtype=tf.float32)
        timestep= tf.math.cumsum(ones, axis=0) / 100.
        
        reg_coeff = self.reg_coeff
        
        #create features
        features = tf.concat([states, states **2, timestep, timestep**2, timestep**3, ones],axis=2)
        features=tf.reshape(features, (-1, self.feature_size))

        #compute returns        
        G = np.zeros(b,dtype=np.float32)
        returns = np.zeros((T,b),dtype=np.float32)
        for t in range(T - 1, -1, -1):
            G = rewards[t]+self.gamma*G
            returns[t] = G
        returns = tf.reshape(returns, (-1, 1))
        
        #solve system of equations (i.e. fit) using least squares
        A = tf.linalg.matmul(tf.transpose(features), features)
        B = tf.linalg.matmul(tf.transpose(features), returns)
        for _ in range(5):
            try:
                sol=np.linalg.lstsq(A.numpy()+reg_coeff * self.eye, B.numpy(),rcond=-1)[0]                
                
                if np.any(np.isnan(sol)):
                    raise RuntimeError('NANs/Infs encountered in baseline fitting')
                
                break
            except RuntimeError:
                reg_coeff *= 10
        else:
             raise RuntimeError('Unable to find a solution')
        
        #set weights vector
        self.w.assign(sol)
        # self.w.assign(tf.transpose(sol))
        
    def call(self, states):
        
        T=states.shape[0]
        b=states.shape[1]
        ones = tf.ones([T,b,1],dtype=tf.float32)
        timestep= tf.math.cumsum(ones, axis=0) / 100.
        
        features = tf.concat([states, states **2, timestep, timestep**2, timestep**3, ones],axis=2)
        
        return tf.linalg.matmul(features,self.w)


def compute_advantages(states,rewards,value_net,gamma):
    
    T=states.shape[0]
    
    values = value_net(states)
    if len(list(values.shape))>2: values = tf.squeeze(values, axis=2)
    values = tf.pad(values,[[0, 1], [0, 0]])
    
    deltas = rewards + gamma * values[1:] - values[:-1] #delta = r + gamma * v - v' #TD error
    # advantages = tf.zeros_like(deltas, dtype=tf.float32)
    advantages = tf.TensorArray(tf.float32, *deltas.shape)
    advantage = tf.zeros_like(deltas[0], dtype=tf.float32)
    
    for t in range(T - 1, -1, -1): #reversed(range(-1,T -1 )):
        advantage = advantage * gamma + deltas[t]
        advantages = advantages.write(t, advantage)
        # advantages[t] = advantage
    
    advantages = advantages.stack()
    
    #Normalize advantages to improve: learning, numerical stability & convergence
    advantages = (advantages - tf.math.reduce_mean(advantages)) / (tf.math.reduce_std(advantages)+np.finfo(np.float32).eps)
    
    return advantages


def adapt(D,value_net,policy,alpha):
    
    #unpack
    states, actions, rewards = D
    
    value_net.fit_params(states,rewards)
    
    with tf.GradientTape() as tape:
        advantages=compute_advantages(states,rewards,value_net,value_net.gamma)
        pi=policy(states)
        log_probs=pi.log_prob(actions)
        loss=-tf.math.reduce_mean(tf.math.reduce_sum(log_probs,axis=2)*advantages) if len(list(log_probs.shape)) > 2 else -tf.math.reduce_mean(log_probs*advantages)
                
    #compute adapted params (via: GD) --> perform 1 gradient step update
    grads = tape.gradient(loss, policy.trainable_variables)
    theta_dash=policy.update_params(grads, alpha) 
    
    return theta_dash 


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

#%% General
tr_eps=7
T_svpg_init=1
T_svpg=2
n_particles=3

h=100
gamma=0.99
alpha=0.1
h1_agent=64 #100 #400 #64 #400
h2_agent=64 #100 #300 #64 #300
visualize=False
eval_eps=7

env_names=['cartpole_custom-v1', 'halfcheetah_custom-v1', 'halfcheetah_custom_norm-v1', 'halfcheetah_custom_rand-v1', 'halfcheetah_custom_rand-v2', 'lunarlander_custom_default_rand-v0']
env_name=env_names[-2]
env=gym.make(env_name)
ds=env.observation_space.shape[0] #state dims
da=env.action_space.shape[0] #action dims
dr=env.unwrapped.randomization_space.shape[0] #N_rand (no. of randomization params)
a_max=env.action_space.high[0]

in_size=ds
out_size=da

setting=env_name.split("_")[0]+f" {env.rand}"

policy_maml_adr = PolicyNetwork(in_size,h,out_size) 
policy_maml_udr = PolicyNetwork(in_size,h,out_size) 
value_net = ValueNetwork(in_size,gamma)
policy_ddpg=DDPG(ds, h1_agent, h2_agent, da, a_max)
policies=[policy_maml_adr,policy_maml_udr,policy_ddpg]

eval_rewards=[[] for _ in range(len(policies))]

#%% Plots
dfs=[]
dfs_sr=[]
filenames=["maml_adr_tf","maml_udr_tf","ddpg_adr_tf"]
labels=["MAML + ADR", "MAML + UDR", "DDPG + ADR"]
keys=['Rewards_Tr','Rewards_Val','Rewards_Eval','Rewards_Disc']

for i, file_name in enumerate(filenames):
    common_name = "_"+file_name+"_"+env_name
    dfs.append(pd.read_pickle(f"plots/results{common_name}.pkl"))
    policies[i].load_weights(f"saved_models/model{common_name}")
    if "adr" in file_name:
        dfs_sr.append(pd.read_pickle(f"plots/sampled_regions{common_name}.pkl"))

#plot results
for key in keys:
    title=f"{key} ({setting})"
    plt.figure(figsize=(16,8))
    plt.grid(1)
    
    for i, df in enumerate(dfs):
        if key in list(df.keys()):
            plt.plot(df[key],label=labels[i])
            plt.fill_between(range(tr_eps), df[key+"_Max"], df[key+"_Min"],alpha=0.2)
    
    # plt.axhline(y = env.spec.reward_threshold, color = 'r', linestyle = '--',label='Solved')
    plt.legend()
    plt.title(title)
    plt.legend(loc="upper right")
    plt.show()


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


#%% Behaviour
#Evaluate
for i, policy in enumerate(policies):
    if visualize: print(f"For {labels[i]}: \n")
    for _ in range(eval_eps):
        env.randomize(["random"]*dr)
        
        s=env.reset()
        
        if "maml" in filenames[i]:
            state=tf.expand_dims(tf.convert_to_tensor(s,dtype=tf.float32),0)
            dist=policy(state,params=None)
            a=tf.squeeze(dist.sample()).numpy()
            s, r, done, _ = env.step(a)
            R = r
        else:
            done=False
            R=0
        
        while not done:
            
            state=tf.expand_dims(tf.convert_to_tensor(s,dtype=tf.float32),0)
            
            if "maml" in filenames[i]:
                states=tf.expand_dims(state,0)
                actions=tf.expand_dims(tf.expand_dims(tf.convert_to_tensor(a),0),0)
                rewards=tf.expand_dims(tf.expand_dims(tf.convert_to_tensor(np.array(r),dtype=np.float32),0),0)
                D=[states, actions, rewards]
                                    
                theta_dash=adapt(D,value_net,policy,alpha)
                
                dist=policy(state,params=theta_dash)
                a=tf.squeeze(dist.sample())
            else:
                a=policy(state)
                
            s, r, done, _ = env.step(a.numpy())
            
            if visualize: env.render()
            
            R+=r
        eval_rewards[i].append(R)
            
    env.close()

#Plot
title=f"Testing Rewards ({setting})"
plt.figure(figsize=(16,8))
plt.grid(1)
for i, eval_reward in enumerate(eval_rewards):
    plt.plot(eval_reward,label=labels[i])
# plt.axhline(y = env.spec.reward_threshold, color = 'r', linestyle = '--',label='Solved')
plt.title(title)
plt.legend(loc="upper right")
plt.show()