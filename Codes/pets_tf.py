# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 16:56:16 2021

@author: TIB001
"""
#TODO: unify all ac/act and ob/obs
#TODO: debug to make it work
#TODO: run with different reward functions and env parameters
#TODO: try with different env than cartpole (e.g. half-cheetah)
#TODO: repeat each experiment with different random seeds (for K trials?), and report the mean and standard deviation of the cost for each condition
#TODO: add more comments/documentation

# %% Imports
import numpy as np
import gym
#------only for spyder IDE
for env in gym.envs.registration.registry.env_specs.copy():
      if 'cartpole_custom' in env:
        print('Remove {} from registry'.format(env))
        del gym.envs.registration.registry.env_specs[env]
#------
import gym_custom
import matplotlib.pyplot as plt
import tqdm
from scipy.stats import truncnorm

import tensorflow as tf
# import keras



#%% Functions

progress=lambda x: tqdm.trange(x, leave=True) #for visualizing/monitoring training progress
# swish = lambda x: x * torch.sigmoid(x)

def set_seed(seed,env,det=True):
    import random
    
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    env.seed(seed)


def collect_rollout(env,policy):
    #sample a rollout from the agent [Rollout length= len(A)]

    T=env._max_episode_steps #task horizon
    O, A, rewards= [env.reset()], [], []

    policy.reset() #policy is MPC #amounts to resetting CEM optimizer's prev sol to its initial value (-> array of size H with all values = avg of action space value range)
    for t in range(T):
        a=policy.act(O[t]) #first optimal action in sequence (initially random: policy.act= np.random.uniform(ac_lb, ac_ub, ac_lb.shape))

        obs, r, done, _ = env.step(a) #execute first action from optimal actions
        
        A.append(a)
        O.append(obs)
        rewards.append(r)
        
        if done:
            break
    
    return np.array(O), np.array(A), np.array(rewards), sum(rewards)


#model
class PE():
    #Probabilistic Ensemble: ensemble of B-many bootsrapped probabilistic NNs (i.e. output parameters of a prob distribution) used to approximate a dynamics model function:
    # a- ‘probabilistic networks[/network models/network dynamics models]’: to capture aleatoric uncertainty (inherent system stochasticity) [through each model encoding a distribution (as opposed to a point estimate/prediction)]
    # b- ‘ensembles’: to capture epistemic uncertainty (subjective uncertainty, due to limited data --> isolating it is especially useful for directing exploration [out of scope])
    
    def __init__(self, B, in_size, n, h, out_size):
        # super().__init__()
        
        self.B=B
        self.n=n
        self.in_size=in_size
        self.out_size=out_size
        
        self.max_logvar = tf.Variable(0.5 * tf.ones([1, out_size // 2]))
        self.min_logvar = tf.Variable(-10.0 * tf.ones([1, out_size // 2]))
        
        self.mu = tf.Variable(initial_value=tf.keras.initializers.Zeros()(shape=[1,self.in_size]), trainable=False)
        self.sigma = tf.Variable(initial_value=tf.keras.initializers.Ones()(shape=[1,self.in_size]), trainable=False)
        
        # self.w, self.b = [], []
        # for l in range(n+1):
        #     ip=in_size if l==0 else h
        #     op=out_size if l==n else h
        #     w, b = self.initialize(ip,op)
        #     self.w.append(w)
        #     self.b.append(b)
        self.w0,self.b0=self.initialize(in_size,h)
        self.w1,self.b1=self.initialize(h,h)
        self.w2,self.b2=self.initialize(h,h)
        self.w3,self.b3=self.initialize(h,out_size)
        
        self.opt_vars=[self.w0,self.b0,self.w1,self.b1,self.w2,self.b2,self.w3,self.b3,self.max_logvar,self.min_logvar]
        
    def initialize(self,in_size,out_size):
        #truncated normal for weights (i.e. draw from normal distribution with samples outside 2*std from mean discarded and resampled)
        #zeros for biases
        
        mu=0.0
        std=1.0/(2.0*np.sqrt(in_size))
        
        w=tf.Variable(initial_value=tf.keras.initializers.TruncatedNormal(mean=mu,stddev=std)(shape=[self.B,in_size,out_size]))
        
        b=tf.Variable(initial_value=tf.keras.initializers.Zeros()(shape=[self.B, 1, out_size]))
        
        return w, b
    
    def fit_input_stats(self, inputs):
        #get mu and sigma of [all] input data fpr later normalization of model [batch] inputs
        
        # self.mu = nn.Parameter(torch.zeros(self.in_size), requires_grad=False)
        # self.sigma = nn.Parameter(torch.zeros(self.in_size), requires_grad=False)

        mu = np.mean(inputs, axis=0, keepdims=True) #over cols (each observation/action col of input) and keeping same col size --> result has size = (1,input_size)
        sigma = np.std(inputs, axis=0, keepdims=True)
        sigma[sigma < 1e-12] = 1.0 #???: why 1 (and not 1e-12 e.g.)??

        self.mu.assign(mu)
        self.sigma.assign(sigma)
    
    def compute_decays(self):
        #returns decays.sum() #decays[layer] = decay_coeffs[layer]*MSE(w[layer]) #decay_coeffs=[input=0.0001, h=0.00025, output=0.0005]
        
        # decays, decays_coeffs = [], [0.0001,0.00025,0.00025,0.0005]
        # for l in range(self.n+1):
        #     decay=decays_coeffs[l]*(self.w[l]**2).sum() / 2.0
        #     decays.append(decay)
        # return sum(decays)
        
        # lin0_decays = 0.0001 * (self.w[0] ** 2).sum() / 2.0
        # lin1_decays = 0.00025 * (self.w[1] ** 2).sum() / 2.0
        # lin2_decays = 0.00025 * (self.w[2] ** 2).sum() / 2.0
        # lin3_decays = 0.0005 * (self.w[3] ** 2).sum() / 2.0
        
        lin0_decays = tf.multiply(0.0001 , tf.nn.l2_loss(self.w0))
        lin1_decays = tf.multiply(0.00025, tf.nn.l2_loss(self.w1))
        lin2_decays = tf.multiply(0.00025 , tf.nn.l2_loss(self.w2))
        lin3_decays = tf.multiply(0.0005 , tf.nn.l2_loss(self.w3))

        return lin0_decays + lin1_decays + lin2_decays + lin3_decays

    
    def __call__(self,inputs):
        # input is 3D: [B,batch_size,input_size] --> input is a function of current observation/state
        # output is 3D: [B, batch_size, output_size] --> output size is obs/target_size * 2 (first half is for expectation/mu/mean of [Delta_]obs distribution and second half is for log(variance) of it) --> ∆s_t+1 = f (s_t; a_t) such that s_t+1 = s_t + ∆s_t+1
        
        #normalize inputs
        inputs = (inputs - self.mu) / self.sigma
        
        inputs = tf.matmul(inputs,self.w0) + self.b0
        inputs = tf.keras.activations.swish(inputs) #swish(inputs)
        inputs = tf.matmul(inputs,self.w1) + self.b1
        inputs = tf.keras.activations.swish(inputs) #swish(inputs)
        inputs = tf.matmul(inputs,self.w2) + self.b2
        inputs = tf.keras.activations.swish(inputs) #swish(inputs)
        inputs = tf.matmul(inputs,self.w3) + self.b3
       
        # #fwd pass
        # for l in range(self.n+1):
        #     inputs = inputs.matmul(self.w[l]) + self.b[l] #after size (till before last layer) = [B,input samples,h]; after NN size=[B,input samples,out_size]
        #     if l<self.n: #skips for last iteration/layer
        #         inputs = nn.SiLU()(inputs) #swish(inputs)
        
        #extract mean and log(var) from network output
        mean = inputs[:, :, :self.out_size // 2]
        logvar = inputs[:, :, self.out_size // 2:]
        
        #bounding variance (becase network gives arbitrary variance for OOD points --> could lead to numerical problems)
        logvar = self.max_logvar - tf.nn.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + tf.nn.softplus(logvar - self.min_logvar)
        
        return mean, logvar


#planner      
class MPC:
    def __init__(self,env,model,optimizer,H,p,pop_size,opt_max_iters,epochs):
        
        self.H=H
        self.p=p
        self.pop_size=pop_size
        self.opt_max_iters=opt_max_iters
        self.model=model
        self.epochs=epochs
        self.optimizer=optimizer
        
        self.ds=env.observation_space.shape[0] #state dims
        self.da=env.action_space.shape[0] #action dims
        self.initial=True
        self.ac_lb= env.action_space.low #env.ac_lb
        self.ac_ub= env.action_space.high #env.ac_ub
        self.cost_obs= env.cost_o
        self.cost_act= env.cost_a
        self.reset() #sol's initial mu/mean
        self.init_var= np.tile(((self.ac_ub - self.ac_lb) / 4)**2, [self.H]) #sol's intial variance
        self.act_buff=np.empty((0,self.da))
        self.inputs=np.empty((0,self.model.in_size))
        self.targets=np.empty((0,self.ds))
        
        
    def reset(self):
        self.prev_sol = np.tile((self.ac_lb + self.ac_ub) / 2, [self.H])
    
    
    def train(self,rollout,b): #Train the policy with rollouts
        
        obs = rollout[0]
        acs = rollout[1]
        self.initial=False
        
        #1- Construct model training inputs
        inputs=np.concatenate([obs[:-1], acs], axis=-1)
        self.inputs=np.concatenate([self.inputs,inputs])
        
        #2- Construct model training targets
        targets = obs[1:] - obs[:-1]
        self.targets=np.concatenate([self.targets,targets])
        
        #3- Train the model
            
        #get mean & var of tr inputs
        self.model.fit_input_stats(self.inputs)
        #create random_idxs from 0 to (no. of tr samples - 1) with size = [B,no. of tr samples]
        idxs = np.random.randint(self.inputs.shape[0], size=[self.model.B, self.inputs.shape[0]])
        num_batches = int(np.ceil(idxs.shape[-1] / b)) #(no. of batches=roundup(no. of [model] training input examples so far / batch size))
        for _ in range(epochs): #for each epoch
            for batch_num in range(num_batches): # for each batch 
                # choose a batch from tr and target inputs randomly (i.e. pick random entries/rows/samples from inputs to construct a batch, via: input[random_idxs[:,batch_idxs]]) --> batch_idxs change with each inner iteration as a function of current batch no. and b, while rest stay constant; this also inserts an additional dimension at the beginning to inputs = B; random_idxs used for each net are shuffled row-wise with each outer loop; idxs are reset w/ every call to train func [i.e. each tr_ep] (which would also have different no. of tr samples)
                batch_idxs = idxs[:, batch_num * b : (batch_num + 1) * b]
                inputs = tf.convert_to_tensor(self.inputs[batch_idxs],dtype=tf.float32)
                targets = tf.convert_to_tensor(self.targets[batch_idxs],dtype=tf.float32)
                # inputs=self.inputs[batch_idxs]
                # targets=self.targets[batch_idxs]
                # Operate on batches:
                with tf.GradientTape() as tape:
                    mean, logvar = self.model(inputs) #fwd pass
                    var = tf.math.exp(-logvar)
                    # Calculate grad, loss & backpropagate
                    loss = tf.math.reduce_mean(tf.math.reduce_mean(((mean - targets)**2) * var + logvar, axis=-1), axis=-1) #MSE losses #???: why does mean over target dimension make sense?
                    loss += tf.math.reduce_mean(tf.math.reduce_mean(logvar, axis=-1), axis=-1) #var losses
                    # loss += 0.01 * (tf.math.reduce_sum(self.model.max_logvar) - tf.math.reduce_sum(self.model.min_logvar)) # a constant
                    # loss += self.model.compute_decays() 
                
                gradients = tape.gradient(loss, self.model.opt_vars)
                self.optimizer.apply_gradients(zip(gradients, self.model.opt_vars))

                
                # self.optimizer.minimize(loss,self.model.opt_vars,tape=self.model.tape)
                # self.optimizer.minimize(loss,self.model.opt_vars)
                
            # shuffle idxs
            idxs_of_idxs = np.argsort(np.random.uniform(size=idxs.shape), axis=-1)
            idxs = idxs[np.arange(idxs.shape[0])[:, None], idxs_of_idxs] #shuffles indicies of each row of idxs randomly (i.e. acc. to idxs_of_idxs)
        
    
    def act(self,obs):
        
        if self.initial:
            action = np.random.uniform(self.ac_lb, self.ac_ub, self.ac_lb.shape)
            return action
        else:
            if self.act_buff.shape[0] > 0:
                action, self.act_buff = self.act_buff[0], self.act_buff[1:] #pop out the optimal action from buffer
                return action
            sol=self.CEM(obs) #get CEM optimizer's sol
            
            self.prev_sol=np.concatenate([np.copy(sol)[self.da:], np.zeros(self.da)]) #update prev sol --> take out first action in sol and pad leftover sequence with trailing zeros to maintain same sol/prev sol shape #???: how is this correct??
            self.act_buff = sol[:self.da].reshape(-1, self.da) #has the first action in the sequence = optimal action
            return self.act(obs)
            
            # while self.act_buff.shape[0] == 0:
            #     sol=self.CEM(obs) #get CEM optimizer's sol
            #     self.prev_sol=np.concatenate([np.copy(sol)[self.da:], np.zeros(self.da)]) #update prev sol --> take out first action in sol and pad leftover sequence with trailing zeros to maintain same sol/prev sol shape
            #     self.act_buff = sol[:self.da].reshape(-1, self.da) #has the first action in the sequence = optimal action
            # action, self.act_buff = self.act_buff[0], self.act_buff[1:] #pop out the optimal action from buffer
        
        # return action
        
    
    def CEM(self,obs): #MPC's optimizer: an action sequence optimizer
        #solution = action sequence = sample
        #population-based method (samples candidate solutions from a distribution then evaluates them and constructs next distribution with best sols from prev one [acc to the defined cost function])
        
        #action sequence is taken to be optimized once per iteration (i.e. per=1)
        sol_dim=self.H*self.da #dimension of an action sequence
        epsilon=0.001 #termination condition representing min variance allowable
        alpha=0.25 #0.1 #controls how much of mean & variance is used for next iteration
        lb=np.tile(self.ac_lb,[self.H])
        ub=np.tile(self.ac_ub,[self.H])
        
        mean, var, t = self.prev_sol, self.init_var, 0
        X = truncnorm(-2, 2, loc=np.zeros_like(mean), scale=np.ones_like(var))

        while (t < self.opt_max_iters) and np.max(var) > epsilon:
            lb_dist = mean - lb
            ub_dist = ub - mean
            constrained_var = np.minimum(np.minimum((lb_dist / 2)**2, (ub_dist / 2)**2), var)
            
            #1- generate sols
            samples = X.rvs(size=[self.pop_size, sol_dim]) * np.sqrt(constrained_var) + mean #sample from destandardized/denormalized distributiion
            samples = samples.astype(np.float32)

            #2- propagate state particles & evaluate actions
            costs = self.get_costs(obs,samples)

            #3- update CEM distribution
            
            elites = samples[np.argsort(costs)][:self.pop_size//10] #first num_elites (here: =10% of population, for simplicity) items of samples after sorting them according to their costs (in ascending order) --> i.e. num_elites samples with lowest costs according to defined cost func
            # num_elites = CEM's number of top/best solutions (out of popsize of candidate sols) thet will be used to update the CEM distrib (i.e. obtain the distribution at the next iteration)
            
            #update distribution params
            mean_elites = np.mean(elites, axis=0)
            var_elites = np.var(elites, axis=0)
            #get portion of mu/mean and sigma/var used for next iteration 
            mean = alpha * mean + (1 - alpha) * mean_elites
            var = alpha * var + (1 - alpha) * var_elites

            t += 1

        return mean
    
    
    def get_costs(self,obs,ac_seqs):
        
        dim=tf.shape(ac_seqs)[0]
        
        #reshape ac_seqs --> from (pop_size,sol_dim[H*da]) to (H,pop_size*p,da) --> pop_size*p = expand each candidate sol to a set of p particles
        ac_seqs = tf.transpose(tf.reshape(ac_seqs, [-1, self.H, self.da]),[1, 0, 2])[:, :, None]
        ac_seqs = tf.reshape(tf.tile(ac_seqs,[1, 1, self.p, 1]), [self.H, -1, self.da])
        
        #reshape obs --> from (ds) to (pop_size*p,ds)
        obs=tf.tile(obs[None].astype(np.float32), [dim * self.p, 1])

        costs = tf.zeros([dim, self.p]) #initialize costs ; size=[pop_size,p] #sum of costs over the planning horizon
        
        for t in range(self.H):
            curr_acs=ac_seqs[t]
            obs_next = self.TS(obs,curr_acs) #propagate state particles using the PE dynamics model (aka: predict next observations)
            
            cost=self.cost_obs(obs_next)+self.cost_act(curr_acs) #evaluate actions
            cost = tf.reshape(cost,[-1, self.p]) #reshape costs to (pop_size,p)
            costs += cost
            obs = obs_next
        
        costs=tf.where(tf.math.is_nan(costs), 1e6 * tf.ones_like(costs), costs) #replace NaNs with a high cost 
        
        return tf.math.reduce_mean(costs,axis=1) #mean of costs # dim is dim to reduce (i.e. for dim=1, it will take the mean of each row (dim=0)) #i.e. here we average over the particles of each sol
    
    
    def TS(self,obs,curr_acs): #trajectory sampling: propagate state particles (aka: predict next observations)
        #implements TSinf
        
        #reshape obs  #(pop_size*p,ds) --> (B,pop_size*p/B,ds) #i.e. divide the pop_size*p observations among B networks
        dim=tf.shape(obs)[-1]
        obs_reshaped=tf.reshape(tf.transpose(tf.reshape(obs, [-1, self.model.B, self.p // self.model.B, dim]),[1, 0, 2, 3]),[self.model.B, -1, dim])

        #reshape curr_acs  #(1,pop_size*p,da) --> (B,pop_size*p/B,da) ##i.e. divide the pop_size*p curr_acs among B networks
        dim=tf.shape(curr_acs)[-1]
        curr_acs=tf.reshape(tf.transpose(tf.reshape(curr_acs, [-1, self.model.B, self.p // self.model.B, dim]),[1, 0, 2, 3]),[self.model.B, -1, dim])
        
        inputs=tf.concat([obs_reshaped, curr_acs], axis=-1)
        mean, logvar = self.model(inputs) #here, input smaples will be = pop_size*p/B #???: does it make sense how inputs are normalized in this call??
        var = tf.math.exp(logvar)
        delta_obs_next=mean + tf.random.normal(shape=tf.shape(mean), mean=0, stddev=1) * tf.sqrt(var) #var.sqrt() = std
        #reshape delta_obs_next/predictions [back to original shape]  #(B,pop_size,p/B,ds) --> (pop_size*p,ds)
        dim=tf.shape(delta_obs_next)[-1]
        delta_obs_next=tf.transpose(tf.reshape(delta_obs_next,[self.model.B, -1, self.p // self.model.B, dim]),[1, 0, 2, 3])
        delta_obs_next=tf.reshape(delta_obs_next,[-1, dim])
        
        obs_next=delta_obs_next+obs
        return obs_next

# %% Inputs
p=20 #no. of particles
B=5 #no. of bootstraps (nets in ensemble)
K=50 #no. of trials
tr_eps=30 #30 #200 #no. of training episodes/iterations
n=3 #no. of NN layers
h=250 #500 #250 #size of hidden layers
H=2 #25 #planning horizon
# r=1 #no. of rollouts done in the environment for every training iteration AND no. of initial rollouts done before first train() call to controller #TODO: code is currently written for r=1; make code general to any r
epochs=5 #5 #100 #propagation method epochs
lr=0.001
b=32 #1 #batch size
pop_size=60 #400 #CEM population size: number of candidate solutions to be sampled every iteration 
opt_max_iters=5 #5 #CEM's max iterations (used as a termination condition)

# %% Environment
env=gym.make('cartpole_custom-v2')
ds=env.observation_space.shape[0] #state dims
da=env.action_space.shape[0] #action dims
set_seed(0,env)

# %% Initializations
in_size=ds+da
out_size=ds*2
model = PE(B,in_size,n,h,out_size)#dynamics model
optimizer=tf.keras.optimizers.Adam(learning_rate=lr) #model optimizer #TODO: use per-parameter options (and adjust model weights and biases into layers) to add different weight decays to each layer's weights
policy = MPC(env,model,optimizer,H,p,pop_size,opt_max_iters,epochs)
#traj_obs, traj_acs, traj_rs_sum, traj_rs = [], [], [], []
plot_rewards=[]

# %% Implementation

#1- Initialize data D with a random controller for one/r trial(s): sample an initial rollout from the agent with random policy
rollout = collect_rollout(env,policy)
#FILLME: apppend to lists: traj_obs, traj_acs, traj_rets, traj_rews

# 2- Training
episodes=progress(tr_eps)
for episode in episodes:
    #train policy with [prev] rollout
    policy.train(rollout,b)
    #sample a [new] rollout from the agent with MPC policy
    rollout = collect_rollout(env,policy)
    #FILLME: record outcome: extend with the sampled rollouts: traj_obs, traj_acs, traj_rets, traj_rews
    #log iteration results & statistics
    plot_rewards.append(rollout[-1])
    if episode % 1 == 0:
        log_msg="Rewards Sum: {:.2f}".format(rollout[-1])
        episodes.set_description(desc=log_msg); episodes.refresh()

# 3- Results & Plot
title="Training Rewards (Learning Curve)"
plt.figure(figsize=(16,8))
plt.grid(1)
plt.plot(plot_rewards)
plt.title(title)
plt.show()

# 4- Testing
