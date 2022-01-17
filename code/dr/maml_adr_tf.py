# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 18:56:33 2021

@author: TIB001
"""

# %% TODOs


# %% Imports
#general
import numpy as np
from scipy.spatial.distance import squareform, pdist

#env
import gym
#------only for spyder IDE
for env in gym.envs.registration.registry.env_specs.copy():
     if 'custom' in env:
         print('Remove {} from registry'.format(env))
         del gym.envs.registration.registry.env_specs[env]
#------
import gym_custom
from baselines.common.vec_env import CloudpickleWrapper

#visualization
import matplotlib.pyplot as plt
import tqdm

#utils
# from scipy.stats import truncnorm
from collections import OrderedDict #, namedtuple
import copy

#ML
import tensorflow as tf
import tensorflow_probability as tfp

#multiprocessing
import multiprocessing as mp
import os
# os.environ["OMP_NUM_THREADS"] = "1"
import queue as Q


#%% General
seeds=[None,1,2,3,4,5]
seed = seeds[1]
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
epsilon = np.finfo(np.float32).eps.item()

LUNAR_LANDER_SOLVED_SCORE = 200.0
check_solved = lambda r: np.median(r) > LUNAR_LANDER_SOLVED_SCORE

#%% Utils

progress=lambda x: tqdm.trange(x, leave=True) #for visualizing/monitoring training progress

def set_seed(seed,env,det=True):
    import random
    
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    env.seed(seed)
        

def collect_rollout_batch(envs, ds, da, policy, T, b, n_workers, queue, params=None,add_noise=False,noise_scale=0.1): # a batch of rollouts
    states=[[] for _ in range(b)]
    rewards = [[] for _ in range(b)]
    actions = [[] for _ in range(b)]
    next_states = [[] for _ in range(b)]
    
    states_mat=np.zeros((T,b,ds),dtype=np.float32)
    actions_mat=np.zeros((T,b,da),dtype=np.float32)
    rewards_mat=np.zeros((T,b),dtype=np.float32)
    
    
    for rollout_idx in range(b):
        queue.put(rollout_idx)
    for _ in range(n_workers):
        queue.put(None)
    
    #each rollout in the batch is the history of stepping through the environment once till termination
    s, rollout_idxs=envs.reset()
    dones=[False]
    
    while (not all(dones)) or (not queue.empty()):
        s=s.astype(np.float32)
        state=tf.convert_to_tensor(s)
        dist=policy(state,params)
        a=dist.sample().numpy()
        if add_noise:
            a = a + np.random.normal(0, noise_scale, size=a.shape)
            a = a.clip(-1, 1)
        s_dash, r, dones, rollout_idxs_new, _ = envs.step(a)
        #append to batch
        for state, next_state, action, reward, rollout_idx in zip(s,s_dash,a,r,rollout_idxs):
            if rollout_idx is not None:
                states[rollout_idx].append(state.astype(np.float32))
                next_states[rollout_idx].append(next_state.astype(np.float32))
                actions[rollout_idx].append(action.astype(np.float32))
                rewards[rollout_idx].append(reward.astype(np.float32))

        #reset
        s, rollout_idxs = s_dash, rollout_idxs_new
    
    for rollout_idx in range(b):
        states_mat[:,rollout_idx]= np.stack(states[rollout_idx])
        actions_mat[:,rollout_idx]= np.stack(actions[rollout_idx])
        rewards_mat[:,rollout_idx]= np.stack(rewards[rollout_idx])
    
    D=[states_mat, actions_mat, rewards_mat]
    
    #concatenate rollouts
    trajs = []
    for rollout_idx in range(b):
        trajs.append(np.concatenate(
            [
                np.array(states[rollout_idx]),
                np.array(actions[rollout_idx]),
                np.array(next_states[rollout_idx])
            ], axis=-1))
    
    return D, trajs


def detach_dist(pi):
    mean = tf.identity(pi.loc.numpy())
    std= tf.Variable(tf.identity(pi.scale.numpy()), name='old_pi/std', trainable=False, dtype=tf.float32)
    
    if isinstance(pi, tfp.distributions.Normal):
        return tfp.distributions.Normal(mean,std)
    elif isinstance(pi, tfp.distributions.MultivariateNormalDiag):
        return tfp.distributions.MultivariateNormalDiag(mean,tf.linalg.diag(std))
    elif isinstance(pi, tfp.distributions.Independent):
        return tfp.distributions.Independent(tfp.distributions.Normal(mean,std),1)
    else:
        raise RuntimeError('Distribution not supported')


def parameters_to_vector(X,Y=None):
    if Y is not None:
        #X=grads; Y=params
        return tf.concat([tf.reshape(x if x is not None else tf.zeros_like(y), [tf.size(y).numpy()]) for (y, x) in zip(Y, X)],axis=0)
    else:
        #X=params
        return tf.concat([tf.reshape(x, [tf.size(x).numpy()]) for x in X],axis=0).numpy()


def vector_to_parameters(vec,params):
    pointer = 0
    for param in params:

        numel = tf.size(param).numpy()
        param.assign(tf.reshape(vec[pointer:pointer + numel],list(param.shape)))

        pointer += numel
    
    return

def params2vec(parameters, grads):
    vec_params, vec_grads = [], []
    for param,grad in zip(parameters,grads):
        vec_params.append(tf.reshape(param,-1))
        vec_grads.append(tf.reshape(grad,-1))
    return tf.concat(vec_params,0), tf.concat(vec_grads,0)
    

def vec2params(vec, grads):
    grads_new=[]
    pointer = 0
    for grad in grads:
        numel = tf.size(grad).numpy()
        grads_new.append(tf.reshape(vec[pointer:pointer + numel],list(grad.shape)))

        pointer += numel
    return grads_new


#%% MAML

class PolicyNetwork(tf.keras.Model):
    def __init__(self, in_size, n, h, out_size):
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
    def __init__(self, in_size, T, b, gamma, reg_coeff=1e-5):
        super().__init__()
        
        self.reg_coeff=reg_coeff
        self.T = T
        self.b = b
        self.gamma=gamma
        
        self.ones = tf.ones([self.T,self.b,1],dtype=tf.float32)
        self.timestep= tf.math.cumsum(self.ones, axis=0) / 100.
        self.feature_size=2*in_size + 4
        self.eye=tf.eye(self.feature_size,dtype=tf.float32)
        
        self.w=tf.Variable(initial_value=tf.keras.initializers.Zeros()(shape=[self.feature_size,1]),dtype=tf.float32,trainable=False)
        
    def fit_params(self, states, rewards):
        
        reg_coeff = self.reg_coeff
        
        #create features
        features = tf.concat([states, states **2, self.timestep, self.timestep**2, self.timestep**3, self.ones],axis=2)
        features=tf.reshape(features, (-1, self.feature_size))

        #compute returns        
        G = np.zeros(self.b,dtype=np.float32)
        returns = np.zeros((self.T,self.b),dtype=np.float32)
        for t in range(self.T - 1, -1, -1):
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
        features = tf.concat([states, states **2, self.timestep, self.timestep**2, self.timestep**3, self.ones],axis=2)
        return tf.linalg.matmul(features,self.w)


def compute_advantages(states,rewards,value_net,gamma):
    
    values = value_net(states)
    if len(list(values.shape))>2: values = tf.squeeze(values, axis=2)
    values = tf.pad(values,[[0, 1], [0, 0]])
    
    deltas = rewards + gamma * values[1:] - values[:-1] #delta = r + gamma * v - v' #TD error
    # advantages = tf.zeros_like(deltas, dtype=tf.float32)
    advantages = tf.TensorArray(tf.float32, *deltas.shape)
    advantage = tf.zeros_like(deltas[0], dtype=tf.float32)
    
    for t in range(value_net.T - 1, -1, -1): #reversed(range(-1,value_net.T -1 )):
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

#%% TRPO

def conjugate_gradients(Avp_f, b, rdotr_tol=1e-10, nsteps=10):
    """
    nsteps = max_iterations
    rdotr = residual
    """
    x = np.zeros_like(b)
    r = b.numpy().copy()
    p = b.numpy().copy() 
    rdotr = r.dot(r) 
    
    for i in range(nsteps):
        Avp = Avp_f(p).numpy()
        alpha = rdotr / p.dot(Avp) 
        x += alpha * p
        r -= alpha * Avp
        new_rdotr = r.dot(r)
        beta = new_rdotr / rdotr
        p = r + beta * p
        rdotr = new_rdotr
        if rdotr < rdotr_tol:
            break
        
    return x


def surrogate_loss(Ds, D_dashes,policy,value_net,gamma,alpha,T,prev_pis=None):
    
    kls, losses, pis =[], [], []
    if prev_pis is None:
        prev_pis = [None] * T
        
    for D, D_dash, prev_pi in zip(Ds,D_dashes,prev_pis):
        
        theta_dash=adapt(D,value_net,policy,alpha)

        states, actions, rewards = D_dash
        
        pi=policy(states,params=theta_dash)
        pis.append(detach_dist(pi))
        
        if prev_pi is None:
            prev_pi = detach_dist(pi)
        
        advantages=compute_advantages(states,rewards,value_net,gamma)
        
        ratio=pi.log_prob(actions)-prev_pi.log_prob(actions)
        loss = - tf.math.reduce_mean(advantages * tf.math.exp(tf.math.reduce_sum(ratio,axis=2))) if len(list(ratio.shape)) > 2 else - tf.math.reduce_mean(advantages * tf.math.exp(ratio))
        losses.append(loss)
        
        #???: which version is correct?
        # kl=tf.math.reduce_mean(pi.kl_divergence(prev_pi))
        kl=tf.math.reduce_mean(prev_pi.kl_divergence(pi))
        
        kls.append(kl)
    
    prev_loss=tf.math.reduce_mean(tf.stack(losses, axis=0))
    kl=tf.math.reduce_mean(tf.stack(kls, axis=0)) #???: why is it always zero?
    
    return prev_loss, kl, pis


def line_search(policy, prev_loss, prev_pis, value_net, alpha, gamma, T, b, D_dashes, Ds, step, prev_params, max_grad_kl, max_backtracks=10, zeta=0.5):
    """backtracking line search"""
    
    for step_frac in [zeta**x for x in range(max_backtracks)]:
        vector_to_parameters(prev_params - step_frac * step, policy.trainable_variables)
        
        loss, kl, _ = surrogate_loss(Ds, D_dashes,policy,value_net,gamma,alpha,T,prev_pis=prev_pis)
        
        #check improvement
        actual_improve = loss - prev_loss
        if not np.isfinite(loss):
            raise RuntimeError('NANs/Infs encountered in line search')
        if (actual_improve.numpy() < 0.0) and (kl.numpy() < max_grad_kl):
            break
    else:
        vector_to_parameters(prev_params, policy.trainable_variables)


def kl_div(Ds,D_dashes,value_net,policy,alpha):
    kls=[]
    for D, D_dash in zip(Ds,D_dashes):
        
        theta_dash=adapt(D,value_net,policy,alpha)

        states, _, _ = D_dash
        
        pi=policy(states,params=theta_dash)
        
        prev_pi = detach_dist(pi)
        
        #???: which version is correct?
        # kl=tf.math.reduce_mean(pi.kl_divergence(prev_pi))
        kl=tf.math.reduce_mean(prev_pi.kl_divergence(pi))
        
        kls.append(kl)
    
    return tf.math.reduce_mean(tf.stack(kls, axis=0))


def HVP(Ds,D_dashes,policy,value_net,alpha,damping):
    def _HVP(v):
        with tf.GradientTape() as outer_tape:
             with tf.GradientTape() as inner_tape:
                 kl = kl_div(Ds,D_dashes,value_net,policy,alpha)
             grad_kl=parameters_to_vector(inner_tape.gradient(kl,policy.trainable_variables),policy.trainable_variables)
             dot=tf.tensordot(grad_kl, v, axes=1)
        return parameters_to_vector(outer_tape.gradient(dot, policy.trainable_variables),policy.trainable_variables) + damping * v            
    return _HVP


#%% Discriminator

class MLP(tf.keras.Model):
    def __init__(self, in_size, h1, h2):
        super(MLP, self).__init__()
        
        self.mlp=tf.keras.models.Sequential(layers=[
            tf.keras.layers.Input(in_size),
            tf.keras.layers.Dense(h1,activation="tanh"),
            tf.keras.layers.Dense(h2,activation="tanh"),
            tf.keras.layers.Dense(1,activation="sigmoid")
            ])

    # Tuple of S-A-S'
    def call(self, x):
        x = self.mlp(x)
        return x


class Discriminator(object):
    def __init__(self, ds, da, h, b_disc, r_disc_scale, lr_disc):
        self.discriminator = MLP(in_size=ds+da+ds, h1=h, h2=h)  #Input: state-action-state' transition; Output: probability that it was from a reference trajectory

        self.disc_loss_func = tf.keras.losses.BinaryCrossentropy()
        self.disc_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_disc)
        self.reward_scale = r_disc_scale
        self.batch_size = b_disc 

    def calculate_rewards(self, randomized_trajectory):
        """
        We want to use the negative of the adversarial calculation (Normally, -log(D)). We want to *reward*
        our simulator for making it easier to discriminate between the reference env + randomized onea
        """
        traj_tensor = tf.convert_to_tensor(randomized_trajectory,dtype=tf.float32) 

        score = tf.stop_gradient((self.discriminator(traj_tensor).numpy()+1e-8).mean())
        
        reward = np.log(score) - np.log(0.5)

        return self.reward_scale * reward, score

    def train(self, ref_traj, rand_traj, eps):
        """Trains discriminator to distinguish between reference and randomized state action tuples"""
        for _ in range(eps):
            randind = np.random.randint(0, len(rand_traj[0]), size=int(self.batch_size))
            refind = np.random.randint(0, len(ref_traj[0]), size=int(self.batch_size))
            
            with tf.GradientTape() as tape:
                rand_batch = tf.convert_to_tensor(rand_traj[randind],dtype=tf.float32)
                ref_batch = tf.convert_to_tensor(ref_traj[refind],dtype=tf.float32)
    
                g_o = self.discriminator(rand_batch)
                e_o = self.discriminator(ref_batch)
                
                disc_loss = self.disc_loss_func(g_o, tf.ones((len(rand_batch), 1))) + self.disc_loss_func(e_o,tf.zeros((len(ref_batch), 1)))
            
            gradients = tape.gradient(disc_loss, self.discriminator.trainable_variables) #calculate gradient
            self.disc_optimizer.apply_gradients(zip(gradients, self.discriminator.trainable_variables)) #backpropagate


r_disc_scale = 1. #reward scale
h_disc=128 #32 #128
lr_disc=0.002 #0.02 #0.002
b_disc=128

# discriminator=Discriminator(ds, da, h_disc, b_disc, r_disc_scale, lr_disc)


#%% Envs

class SubprocVecEnv(gym.Env):
    def __init__(self,env_funcs,ds,da,queue,lock):
        
        self.parent_conns, self.child_conns = zip(*[mp.Pipe() for _ in env_funcs])
        # self.workers = [EnvWorker(child_conn, env_func, queue, lock) for (child_conn, env_func) in zip(self.child_conns, env_funcs)]
        self.workers = [mp.Process(target=envworker,args=(child_conn, parent_conn, CloudpickleWrapper(env_func),queue,lock)) for (child_conn, parent_conn, env_func) in zip(self.child_conns, self.parent_conns, env_funcs)]
        
        for worker in self.workers:
            worker.daemon = True #making child processes daemonic to not continue running when master process exists
            worker.start()
        for child_conn in self.child_conns:
            child_conn.close()
        
        self.waiting = False
        self.closed = False
        
    def step(self, actions):
        
        #step through each env asynchronously
        for parent_conn, action in zip(self.parent_conns, actions):
            parent_conn.send(('step',action))
        self.waiting = True
        
        #wait for all envs to finish stepping and then collect results
        results = [parent_conn.recv() for parent_conn in self.parent_conns]
        self.waiting = False
        states, rewards, dones, rollouts_idxs, infos = zip(*results)
        
        return np.stack(states), np.stack(rewards), np.stack(dones), rollouts_idxs, infos
    
    def randomize(self, randomized_values):
        for parent_conn, val in zip(self.parent_conns, randomized_values):
            parent_conn.send(('randomize', val))
        results = [parent_conn.recv() for parent_conn in self.parent_conns]
        self.waiting = False
    
    def reset_task(self, tasks):
        for parent_conn, task in zip(self.parent_conns,tasks):
            parent_conn.send(('reset_task',task))
        return np.stack([parent_conn.recv() for parent_conn in self.parent_conns])
    
    def reset(self):
        for parent_conn in self.parent_conns:
            parent_conn.send(('reset',None))
        results = [parent_conn.recv() for parent_conn in self.parent_conns]
        states, rollouts_idxs = zip(*results)
        return np.stack(states), rollouts_idxs
    
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
        

def envworker(child_conn, parent_conn, env_func, queue, lock):
    parent_conn.close()
    env = env_func.x()
    done=False
    rollout_idx = None
    ds=env.observation_space.shape[0]
    
    def try_reset(lock):
        with lock:
            try:
                rollout_idx = queue.get()
                done = (rollout_idx is None)
            except Q.Empty:
                done = True
        if done:
            state = np.zeros(ds, dtype=np.float32)
        else:
            state = env.reset()
        return state, rollout_idx, done
    
    while True:
        func, arg = child_conn.recv()
        
        if func == 'step':
            if done:
                state, reward, done_env, info = np.zeros(ds, dtype=np.float32), 0.0, True, {}
            else:
                state, reward, done_env, info = env.step(arg)
            if done_env and not done:
                state, rollout_idx, done = try_reset(lock)
            child_conn.send((state,reward,done_env,rollout_idx,info))
        elif func == 'reset':
            state, rollout_idx, done = try_reset(lock)
            child_conn.send((state,rollout_idx))
        elif func == 'reset_task':
            env.reset_task(arg)
            child_conn.send(True)
        elif func == 'close':
            child_conn.close()
            break
        elif func == 'randomize':
            env.randomize(arg)
            child_conn.send(None)


def make_env(env_name,seed=None, rank=None):
    def _make_env():
        env = gym.make(env_name)
        if seed is not None and rank is not None:
            env.seed(seed+rank)
        return env
    return _make_env


def make_vec_envs(env_name, seed, n_workers, queue, lock):
    envs=[make_env(env_name,seed,rank) for rank in range(n_workers)]
    envs=SubprocVecEnv(envs, ds, da, queue, lock)
    return envs


def reset_tasks(envs,tasks):
    return all(envs.reset_task(tasks))

#%% ADR Policy (or: ensemble of policies / SVPG particles)

#============================
# Adjust Torch Distributions
#============================

# Categorical
FixedCategorical = tfp.distributions.Categorical

old_sample = FixedCategorical.sample
FixedCategorical.sample = lambda self: tf.expand_dims(old_sample(self), -1)

log_prob_cat = FixedCategorical.log_prob
FixedCategorical.log_probs = lambda self, actions: tf.expand_dims(tf.reduce_sum(tf.reshape(log_prob_cat(self, tf.squeeze(actions,-1)),shape=(tf.size(actions).numpy()[0], -1)),-1),-1)

FixedCategorical.mode = lambda self: tf.argmax(self.probs,-1)

class Categorical(tf.keras.Model):
    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()
        
        self.linear=tf.keras.models.Sequential(layers=[
            tf.keras.layers.Input(num_inputs),
            tf.keras.layers.Dense(num_outputs,kernel_initializer=tf.keras.initializers.Orthogonal(gain=0.01))
            ])

    def call(self, x):
        x = self.linear(x)
        return FixedCategorical(logits=x)


# Normal
FixedNormal = tfp.distributions.Normal

log_prob_normal = FixedNormal.log_prob
FixedNormal.log_probs = lambda self, actions: tf.reduce_sum(log_prob_normal(self, actions),-1, keepdims=True)

normal_entropy = FixedNormal.entropy
FixedNormal.entropy = lambda self: tf.reduce_sum(normal_entropy(self),-1)

FixedNormal.mode = lambda self: self.mean


class AddBias(tf.keras.Model):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = tf.Variable(tf.expand_dims(bias,1))

    def call(self, x):
        if tf.size(x).numpy()== 2:
            bias = tf.reshape(tf.transpose(self._bias),(1, -1))
        else:
            bias = tf.reshape(tf.transpose(self._bias),(1, -1, 1, 1))

        return x + bias

class DiagGaussian(tf.keras.Model):
    def __init__(self, num_inputs, num_outputs):
        super(DiagGaussian, self).__init__()

        self.fc_mean = tf.keras.layers.Dense(num_outputs,kernel_initializer=tf.keras.initializers.Orthogonal(gain=1)) 
        self.logstd = AddBias(tf.zeros(num_outputs))

    def call(self, x):
        action_mean = self.fc_mean(x)

        #  An ugly hack for my KFAC implementation.
        zeros = tf.zeros(tf.size(action_mean))

        action_logstd = self.logstd(zeros)
        return FixedNormal(action_mean, tf.exp(action_logstd))
    
#======
# SVPG
#======

class SVPGParticleCritic(tf.keras.Model):
    def __init__(self, in_size, h):
        super(SVPGParticleCritic, self).__init__()
        
        self.critic=tf.keras.models.Sequential(layers=[
            tf.keras.layers.Input(in_size),
            tf.keras.layers.Dense(h,activation="tanh",kernel_initializer=tf.keras.initializers.Orthogonal(gain=np.sqrt(2))),
            tf.keras.layers.Dense(h,activation="tanh",kernel_initializer=tf.keras.initializers.Orthogonal(gain=np.sqrt(2))),
            tf.keras.layers.Dense(1,kernel_initializer=tf.keras.initializers.Orthogonal(gain=np.sqrt(2)))
            ])

    def call(self, x):
        return self.critic(x)


class SVPGParticleActor(tf.keras.Model):
    def __init__(self, in_size, h):
        super(SVPGParticleActor, self).__init__()
        
        self.actor=tf.keras.models.Sequential(layers=[
            tf.keras.layers.Input(in_size),
            tf.keras.layers.Dense(h,activation="tanh",kernel_initializer=tf.keras.initializers.Orthogonal(gain=np.sqrt(2))),
            tf.keras.layers.Dense(h,activation="tanh",kernel_initializer=tf.keras.initializers.Orthogonal(gain=np.sqrt(2)))
            ])

    def call(self, x):
        return self.actor(x)


class SVPGParticle(tf.keras.Model):
    """Implements a AC architecture for a Discrete A2C Policy, used inside of SVPG"""
    def __init__(self, in_size, out_size, h, type_particles, freeze=False):
        super(SVPGParticle, self).__init__()

        self.critic = SVPGParticleCritic(in_size=in_size, h=h)
        self.actor = SVPGParticleActor(in_size=in_size, h=h)
        self.dist = Categorical(h, out_size) if type_particles=="discrete" else DiagGaussian(h, out_size)

        self.critic.trainable=(not freeze)
        self.actor.trainable=(not freeze)
        self.dist.trainable=(not freeze)
        
        self.reset()
        
    def reset(self):
        self.saved_log_probs = []
        self.saved_klds = []
        self.rewards = []

    def call(self, x):
        actor = self.actor(x)
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
            policy = SVPGParticle(in_size=self.dr, out_size=self.out_size, h=h, type_particles=type_particles) 
            prior_policy = SVPGParticle(in_size=self.dr, out_size=self.out_size, h=h, type_particles=type_particles, freeze=True) 
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr_svpg)
            
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

        X_np = X.numpy()
        pairwise_dists = squareform(pdist(X_np))**2

        # Median trick
        h = np.median(pairwise_dists)  
        h = np.sqrt(0.5 * h / np.log(self.n_particles+1))

        # Compute RBF Kernel
        k = tf.exp(-tf.convert_to_tensor(pairwise_dists,dtype=tf.float32) / h**2 / 2)

        # Compute kernel gradient
        grad_k = -tf.matmul(k,X)
        sum_k = tf.reduce_sum(k,1)
        grad_k_new=[]
        for i in range(X.shape[1]):
            grad_k_new.append(grad_k[:, i] + tf.tensordot(X[:, i],sum_k,1))
        
        grad_k=tf.stack(grad_k_new,1)
        grad_k /= (h ** 2)

        return k, grad_k

    def select_action(self, policy_idx, state):
        state = tf.expand_dims(tf.convert_to_tensor(state,dtype=tf.float32),0)
        policy = self.particles[policy_idx]
        prior_policy = self.prior_particles[policy_idx]
        dist, value = policy(state)
        prior_dist, _ = prior_policy(state)

        action = dist.sample()               
            
        policy.saved_log_probs.append(dist.log_prob(action))
        policy.saved_klds.append(dist.kl_divergence(prior_dist))
        
        if self.dr == 1 or self.type_particles=="discrete":
            action = tf.squeeze(action).numpy()
        else:
            action = tf.squeeze(action).numpy()

        return action, value

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
        self.values=[[] for _ in range(self.n_particles)]
        self.tapes=[]
        # Store the last states for each particle (calculating rewards)
        self.masks = np.ones((self.n_particles, self.T_svpg))

        for i in range(self.n_particles):
            self.particles[i].reset()
            current_sim_params = self.last_states[i]

            with tf.GradientTape(persistent=True) as tape:
                for t in range(self.T_svpg):
                    self.simulation_instances[i][t] = current_sim_params
    
                    action, value = self.select_action(i, current_sim_params)  
                    self.values[i].append(value)
                    
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
            self.tapes.append(tape)

        return np.array(self.simulation_instances)

    def train(self, simulator_rewards):
        policy_grads = []
        parameters = []
        grads_tot=[]
        
        for i in range(self.n_particles):
            policy_grad_particle = []
            
            with self.tapes[i]:
                # Calculate the value of last state - for Return Computation
                _, next_value = self.select_action(i, self.last_states[i]) 
    
                particle_rewards = tf.convert_to_tensor(simulator_rewards[i],dtype=tf.float32)
                masks = tf.convert_to_tensor(self.masks[i],dtype=tf.float32)
                
                # Calculate entropy-augmented returns, advantages
                returns = self.compute_returns(next_value, particle_rewards, masks, self.particles[i].saved_klds)
                returns = tf.stop_gradient(tf.concat(returns,0))
                advantages = returns - tf.concat(self.values[i],0)
                
                # Compute value loss, update critic
                critic_loss = 0.5 * tf.reduce_mean(tf.square(advantages))
                
                # Store policy gradients for SVPG update
                for log_prob, advantage in zip(self.particles[i].saved_log_probs, advantages):
                    policy_grad_particle.append(log_prob * tf.stop_gradient(advantage))
                policy_grad = -tf.reduce_mean(tf.concat(policy_grad_particle,0))
                
            gradients_c = self.tapes[i].gradient(critic_loss, self.particles[i].trainable_variables) #calculate gradient
            self.optimizers[i].apply_gradients(zip(gradients_c, self.particles[i].trainable_variables))
                
            gradients_p = self.tapes[i].gradient(policy_grad, self.particles[i].trainable_variables) #calculate gradient
            
            gradients=[]
            for idx, grad in enumerate(gradients_c):
                if grad is not None:
                    gradients.append(grad)
                else:
                    gradients.append(gradients_p[idx])
            grads_tot.append(gradients)
            
            # Vectorize parameters and PGs
            vec_param, vec_policy_grad = params2vec(self.particles[i].trainable_variables, gradients)

            policy_grads.append(tf.expand_dims(vec_policy_grad,0))
            parameters.append(tf.expand_dims(vec_param,0))

        # calculating the kernel matrix and its gradients
        parameters = tf.concat(parameters,0)
        k, grad_k = self.compute_kernel(parameters)

        policy_grads = 1 / self.temp * tf.concat(policy_grads,0)
        grad_logp = tf.linalg.matmul(k, policy_grads)

        grad_theta = (grad_logp + grad_k) / self.n_particles

        # update param gradients
        for i in range(self.n_particles):
            grads_tot[i]=vec2params(grad_theta[i],grads_tot[i])
            self.optimizers[i].apply_gradients(zip(grads_tot[i], self.particles[i].trainable_variables))
            
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
            if isinstance(action, np.float32):
                action = np.clip(action, -1, 1)
            else:
                action /= np.linalg.norm(action, ord=2)

        return np.array(action)


n_particles=10 #5 #10
temp=10. #temperature
types_particles=["discrete","continuous"] #???: which one is better?
type_particles=types_particles[0]
kld_coeff=0. #kld = KL Divergence
T_svpg_reset=25 #how often to fully reset svpg particles
delta_max=0.05 #maximum allowable change to env randomization params caused by svpg particles (If discrete, this is fixed, If continuous, this is max)
T_svpg_init=100 #1000 #0 #number of svpg steps to take before updates
T_svpg=5 #length of one svpg particle rollout
lr_svpg=0.0003 #0.03 #0.0003
gamma_svpg=0.99
h_svpg=100 #16 #100

# svpg = SVPG(n_particles, dr, h_svpg, delta_max, T_svpg, T_svpg_reset, temp, kld_coeff, lr_svpg, gamma_svpg, type_particles)


#%% Main Func
if __name__ == '__main__':
    
    # %% Inputs
    #model / policy
    n=2 #no. of NN layers
    h=100 #size of hidden layers
    
    #optimizer
    alpha=0.1 #adaptation step size / learning rate
    # beta=0.001 #meta step size / learning rate #TIP: controlled and adapted by TRPO (in maml step) [to guarantee monotonic improvement, etc]
    
    #general
    # K=30 #no. of trials
    tr_eps=500 #20 #200 #no. of training episodes/iterations
    log_ival=1 #logging interval
    b = n_particles #20 #16 #32 #batch size: Number of trajectories (rollouts) to sample from each task
    n_workers = n_particles #mp.cpu_count() - 1 #=n_envs
    
    #VPG
    gamma = 0.99
    
    #TRPO
    max_grad_kl=1e-2
    max_backtracks=15
    accept_ratio=0.1
    zeta=0.8 #0.5
    rdotr_tol=1e-10
    nsteps=10
    damping=1e-5 #0.01 
        
    
    #%% Initializations
    #multiprocessing
    queue = mp.Queue()
    lock = mp.Lock()
    
    #environment
    env_names=['cartpole_custom-v1','halfcheetah_custom-v1','halfcheetah_custom_norm-v1','halfcheetah_custom_rand-v1','lunarlander_custom_default_rand-v0']
    env_name=env_names[-1]
    env=gym.make(env_name)
    T_env=env._max_episode_steps #200 #task horizon
    ds=env.observation_space.shape[0] #state dims
    da=env.action_space.shape[0] #action dims
    dr=env.unwrapped.randomization_space.shape[0] #N_rand (no. of randomization params)
    
    env_ref=make_vec_envs(env_name, seed, n_workers, queue, lock)
    env_rand=make_vec_envs(env_name, seed, n_workers, queue, lock)
    
    discriminator=Discriminator(ds, da, h_disc, b_disc, r_disc_scale, lr_disc)
    svpg = SVPG(n_particles, dr, h_svpg, delta_max, T_svpg, T_svpg_reset, temp, kld_coeff, lr_svpg, gamma_svpg, type_particles)
    
    #models
    in_size=ds
    out_size=da
    policy = PolicyNetwork(in_size,n,h,out_size) #dynamics model
    value_net=ValueNetwork(in_size,T_env,b,gamma)
    
    #results 
    plot_tr_rewards=[]
    plot_val_rewards=[]
    best_reward=-1e6
    
    set_seed(seed,env)
    
    #%% Implementation
        
    episodes=progress(tr_eps)
    for episode in episodes:
        
        #rollout svpg particles (each particle represents a different rand env and each timestep in that rollout represents different values for its randomization params)
        simulation_instances = svpg.step() if episode >= T_svpg_init else -1 * np.ones((n_particles,T_svpg,dr))
        
        #create empty storages
        rewards_tr_ep, rewards_val_ep = [], []
        Ds, D_dashes, D_dashes_ref, D_dashes_rand=[], [], [], []
        rewards_disc = np.zeros(simulation_instances.shape[:2])
        scores_disc=np.zeros(simulation_instances.shape[:2])
        
        # Reshape to work with vectorized environments
        simulation_instances = np.transpose(simulation_instances, (1, 0, 2))
        
        #inner/adaptation loop
        for t_svpg in range(T_svpg):
            #randomize rand envs (with svpg particle [randomization parameter] values [at the current svpg timestep]) #!!!: this only works if transitions within the same rollout is collected in the same environment (i.e. by the same [env] worker) #that's why we choose: n_particles=b(rollout batch size)=n_workers/n_envs
            env_rand.randomize(simulation_instances[t_svpg])
            
            #collect pre-adaptation rollout batch in rand envs (one rollout for each svpg particle)
            D,_=collect_rollout_batch(env_rand, ds, da, policy, T_env, b, n_workers, queue)
            Ds.append(D)
            _, _, rewards = D
            rewards_tr_ep.append(rewards)
            
            #adapt agent [meta-]parameters (via VPG w/ baseline)
            theta_dash=adapt(D,value_net,policy,alpha)
            
            #collect post-adaptaion rollout batch in ref envs
            _,ref_traj=collect_rollout_batch(env_ref, ds, da, policy, T_env, b, n_workers, queue, params=theta_dash)
            D_dashes_ref.append(ref_traj)
            
            #collect post-adaptation rollout batch in rand envs
            D_dash,rand_traj=collect_rollout_batch(env_rand, ds, da, policy, T_env, b, n_workers, queue, params=theta_dash)
            D_dashes_rand.append(rand_traj)
            D_dashes.append(D_dash)
            _, _, rewards = D_dash
            rewards_val_ep.append(rewards)
        
        #ADR updates
        for t, (ref_traj, rand_traj) in enumerate(zip(D_dashes_ref,D_dashes_rand)):
            T_disc_eps=0 #agent timesteps in the current iteration/episode

            #calculate discriminator reward
            for i in range(n_particles):
                T_disc_eps += len(rand_traj[i])
                
                r_disc, score_disc = discriminator.calculate_rewards(rand_traj[i])
                rewards_disc[i][t]= r_disc
                scores_disc[i][t]=score_disc
            
            #train discriminator
            flattened_rand = [rand_traj[i] for i in range(n_particles)]
            flattened_rand = np.concatenate(flattened_rand)
    
            flattened_ref= [ref_traj[i]for i in range(n_particles)]
            flattened_ref = np.concatenate(flattened_ref)
            
            discriminator.train(ref_traj=flattened_ref, rand_traj=flattened_rand, eps=T_disc_eps)
            
            #update svpg particles' params (ie. train their policies)
            if episode >= T_svpg_init:
                svpg.train(rewards_disc)
        
        #outer loop: update meta-params (via: TRPO) #!!!: since MAML uses TRPO it is on-policy, so care should be taken that order of associated transitions is preserved
        with tf.GradientTape() as tape:
            prev_loss, _, prev_pis = surrogate_loss(Ds, D_dashes,policy,value_net,gamma,alpha,T_env)
        grads = parameters_to_vector(tape.gradient(prev_loss, policy.trainable_variables),policy.trainable_variables)
        prev_loss=tf.identity(prev_loss)
        hvp=HVP(Ds,D_dashes,policy,value_net,alpha,damping)
        search_step_dir=conjugate_gradients(hvp, grads)
        max_length=np.sqrt(2.0 * max_grad_kl / np.dot(search_step_dir, hvp(search_step_dir)))
        full_step=search_step_dir*max_length        
        prev_params = parameters_to_vector(policy.trainable_variables)
        line_search(policy, prev_loss, prev_pis, value_net, alpha, gamma, T_env, b, D_dashes, Ds, full_step, prev_params, max_grad_kl, max_backtracks, zeta)
    
        #compute & log results
        # compute rewards
        reward_ep = (tf.math.reduce_mean(tf.stack([tf.math.reduce_mean(tf.math.reduce_sum(rewards, axis=0)) for rewards in rewards_tr_ep], axis=0))).numpy() #sum over T, mean over b, stack horiz one reward per task, mean of tasks
        reward_val=(tf.math.reduce_mean(tf.stack([tf.math.reduce_mean(tf.math.reduce_sum(rewards, axis=0)) for rewards in rewards_val_ep], axis=0))).numpy()
        #save best running model [params]
        if reward_val>best_reward: 
            best_reward=reward_val
            policy.save_weights("saved_models/"+env_name)
        #save plot data
        plot_tr_rewards.append(reward_ep)
        plot_val_rewards.append(reward_val)
        #log iteration results & statistics
        if episode % log_ival == 0:
            log_msg="Rewards Tr: {:.2f}, Rewards Val: {:.2f}".format(reward_ep,reward_val)
            episodes.set_description(desc=log_msg); episodes.refresh()
    
    #%% Results & Plot
    title="Meta-Training Training Rewards (Learning Curve)"
    plt.figure(figsize=(16,8))
    plt.grid(1)
    plt.plot(plot_tr_rewards)
    plt.title(title)
    # plt.savefig('plots/mtr_tr.png')
    plt.show()
    
    title="Meta-Training Testing Rewards (Learning Curve)"
    plt.figure(figsize=(16,8))
    plt.grid(1)
    plt.plot(plot_val_rewards)
    plt.title(title)
    # plt.savefig('plots/mtr_ts.png')
    plt.show()
