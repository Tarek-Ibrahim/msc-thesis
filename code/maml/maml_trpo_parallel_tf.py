# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 18:56:33 2021

@author: TIB001
"""

# %% TODOs
# TODO: make sure algorithm runs learns
# TODO: include [custom] environments in the main repo (and remove the need for the gym-custom repo) 
# TODO: study information theory, understand TRPO better, understand MAML better (incl. higher order derivatives in pytorch), summarize meta-learning, upload summaries to repo
# TODO: implements GrBAL/ReBAL (MAML + model-based) --> 1st major milestone
# TODO: look into performance and memory issues

# %% Imports
#general
import numpy as np

#env
import gym
#------only for spyder IDE
for env in gym.envs.registration.registry.env_specs.copy():
     if 'custom' in env:
         print('Remove {} from registry'.format(env))
         del gym.envs.registration.registry.env_specs[env]
#------
import gym_custom

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
# import os
# os.environ["OMP_NUM_THREADS"] = "1"
import queue as Q


#%% Functions

#--------
# Utils
#--------

progress=lambda x: tqdm.trange(x, leave=True) #for visualizing/monitoring training progress

def set_seed(seed,env,det=True):
    import random
    
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    env.seed(seed)
        
        
#--------
# Common
#--------

def collect_rollout_batch(envs, ds, da, policy, T, b, n_workers, queue, params=None): # a batch of rollouts
    states=[[] for _ in range(b)]
    rewards = [[] for _ in range(b)]
    actions = [[] for _ in range(b)]
    
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
        s_dash, r, dones, rollout_idxs_new, _ = envs.step(a)
        #append to batch
        for state, action, reward, rollout_idx in zip(s,a,r,rollout_idxs):
            if rollout_idx is not None:
                states[rollout_idx].append(state.astype(np.float32))
                actions[rollout_idx].append(action.astype(np.float32))
                rewards[rollout_idx].append(reward.astype(np.float32))
        #reset
        s, rollout_idxs = s_dash, rollout_idxs_new
    
    for rollout_idx in range(b):
        states_mat[:,rollout_idx]= np.stack(states[rollout_idx])
        actions_mat[:,rollout_idx]= np.stack(actions[rollout_idx])
        rewards_mat[:,rollout_idx]= np.stack(rewards[rollout_idx])
    
    D=[states_mat, actions_mat, rewards_mat]
    
    return D

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


def reset_tasks(envs,tasks):
    return all(envs.reset_task(tasks))


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

#--------
# Models
#--------

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


def detach_dist(pi):
    mean = tf.identity(pi.loc.numpy())
    std= tf.Variable(tf.identity(pi.scale.numpy()), name='old_pi/std', trainable=False, dtype=tf.float32)
    
    return tfp.distributions.Normal(mean,std)
    #return tfp.distributions.MultivariateNormalDiag(mean,tf.linalg.diag(std))
    #return tfp.distributions.Independent(tfp.distributions.Normal(mean,std),1)
    
#--------
# TRPO
#--------

def conjugate_gradients(Avp_f, b, rdotr_tol=1e-10, nsteps=10):
    """
    nsteps = max_iterations
    rdotr = residual
    """
    x = np.zeros_like(b) #tf.zeros_like(b,dtype=tf.float32)
    r = b.numpy().copy() #tf.identity(b)
    p = b.numpy().copy() #tf.identity(b)
    rdotr = r.dot(r) #tf.tensordot(r, r,axes=1)
    
    for i in range(nsteps):
        Avp = Avp_f(p).numpy()
        alpha = rdotr / p.dot(Avp) #tf.tensordot(p, Avp,axes=1)
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


#----------------
# Multiprocessing
#----------------

class SubprocVecEnv(gym.Env):
    def __init__(self,env_funcs,ds,da,queue,lock):
        
        self.parent_conns, self.child_conns = zip(*[mp.Pipe() for _ in env_funcs])
        self.workers = [EnvWorker(child_conn, env_func, queue, lock) for (child_conn, env_func) in zip(self.child_conns, env_funcs)]
        
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
        
        
class EnvWorker(mp.Process):
    def __init__(self, child_conn, env_func, queue, lock):
        super().__init__()
        
        self.child_conn = child_conn
        self.env = env_func()
        self.queue = queue
        self.lock = lock
        self.rollout_idx = None
        self.done = False
        self.ds = self.env.observation_space.shape
    
    def try_reset(self):
        with self.lock:
            try:
                self.rollout_idx = self.queue.get()
                self.done = (self.rollout_idx is None)
            except Q.Empty:
                self.done = True
        if self.done:
            state = np.zeros(self.ds, dtype=np.float32)
        else:
            state = self.env.reset()
        return state
    
    def run(self):
        while True:
            func, arg = self.child_conn.recv()
            
            if func == 'step':
                if self.done:
                    state, reward, done, info = np.zeros(self.ds, dtype=np.float32), 0.0, True, {}
                else:
                    state, reward, done, info = self.env.step(arg)
                if done and not self.done:
                    state = self.try_reset()
                self.child_conn.send((state,reward,done,self.rollout_idx,info))
                
            elif func == 'reset':
                state = self.try_reset()
                self.child_conn.send((state,self.rollout_idx))
              
            elif func == 'reset_task':
                self.env.reset_task(arg)
                self.child_conn.send(True)
                
            elif func == 'close':
                self.child_conn.close()
                break

def make_env(env_name,seed=None):
    def _make_env():
        env = gym.make(env_name)
        env.seed(seed)
        return env
    return _make_env

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
    b=20 #16 #32 #batch size: Number of trajectories (rollouts) to sample from each task
    meta_b=40 #15 #30 #number of tasks sampled
    
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
    
    #multiprocessing
    n_workers = mp.cpu_count() - 1
    
    # %% Initializations
    #common
    seed=0
    
    #multiprocessing
    queue = mp.Queue()
    lock = mp.Lock()
    
    #environment
    env_names=['cartpole_custom-v1','halfcheetah_custom-v1']
    env_name=env_names[1]
    env=gym.make(env_name)
    T=env._max_episode_steps #200 #task horizon
    ds=env.observation_space.shape[0] #state dims
    da=env.action_space.shape[0] #action dims
    env_funcs=[make_env(env_name,seed)] * n_workers
    envs=SubprocVecEnv(env_funcs, ds, da, queue, lock)
    
    #models
    in_size=ds
    out_size=da
    policy = PolicyNetwork(in_size,n,h,out_size) #dynamics model
    value_net=ValueNetwork(in_size,T,b,gamma)
    
    #results 
    plot_tr_rewards=[]
    plot_val_rewards=[]
    best_reward=-1e6
    
    set_seed(seed,env)
    
    # %% Implementation
        
    episodes=progress(tr_eps)
    for episode in episodes:
        
        #sample batch of tasks
        tasks = env.sample_tasks(meta_b) 
        rewards_tr_ep, rewards_val_ep = [], []
        Ds, D_dashes=[], []
        
        for task in tasks:
            
            #set env task to current task
            reset_tasks(envs,[task] * n_workers)
            
            #sample b trajectories/rollouts using f_theta
            D=collect_rollout_batch(envs, ds, da, policy, T, b, n_workers, queue)
            Ds.append(D)
            _, _, rewards = D
            rewards_tr_ep.append(rewards)
            
            #compute loss (via: VPG w/ baseline)
            theta_dash=adapt(D,value_net,policy,alpha)
            
            #sample b trajectories/rollouts using f_theta'
            D_dash=collect_rollout_batch(envs, ds, da, policy, T, b, n_workers, queue, params=theta_dash)
            D_dashes.append(D_dash)
            _, _, rewards = D_dash
            rewards_val_ep.append(rewards)
        
        #update meta-params (via: TRPO) 
        #compute surrogate loss
        with tf.GradientTape() as tape:
            prev_loss, _, prev_pis = surrogate_loss(Ds, D_dashes,policy,value_net,gamma,alpha,T)
        grads = parameters_to_vector(tape.gradient(prev_loss, policy.trainable_variables),policy.trainable_variables)
        
        prev_loss=tf.identity(prev_loss)
        hvp=HVP(Ds,D_dashes,policy,value_net,alpha,damping)
        search_step_dir=conjugate_gradients(hvp, grads)
        # max_length=tf.math.sqrt(2.0 * max_grad_kl / tf.tensordot(search_step_dir, hvp(search_step_dir)))
        max_length=np.sqrt(2.0 * max_grad_kl / np.dot(search_step_dir, hvp(search_step_dir)))
        full_step=search_step_dir*max_length        
        prev_params = parameters_to_vector(policy.trainable_variables)
        line_search(policy, prev_loss, prev_pis, value_net, alpha, gamma, T, b, D_dashes, Ds, full_step, prev_params, max_grad_kl, max_backtracks, zeta)
    
        #compute & log results
        # compute rewards
        reward_ep = (tf.math.reduce_mean(tf.stack([tf.math.reduce_mean(tf.math.reduce_sum(rewards, axis=0)) for rewards in rewards_tr_ep], axis=0))).numpy() #sum over T, mean over b, stack horiz one reward per task, mean of tasks
        reward_val=(tf.math.reduce_mean(tf.stack([tf.math.reduce_mean(tf.math.reduce_sum(rewards, axis=0)) for rewards in rewards_val_ep], axis=0))).numpy()
        #save best running model [params]
        if reward_val>best_reward: 
            best_reward=reward_val
            policy.save_weights("saved_models/"+env_name)
        #log iteration results & statistics
        plot_tr_rewards.append(reward_ep)
        plot_val_rewards.append(reward_val)
        if episode % log_ival == 0:
            log_msg="Rewards Tr: {:.2f}, Rewards Val: {:.2f}".format(reward_ep,reward_val)
            episodes.set_description(desc=log_msg); episodes.refresh()
    
    #%% Results & Plot
    title="Meta-Training Training Rewards (Learning Curve)"
    plt.figure(figsize=(16,8))
    plt.grid(1)
    plt.plot(plot_tr_rewards)
    plt.title(title)
    plt.savefig('plots/mtr_tr.png')
    # plt.show()
    
    title="Meta-Training Testing Rewards (Learning Curve)"
    plt.figure(figsize=(16,8))
    plt.grid(1)
    plt.plot(plot_val_rewards)
    plt.title(title)
    plt.savefig('plots/mtr_ts.png')
    # plt.show()
