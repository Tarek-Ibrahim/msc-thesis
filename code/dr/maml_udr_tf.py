# %% TODOs


# %% Imports
#general
import numpy as np
import pandas as pd
import os
# os.environ["OMP_NUM_THREADS"] = "1"

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
from collections import OrderedDict
from scipy.spatial.distance import squareform, pdist

#ML
import tensorflow as tf
import tensorflow_probability as tfp
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

#multiprocessing
import multiprocessing as mp
import queue as Q

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
    
    return D


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


def surrogate_loss(Ds, D_dashes,policy,value_net,gamma,alpha,prev_pis=None):
    
    kls, losses, pis =[], [], []
    if prev_pis is None:
        prev_pis = [None] * len(Ds)
        
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


def line_search(policy, prev_loss, prev_pis, value_net, alpha, gamma, b, D_dashes, Ds, step, prev_params, max_grad_kl, max_backtracks=10, zeta=0.5):
    """backtracking line search"""
    
    for step_frac in [zeta**x for x in range(max_backtracks)]:
        vector_to_parameters(prev_params - step_frac * step, policy.trainable_variables)
        
        loss, kl, _ = surrogate_loss(Ds, D_dashes,policy,value_net,gamma,alpha,prev_pis=prev_pis)
        
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


#%% Main Func
if __name__ == '__main__':
    
    #%% Inputs
    #MAML model / policy
    # n=2 #no. of NN layers
    h=100 #size of hidden layers
    
    #MAML optimizer
    alpha=0.1 #adaptation step size / learning rate
    # beta=0.001 #meta step size / learning rate #TIP: controlled and adapted by TRPO (in maml step) [to guarantee monotonic improvement, etc]
    
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
    
    #UDR
    T_rand_rollout=5
    
    #eval
    evaluate=True
    eval_eps=3
    
    #Env
    env_names=['cartpole_custom-v1', 'halfcheetah_custom-v1', 'halfcheetah_custom_norm-v1', 'halfcheetah_custom_rand-v1', 'halfcheetah_custom_rand-v2', 'lunarlander_custom_default_rand-v0']
    env_name=env_names[-2]
    
    #general
    tr_eps=500 #20 #200 #no. of training episodes/iterations
    log_ival=1 #logging interval
    n_workers = 10 #mp.cpu_count() - 1 #=n_envs
    b = n_workers #20 #16 #32 #batch size: Number of trajectories (rollouts) to sample from each task
    file_name=os.path.basename(__file__).split(".")[0]
    common_name = "_"+file_name+"_"+env_name
    verbose=1 #or: False/True (False/0: display progress bar; True/1: display 1 log newline per episode)
    
    #Seed
    # seeds=[None,1,2,3,4,5]
    seeds=[1,2,3]
    seed = seeds[1]
    
    plot_tr_rewards_all=[]
    plot_val_rewards_all=[]
    plot_eval_rewards_all=[]
    
    for seed in seeds:
        
        print(f"For Seed: {seed} \n")
  
        #%% Initializations
        #multiprocessing
        queue = mp.Queue()
        lock = mp.Lock()
        
        #environment
        env=gym.make(env_name)
        set_seed(seed,env)
        T_env=env._max_episode_steps #200 #task horizon
        ds=env.observation_space.shape[0] #state dims
        da=env.action_space.shape[0] #action dims
        dr=env.unwrapped.randomization_space.shape[0] #N_rand (no. of randomization params)
        
        env_rand=make_vec_envs(env_name, seed, n_workers, queue, lock)
        
        #models
        in_size=ds
        out_size=da
        policy = PolicyNetwork(in_size,h,out_size) #dynamics model
        value_net=ValueNetwork(in_size,gamma)
        
        #results 
        plot_tr_rewards=[]
        plot_val_rewards=[]
        best_reward=-1e6
        total_timesteps=[]
        t_agent = 0
        
        #evaluation
        eval_rewards_mean=0
        eval_freq = n_workers * T_env
        t_eval=0 # agent timesteps since eval
        plot_eval_rewards=[]
 
        #%% Implementation
            
        episodes=progress(tr_eps) if not verbose else range(tr_eps)
        for episode in episodes:
            
            #rollout svpg particles (each particle represents a different rand env and each timestep in that rollout represents different values for its randomization params)
            simulation_instances = -1 * np.ones((n_workers,T_rand_rollout,dr))
            
            #create empty storages
            rewards_tr_ep, rewards_val_ep = [], []
            Ds, D_dashes=[], []
            
            # Reshape to work with vectorized environments
            simulation_instances = np.transpose(simulation_instances, (1, 0, 2))
            
            #inner/adaptation loop
            for t_rand_rollout in range(T_rand_rollout):
                #randomize rand envs (with svpg particle [randomization parameter] values [at the current svpg timestep]) #!!!: this only works if transitions within the same rollout is collected in the same environment (i.e. by the same [env] worker) #that's why we choose: n_particles=b(rollout batch size)=n_workers/n_envs
                env_rand.randomize(simulation_instances[t_rand_rollout])
                
                #collect pre-adaptation rollout batch in rand envs (one rollout for each svpg particle)
                D=collect_rollout_batch(env_rand, ds, da, policy, T_env, b, n_workers, queue)
                Ds.append(D)
                _, _, rewards = D
                rewards_tr_ep.append(rewards)
                
                #adapt agent [meta-]parameters (via VPG w/ baseline)
                theta_dash=adapt(D,value_net,policy,alpha)
                
                #collect post-adaptation rollout batch in rand envs
                D_dash=collect_rollout_batch(env_rand, ds, da, policy, T_env, b, n_workers, queue, params=theta_dash)
                D_dashes.append(D_dash)
                _, _, rewards = D_dash
                rewards_val_ep.append(rewards)
                t_eval += rewards.size
                t_agent += rewards.size
    
            #outer loop: update meta-params (via: TRPO) #!!!: since MAML uses TRPO it is on-policy, so care should be taken that order of associated transitions is preserved
            with tf.GradientTape() as tape:
                prev_loss, _, prev_pis = surrogate_loss(Ds, D_dashes,policy,value_net,gamma,alpha)
            grads = parameters_to_vector(tape.gradient(prev_loss, policy.trainable_variables),policy.trainable_variables)
            prev_loss=tf.identity(prev_loss)
            hvp=HVP(Ds,D_dashes,policy,value_net,alpha,damping)
            search_step_dir=conjugate_gradients(hvp, grads)
            max_length=np.sqrt(2.0 * max_grad_kl / np.dot(search_step_dir, hvp(search_step_dir)))
            full_step=search_step_dir*max_length        
            prev_params = parameters_to_vector(policy.trainable_variables)
            line_search(policy, prev_loss, prev_pis, value_net, alpha, gamma, b, D_dashes, Ds, full_step, prev_params, max_grad_kl, max_backtracks, zeta)
            
            #evaluation
            if evaluate and t_eval>eval_freq:
                t_eval %= eval_freq
                eval_rewards=[]
    
                for _ in range(eval_eps):
                    env.randomize(["random"]*dr)
                    
                    s=env.reset()
                    
                    state=tf.expand_dims(tf.convert_to_tensor(s,dtype=tf.float32),0)
                    dist=policy(state,params=None)
                    a=tf.squeeze(dist.sample()).numpy()
                    s, r, done, _ = env.step(a)
                    
                    R = r
                    
                    while not done:
                        
                        state=tf.expand_dims(tf.convert_to_tensor(s,dtype=tf.float32),0)
                        states=tf.expand_dims(state,0)
                        actions=tf.expand_dims(tf.expand_dims(tf.convert_to_tensor(a),0),0)
                        rewards=tf.expand_dims(tf.expand_dims(tf.convert_to_tensor(np.array(r),dtype=np.float32),0),0)
                        D=[states, actions, rewards]
                                            
                        theta_dash=adapt(D,value_net,policy,alpha)
                        
                        dist=policy(state,params=theta_dash)
                        a=tf.squeeze(dist.sample()).numpy()
                        s, r, done, _ = env.step(a)
                        
                        # env.render()
                        
                        R+=r
                        
                    eval_rewards.append(R)
                
                eval_rewards_mean=np.mean(np.array(eval_rewards).flatten())
                plot_eval_rewards.append(eval_rewards_mean)
        
            #compute & log results
            # compute rewards
            reward_ep = (tf.math.reduce_mean(tf.stack([tf.math.reduce_mean(tf.math.reduce_sum(rewards, axis=0)) for rewards in rewards_tr_ep], axis=0))).numpy() #sum over T, mean over b, stack horiz one reward per task, mean of tasks
            reward_val=(tf.math.reduce_mean(tf.stack([tf.math.reduce_mean(tf.math.reduce_sum(rewards, axis=0)) for rewards in rewards_val_ep], axis=0))).numpy()
            #save best running model [params]
            if eval_rewards_mean>best_reward: 
                best_reward=eval_rewards_mean
                policy.save_weights(f"saved_models/model{common_name}")
            #save plot data
            plot_tr_rewards.append(reward_ep)
            plot_val_rewards.append(reward_val)
            total_timesteps.append(t_agent)
            #log iteration results & statistics
            if episode % log_ival == 0:
                log_msg="Rewards Tr: {:.2f}, Rewards Val: {:.2f}, Rewards Eval: {:.2f}, Total Timesteps: {}".format(reward_ep, reward_val, eval_rewards_mean, t_agent)
                if verbose:
                    print(log_msg+f" episode:{episode} \n")
                else:
                    episodes.set_description(desc=log_msg); episodes.refresh()
        
        plot_tr_rewards_all.append(plot_tr_rewards)
        plot_val_rewards_all.append(plot_val_rewards)
        plot_eval_rewards_all.append(plot_eval_rewards)
        
        env.close()
        env_rand.close()
    
    #%% Results & Plot
    #process results
    plot_tr_rewards_mean = np.stack(plot_tr_rewards_all).mean(0)
    plot_val_rewards_mean = np.stack(plot_val_rewards_all).mean(0)
    plot_eval_rewards_mean = np.stack(plot_eval_rewards_all).mean(0)
    
    plot_tr_rewards_max= np.maximum.reduce(plot_tr_rewards_all)
    plot_val_rewards_max = np.maximum.reduce(plot_val_rewards_all)
    plot_eval_rewards_max = np.maximum.reduce(plot_eval_rewards_all)
    
    plot_tr_rewards_min = np.minimum.reduce(plot_tr_rewards_all)
    plot_val_rewards_min = np.minimum.reduce(plot_val_rewards_all)
    plot_eval_rewards_min = np.minimum.reduce(plot_eval_rewards_all)
    
    #save results to df
    df = pd.DataFrame(list(zip(plot_tr_rewards_mean,
                               plot_tr_rewards_max,
                               plot_tr_rewards_min,
                               plot_val_rewards_mean,
                               plot_val_rewards_max,
                               plot_val_rewards_min,
                               plot_eval_rewards_mean,
                               plot_eval_rewards_max,
                               plot_eval_rewards_min,
                               total_timesteps)),
                      columns =['Rewards_Tr', 'Rewards_Tr_Max', 'Rewards_Tr_Min', 'Rewards_Val', 'Rewards_Val_Max','Rewards_Val_Min', 'Rewards_Eval', 'Rewards_Eval_Max', 'Rewards_Eval_Min', 'Total_Timesteps'])
    df.to_pickle(f"plots/results{common_name}.pkl")
    
    #plot results
    title="Meta-Training Training Rewards"
    plt.figure(figsize=(16,8))
    plt.grid(1)
    plt.plot(plot_tr_rewards_mean)
    plt.fill_between(range(tr_eps), plot_tr_rewards_max, plot_tr_rewards_min,alpha=0.2)
    # plt.axhline(y = env.spec.reward_threshold, color = 'r', linestyle = '--',label='Solved')
    plt.title(title)
    plt.legend(loc="upper right")
    plt.savefig(f'plots/mtr_tr{common_name}.png')
    
    title="Meta-Training Testing Rewards"
    plt.figure(figsize=(16,8))
    plt.grid(1)
    plt.plot(plot_val_rewards_mean)
    plt.fill_between(range(tr_eps), plot_val_rewards_max, plot_val_rewards_min,alpha=0.2)
    # plt.axhline(y = env.spec.reward_threshold, color = 'r', linestyle = '--',label='Solved')
    plt.title(title)
    plt.legend(loc="upper right")
    plt.savefig(f'plots/mtr_ts{common_name}.png')
    
    title="Meta-Testing Rewards"
    plt.figure(figsize=(16,8))
    plt.grid(1)
    plt.plot(plot_eval_rewards_mean)
    plt.fill_between(range(tr_eps), plot_eval_rewards_max, plot_eval_rewards_min,alpha=0.2)
    # plt.axhline(y = env.spec.reward_threshold, color = 'r', linestyle = '--',label='Solved')
    plt.title(title)
    plt.legend(loc="upper right")
    plt.savefig(f'plots/mts{common_name}.png')
    
    #TODO: plot control actions (of which episode(s)?)
        
