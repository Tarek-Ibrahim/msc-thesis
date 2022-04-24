#%% Imports
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.optimizers import Adam
import tqdm
from collections import OrderedDict
from scipy.spatial.distance import squareform, pdist
from baselines.common.vec_env import CloudpickleWrapper
import queue as Q
import gym
#------only for spyder IDE
for env in gym.envs.registration.registry.env_specs.copy():
      if 'custom' in env:
          print('Remove {} from registry'.format(env))
          del gym.envs.registration.registry.env_specs[env]
#------
import gym_custom
import multiprocessing as mp

#%% Common

def progress(x): #for visualizing/monitoring training progress
    return tqdm.trange(x, leave=True)

def set_seed(seed,env):
    import random
    
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    env.seed(seed)

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


def weighted_mean(tensor, axis=None, weights=None):
    if weights is None:
        out = tf.reduce_mean(tensor)
    if axis is None:
        out = tf.reduce_sum(tensor * weights) / tf.reduce_sum(weights)
    else:
        mean_dim = tf.reduce_sum(tensor * weights, axis=axis) / (tf.reduce_sum(weights, axis=axis))
        out = tf.reduce_mean(mean_dim)
    return out


def weighted_normalize(tensor, axis=None, weights=None, epsilon=1e-8):
    mean = weighted_mean(tensor, axis=axis, weights=weights)
    num = tensor * (1 if weights is None else weights) - mean
    std = tf.math.sqrt(weighted_mean(num ** 2, axis=axis, weights=weights))
    out = num/(std + epsilon)
    return out


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


def collect_rollout_batch(envs, ds, da, policy, T, b, n_workers, queue, params=None,add_noise=False,noise_scale=0.1): # a batch of rollouts
    states=[[] for _ in range(b)]
    rewards = [[] for _ in range(b)]
    actions = [[] for _ in range(b)]
    next_states = [[] for _ in range(b)]
    
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
            # a = a.clip(-1, 1)
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
    
    T_max=max(map(len,rewards))
    states_mat=np.zeros((T_max,b,ds),dtype=np.float32)
    actions_mat=np.zeros((T_max,b,da),dtype=np.float32)
    rewards_mat=np.zeros((T_max,b),dtype=np.float32)
    masks_mat=np.zeros((T_max,b),dtype=np.float32)
    
    for rollout_idx in range(b):
        T_rollout=len(rewards[rollout_idx])
        states_mat[:T_rollout,rollout_idx]= np.stack(states[rollout_idx])
        actions_mat[:T_rollout,rollout_idx]= np.stack(actions[rollout_idx])
        rewards_mat[:T_rollout,rollout_idx]= np.stack(rewards[rollout_idx])
        masks_mat[:T_rollout,rollout_idx]=1.0
    
    D=[states_mat, actions_mat, rewards_mat, masks_mat]
    
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


def make_vec_envs(env_name, seed, n_workers, ds, da, queue, lock):
    envs=[make_env(env_name,seed,rank) for rank in range(n_workers)]
    envs=SubprocVecEnv(envs, ds, da, queue, lock)
    return envs

def reset_tasks(envs,tasks):
    return all(envs.reset_task(tasks))

#%% DDPG

class Actor(tf.keras.Model):
    def __init__(self, in_size, h1, h2, out_size, max_action):
        super(Actor, self).__init__()
        
        self.actor=tf.keras.models.Sequential(layers=[
            tf.keras.layers.Input(in_size),
            tf.keras.layers.Dense(h1,activation="relu"),
            tf.keras.layers.Dense(h2,activation="relu"),
            tf.keras.layers.Dense(out_size)
            ])

        self.max_action = max_action

    def call(self, x):
        x=self.max_action * tf.keras.activations.tanh(self.actor(x)) if self.max_action > 0 else self.actor(x)
        return x


class Critic(tf.keras.Model):
    def __init__(self, in_size, h1, h2):
        super(Critic, self).__init__()
        
        self.critic=tf.keras.models.Sequential(layers=[
            tf.keras.layers.Input(in_size),
            tf.keras.layers.Dense(h1,activation="relu"),
            tf.keras.layers.Dense(h2,activation="relu"),
            tf.keras.layers.Dense(1)
            ])

    def call(self, x, u):
        x=self.critic(tf.concat([x, u], 1))
        return x 
                
class DDPG(object):
    def __init__(self, in_size, out_size, h1, h2, lr_agent, batch_size, epochs, a_max=1., tau=0.005, gamma=0.99):
        
        self.gamma = gamma
        self.epochs = epochs
        self.batch_size = batch_size
        self.tau=tau
        self.a_max = a_max
        
        self.dr=in_size
        self.last_states = np.random.uniform(0, 1, (self.dr,))
        self.timesteps = 0
        
        self.actor = Actor(in_size=in_size, h1=h1, h2=h2, out_size=out_size, max_action=a_max)
        self.actor_target = Actor(in_size=in_size, h1=h1, h2=h2, out_size=out_size, max_action=a_max)
        self.actor_target.set_weights(self.actor.get_weights())
        self.actor_optimizer = Adam(learning_rate=lr_agent)

        self.critic = Critic(in_size=in_size+out_size, h1=h1, h2=h2) 
        self.critic_target = Critic(in_size=in_size+out_size, h1=h1, h2=h2) 
        self.critic_target.set_weights(self.critic.get_weights())
        self.critic_optimizer = Adam(learning_rate=lr_agent*10.)
    
    def select_action(self, state):
        state = tf.convert_to_tensor(state,dtype=tf.float32) 
        return self.actor(state).numpy()
    
    def step(self,RB, T_svpg, T_init, H_svpg):
        
        self.simulation_instance = np.zeros((T_svpg, self.dr))
        RB_samples=[]
        #reset
        current_sim_params = self.last_states
        done=False
        # add_to_buffer=True

        for t in range(T_svpg):
            self.simulation_instance[t] = current_sim_params

            if len(RB.storage) > T_init:
                action = self.select_action(current_sim_params)
            elif self.a_max > 0:
                action = self.a_max * np.random.uniform(-1, 1, (self.dr,))
            else:
                action = np.random.uniform(-1, 1, (self.dr,))
            
            #step
            next_params = current_sim_params + action
            reward = 1.
            done=True if next_params < 0 or next_params > 1 else False
            # next_params = np.clip(next_params,0,1)
            done_bool = 0 if self.timesteps + 1 == H_svpg else float(done)
            
            # if add_to_buffer:
            RB_samples.append([current_sim_params, next_params, action, reward, done_bool])
            
            if done_bool:
                current_sim_params = np.random.uniform(0, 1, (self.dr,))                
                self.timesteps = 0
                # add_to_buffer=False
                # break
            else:
                current_sim_params = next_params
                self.timesteps += 1

        self.last_states = current_sim_params

        return np.array(self.simulation_instance), RB_samples

    def train(self, RB):
        for epoch in range(self.epochs):
            # Sample replay buffer 
            x, y, u, r, d = RB.sample(self.batch_size)
            state = tf.convert_to_tensor(x,dtype=tf.float32) 
            action = tf.convert_to_tensor(u,dtype=tf.float32) 
            next_state = tf.convert_to_tensor(y,dtype=tf.float32) 
            done = tf.convert_to_tensor(1-d,dtype=tf.float32) 
            reward = tf.convert_to_tensor(r,dtype=tf.float32)  

            # Compute the target Q value
            target_Q = self.critic_target(next_state, self.actor_target(next_state))
            target_Q = reward + (done * self.gamma * target_Q)

            
            with tf.GradientTape() as tape:
                # Get current Q estimate
                current_Q = self.critic(state, action)
    
                # Compute critic loss
                critic_loss = tf.keras.losses.MeanSquaredError()(current_Q, target_Q)

            # Optimize the critic
            gradients = tape.gradient(critic_loss, self.critic.trainable_variables) #calculate gradient
            self.critic_optimizer.apply_gradients(zip(gradients, self.critic.trainable_variables)) #backpropagate

            # Compute actor loss
            with tf.GradientTape() as tape:
                actor_loss = -tf.reduce_mean(self.critic(state, self.actor(state)))
            
            # Optimize the actor
            gradients = tape.gradient(actor_loss, self.actor.trainable_variables) #calculate gradient
            self.actor_optimizer.apply_gradients(zip(gradients, self.actor.trainable_variables)) #backpropagate
            
            # Update the frozen target models
            for param, target_param in zip(self.critic.trainable_variables, self.critic_target.trainable_variables):
                target_param.assign(self.tau * param + (1 - self.tau) * target_param)

            for param, target_param in zip(self.actor.trainable_variables, self.actor_target.trainable_variables):
                target_param.assign(self.tau * param + (1 - self.tau) * target_param)


#%% Discriminator

class Disc_MLP(tf.keras.Model):
    def __init__(self, in_size, h1, h2):
        super(Disc_MLP, self).__init__()
        
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


class Discriminator(object): #basically: a binary classifier 
    def __init__(self, ds, da, h, b_disc, r_disc_scale, lr_disc):
        self.discriminator = Disc_MLP(in_size=ds+da+ds, h1=h, h2=h)  #Input: state-action-state' transition; Output: probability that it was from a reference trajectory

        self.disc_loss_func = tf.keras.losses.BinaryCrossentropy()
        self.disc_optimizer = Adam(learning_rate=lr_disc)
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
            randind = np.random.randint(0, len(rand_traj), size=int(self.batch_size))
            refind = np.random.randint(0, len(ref_traj), size=int(self.batch_size))
            
            with tf.GradientTape() as tape:
                rand_batch = tf.convert_to_tensor(rand_traj[randind],dtype=tf.float32)
                ref_batch = tf.convert_to_tensor(ref_traj[refind],dtype=tf.float32)
    
                g_o = self.discriminator(rand_batch)
                e_o = self.discriminator(ref_batch)
                
                disc_loss = self.disc_loss_func(g_o, tf.ones((len(rand_batch), 1))) + self.disc_loss_func(e_o,tf.zeros((len(ref_batch), 1))) #ref label = 0; rand label = 1
            
            gradients = tape.gradient(disc_loss, self.discriminator.trainable_variables) #calculate gradient
            self.disc_optimizer.apply_gradients(zip(gradients, self.discriminator.trainable_variables)) #backpropagate

#%% Active DR

def map_rewards(map_type,reward_scale,rand_reward,ref_reward):
    
    if "neg" in map_type:
        simulator_reward = -rand_reward
    elif "delta" in map_type:
        simulator_reward = ref_reward-rand_reward #assumes thr_r >0.
    
    return reward_scale * simulator_reward

#%% MAML

def adapt(D,value_net,policy,alpha):
    
    #unpack
    states, actions, rewards, masks = D
    
    value_net.fit_params(states,rewards,masks)
    
    with tf.GradientTape() as tape:
        advantages=compute_advantages(states,rewards,value_net,value_net.gamma,masks)
        pi=policy(states)
        log_probs=pi.log_prob(actions)
        if len(log_probs.shape) > 2:
            log_probs = tf.reduce_sum(log_probs, axis=2)
        loss=-weighted_mean(log_probs*advantages,axis=0,weights=masks)
                
    #compute adapted params (via: GD) --> perform 1 gradient step update
    grads = tape.gradient(loss, policy.trainable_variables)
    theta_dash=policy.update_params(grads, alpha) 
    
    return theta_dash 

#%% TRPO (& PPO & VPG)

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
        
    def fit_params(self, states, rewards, masks):
        
        T_max=states.shape[0]
        b=states.shape[1]
        ones = tf.expand_dims(masks,2)
        states = states * ones
        timestep= tf.math.cumsum(ones, axis=0) / 100.
        
        reg_coeff = self.reg_coeff
        
        #create features
        features = tf.concat([states, states **2, timestep, timestep**2, timestep**3, ones],axis=2)
        features=tf.reshape(features, (-1, self.feature_size))

        #compute returns        
        G = np.zeros(b,dtype=np.float32)
        returns = np.zeros((T_max,b),dtype=np.float32)
        for t in range(T_max - 1, -1, -1):
            G = rewards[t]*masks[t]+self.gamma*G
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
        
    def call(self, states, masks):
        
        ones = tf.expand_dims(masks,2)
        states = states * ones
        timestep= tf.math.cumsum(ones, axis=0) / 100.
        
        features = tf.concat([states, states **2, timestep, timestep**2, timestep**3, ones],axis=2)
        
        return tf.linalg.matmul(features,self.w)


def compute_advantages(states,rewards,value_net,gamma,masks,returns=False):
    
    T_max=states.shape[0]
    
    values = value_net(states,masks)
    if len(list(values.shape))>2: values = tf.squeeze(values, axis=2)
    values = tf.pad(values*masks,[[0, 1], [0, 0]])
    
    deltas = rewards + gamma * values[1:] - values[:-1] #delta = r + gamma * v - v' #TD error
    # advantages = tf.zeros_like(deltas, dtype=tf.float32)
    advantages = tf.TensorArray(tf.float32, *deltas.shape)
    advantage = tf.zeros_like(deltas[0], dtype=tf.float32)
    
    for t in range(T_max - 1, -1, -1): #reversed(range(-1,T -1 )):
        
        advantage = advantage * gamma + deltas[t] if not returns else rewards[t]+gamma*advantage*masks[t]
        advantages = advantages.write(t, advantage)
        # advantages[t] = advantage
    
    advantages = advantages.stack()
    
    #Normalize advantages to improve: learning, numerical stability & convergence
    advantages = weighted_normalize(advantages,weights=masks)
    
    return advantages


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


def surrogate_loss(D_dashes,policy,value_net,gamma,alpha=None,prev_pis=None, Ds=None, alg="trpo", clip=0.2):
    
    kls, losses, pis =[], [], []
    if Ds is None:
        Ds = [None] * len(D_dashes)
        
    if prev_pis is None:
        prev_pis = [None] * len(D_dashes)
        
    for D, D_dash, prev_pi in zip(Ds,D_dashes,prev_pis):

        states, actions, rewards, masks = D_dash
        
        if D is not None:
            theta_dash=adapt(D,value_net,policy,alpha)
            pi=policy(states,params=theta_dash)
        else:
            value_net.fit_params(states,rewards,masks)
            pi=policy(states)
        
        pis.append(detach_dist(pi))
        
        if prev_pi is None:
            prev_pi = detach_dist(pi)
        
        advantages=compute_advantages(states,rewards,value_net,gamma,masks)
        
        ratio=pi.log_prob(actions)-prev_pi.log_prob(actions)
        if len(ratio.shape) > 2:
            ratio = tf.reduce_sum(ratio, axis=2)
        ratio = tf.exp(ratio)
        surr1= ratio * advantages
        if alg=="trpo":
            loss = surr1
        elif alg=="ppo":
            surr2=tf.clip_by_value(ratio,1-clip,1+clip)*advantages
            loss = tf.minimum(surr1,surr2)
            
        loss = - weighted_mean(loss,axis=0,weights=masks)
        losses.append(loss)
        
        if len(actions.shape) > 2:
            masks = tf.expand_dims(masks, axis=2)
        #???: which version is correct?
        # kl=tf.math.reduce_mean(pi.kl_divergence(prev_pi))
        # kl=tf.math.reduce_mean(prev_pi.kl_divergence(pi))
        kl= weighted_mean(prev_pi.kl_divergence(pi),axis=0,weights=masks)
        # kl= weighted_mean(prev_pi.kl_divergence(pi),weights=masks)
        
        kls.append(kl)
    
    prev_loss=tf.math.reduce_mean(tf.stack(losses, axis=0))
    kl=tf.math.reduce_mean(tf.stack(kls, axis=0)) #???: why is it always zero?
    
    return prev_loss, kl, pis


def line_search(policy, prev_loss, prev_pis, value_net, gamma, b, D_dashes, step, prev_params, max_grad_kl, max_backtracks=10, zeta=0.5, alpha=None, Ds=None):
    """backtracking line search"""
    
    for step_frac in [zeta**x for x in range(max_backtracks)]:
        vector_to_parameters(prev_params - step_frac * step, policy.trainable_variables)
        
        loss, kl, _ = surrogate_loss(D_dashes,policy,value_net,gamma,alpha=alpha,prev_pis=prev_pis,Ds=Ds)
        
        #check improvement
        actual_improve = loss - prev_loss
        if not np.isfinite(loss):
            raise RuntimeError('NANs/Infs encountered in line search')
        if (actual_improve.numpy() < 0.0) and (kl.numpy() < max_grad_kl):
            break
    else:
        vector_to_parameters(prev_params, policy.trainable_variables)


def kl_div(D_dashes,value_net,policy,alpha=None,Ds=None):
    kls=[]
    
    if Ds is None:
        Ds = [None] * len(D_dashes)
                          
    for D, D_dash in zip(Ds,D_dashes):
        
        states, actions, _, masks = D_dash
        
        if D is not None:
            theta_dash=adapt(D,value_net,policy,alpha)
        else:
            theta_dash=None
        
        pi=policy(states,params=theta_dash)
        
        prev_pi = detach_dist(pi)
        
        if len(actions.shape) > 2:
            masks = tf.expand_dims(masks, axis=2)
        #???: which version is correct?
        # kl=tf.math.reduce_mean(pi.kl_divergence(prev_pi))
        # kl=tf.math.reduce_mean(prev_pi.kl_divergence(pi))
        kl= weighted_mean(prev_pi.kl_divergence(pi),axis=0,weights=masks)
        # kl= weighted_mean(prev_pi.kl_divergence(pi),weights=masks)
        
        kls.append(kl)
    
    return tf.math.reduce_mean(tf.stack(kls, axis=0))


def HVP(D_dashes,policy,value_net,damping,alpha=None,Ds=None):
    def _HVP(v):
        with tf.GradientTape() as outer_tape:
             with tf.GradientTape() as inner_tape:
                 kl = kl_div(D_dashes,value_net,policy,alpha,Ds)
             grad_kl=parameters_to_vector(inner_tape.gradient(kl,policy.trainable_variables),policy.trainable_variables)
             dot=tf.tensordot(grad_kl, v, axes=1)
        return parameters_to_vector(outer_tape.gradient(dot, policy.trainable_variables),policy.trainable_variables) + damping * v            
    return _HVP

#%% SAC

class SAC_MLP(tf.keras.Model):
    def __init__(self, in_size, h, out_size):
        super(SAC_MLP, self).__init__()
        
        self.mlp=tf.keras.models.Sequential(layers=[
            tf.keras.layers.Input(in_size),
            tf.keras.layers.Dense(h,activation="relu"),
            tf.keras.layers.Dense(h,activation="relu"),
            tf.keras.layers.Dense(out_size)
            ])

    def call(self, *inputs):
        inputs=tf.concat(inputs,1)
        outputs = self.mlp(inputs)
        return outputs


class SAC_Policy(SAC_MLP):
    def __init__(self, in_size, h, out_size, a_max, a_min):
        super().__init__(in_size, h, out_size)
        
        self.action_scale=tf.convert_to_tensor((a_max - a_min)/2.,dtype=tf.float32)
        self.action_bias=tf.convert_to_tensor((a_max + a_min)/2.,dtype=tf.float32)
    
    def call(self, state):
        output = self.mlp(state)
        mean = output[..., :output.shape[-1] // 2]
        log_std=output[..., output.shape[-1] // 2:]
        log_std = tf.clip_by_value(log_std, -20, 2)
        
        std = tf.exp(log_std)
        dist = tfp.distributions.Normal(mean,std)
         
        x_t = mean + std * tfp.distributions.Normal(tf.zeros_like(mean), tf.ones_like(std)).sample() # for reparameterization trick (mean + std * N(0,1)) #???: should it be wrapped in a tf.stop_gradient()?       
        y_t = tf.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = dist.log_prob(x_t) - tf.math.log(self.action_scale * (1 - tf.pow(y_t,2)) + 1e-6)
        log_prob = tf.reduce_sum(log_prob,1, keepdims=True)
        mean = tf.tanh(mean) * self.action_scale + self.action_bias
        
        return action, log_prob, mean


class SAC(object):
    def __init__(self,in_size, out_size,h, lr,batch_size,epochs,alpha,delta_max,entropy_tuning_method, T_alpha_init,tau=0.005, gamma=0.99, alpha_min=0.0001, alpha_discount=0.1):
        
        self.batch_size=batch_size
        self.epochs=epochs
        self.tau=tau
        self.gamma=gamma
        self.delta_max=delta_max
        self.dr = in_size
        
        self.alpha=alpha
        self.entropy_tuning_method=entropy_tuning_method
        self.alpha_min=alpha_min
        self.alpha_discount=alpha_discount
        self.T_alpha_init=T_alpha_init
        
        self.last_states = np.random.uniform(0, 1, (self.dr,))
        self.timesteps = 0
        
        a_max=np.array([delta_max])
        a_min=np.array([-delta_max])
        
        #Q-Functions
        self.q1=SAC_MLP(in_size=in_size+out_size,h=h,out_size=1)
        self.q2=SAC_MLP(in_size=in_size+out_size,h=h,out_size=1)
        
        self.q1_target=SAC_MLP(in_size=in_size+out_size,h=h,out_size=1)
        self.q2_target=SAC_MLP(in_size=in_size+out_size,h=h,out_size=1)
        
        self.q1_target.set_weights(self.q1.get_weights())
        self.q2_target.set_weights(self.q2.get_weights())
        
        self.q1_optimizer = Adam(lr*10.)
        self.q2_optimizer = Adam(lr*10.)
        
        #Policy
        self.pi=SAC_Policy(in_size, h, 2*out_size, a_max, a_min)
        self.pi_optimizer=Adam(lr)
        
        #Temperature / Entropy (Automatic Tuning)
        if self.entropy_tuning_method=="learn":
            self.target_entropy = -tf.reduce_prod(tf.convert_to_tensor(self.dr)).numpy()
            self.log_alpha = tf.Variable(initial_value=tf.keras.initializers.Zeros()(shape=[1]),dtype=tf.float32,trainable=True)
            self.alpha_optimizer = Adam(lr)
        
    def select_action(self, state):
        state = tf.expand_dims(tf.convert_to_tensor(state),0)
        action, _, _ = self.pi(state)
        return action.numpy()[0]
    
    def step(self,RB,T_svpg,T_init,H_svpg):
        
        self.simulation_instance = np.zeros((T_svpg, self.dr))
        #reset
        current_sim_params = self.last_states
        # done=False
        # add_to_buffer=True
        RB_samples=[]

        for t in range(T_svpg):
            self.simulation_instance[t] = current_sim_params

            action = self.select_action(current_sim_params) if len(RB.storage) > T_init else self.delta_max * np.random.uniform(-1, 1, (self.dr,))
            
            #step
            next_params = current_sim_params + action
            reward = 1.
            done=True if next_params < 0. or next_params > 1. else False
            # next_params = np.clip(next_params,0,1)
            done_bool = 0 if self.timesteps + 1 == H_svpg else float(done)
            
            RB_samples.append([current_sim_params, next_params, action, reward, done_bool])
            
            if done_bool:
                current_sim_params = np.random.uniform(0, 1, (self.dr,))                
                self.timesteps = 0
                # add_to_buffer=False
                # break
            else:
                current_sim_params = next_params
                self.timesteps += 1

        self.last_states = current_sim_params

        return np.array(self.simulation_instance), RB_samples
    
    
    def train(self,RB):
        
        if len(RB.storage) > self.T_alpha_init and self.entropy_tuning_method=="anneal":
            self.alpha *= self.alpha_discount
            if self.alpha < self.alpha_min:
                self.alpha = self.alpha_min
                    
        for epoch in range(self.epochs):
           
            # Sample replay buffer 
            x, y, u, r, d = RB.sample(self.batch_size)
            state = tf.convert_to_tensor(x,dtype=tf.float32) 
            action = tf.convert_to_tensor(u,dtype=tf.float32) 
            next_state = tf.convert_to_tensor(y,dtype=tf.float32) 
            done = tf.convert_to_tensor(1-d,dtype=tf.float32) 
            reward = tf.convert_to_tensor(r,dtype=tf.float32) 
            
            next_state_action, next_state_log_pi, _ = self.pi(next_state)
            q1_next_target = self.q1_target(next_state, next_state_action)
            q2_next_target = self.q2_target(next_state, next_state_action)
            min_q_next_target = tf.minimum(q1_next_target, q2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward + done * self.gamma * (min_q_next_target)
            
            with tf.GradientTape() as tape_q1:
                q1_value=self.q1(state,action)
                q1_loss = tf.keras.losses.MeanSquaredError()(q1_value, next_q_value)
                
            gradients = tape_q1.gradient(q1_loss, self.q1.trainable_variables) #calculate gradient
            self.q1_optimizer.apply_gradients(zip(gradients, self.q1.trainable_variables)) #backpropagate
            
            with tf.GradientTape() as tape_q2:
                q2_value=self.q2(state,action)
                q2_loss = tf.keras.losses.MeanSquaredError()(q2_value, next_q_value)
            
            gradients = tape_q2.gradient(q2_loss, self.q2.trainable_variables) #calculate gradient
            self.q2_optimizer.apply_gradients(zip(gradients, self.q2.trainable_variables)) #backpropagate
            
            with tf.GradientTape() as tape_pi:
                actions_new, log_pi, _ = self.pi(state)
                q_actions_new = tf.minimum(self.q1(state,actions_new),self.q2(state,actions_new))
                policy_loss = tf.reduce_mean((self.alpha* log_pi) - q_actions_new)
    
            gradients = tape_pi.gradient(policy_loss, self.pi.trainable_variables) #calculate gradient
            self.pi_optimizer.apply_gradients(zip(gradients, self.pi.trainable_variables)) #backpropagate
            
            if self.entropy_tuning_method=="learn":
                
                with tf.GradientTape() as tape_alpha:
                    alpha_loss = -tf.reduce_mean(self.log_alpha * tf.stop_gradient(log_pi + self.target_entropy))
                
                gradients = tape_alpha.gradient(alpha_loss, [self.log_alpha]) #calculate gradient
                self.alpha_optimizer.apply_gradients(zip(gradients, [self.log_alpha])) #backpropagate
    
                self.alpha = tf.exp(self.log_alpha)
            
            # Update the frozen target models
            for param, target_param in zip(self.q1.trainable_variables, self.q1_target.trainable_variables):
                target_param.assign(self.tau * param + (1 - self.tau) * target_param)

            for param, target_param in zip(self.q2.trainable_variables, self.q2_target.trainable_variables):
                target_param.assign(self.tau * param + (1 - self.tau) * target_param)

#%% SVPG (A2C- & DDPG-based)

class SVPGParticleCritic(tf.keras.Model):
    def __init__(self, in_size, h):
        super(SVPGParticleCritic, self).__init__()
        
        self.critic=tf.keras.models.Sequential(layers=[
            tf.keras.layers.Input(in_size),
            tf.keras.layers.Dense(h,activation="tanh"),
            tf.keras.layers.Dense(h,activation="tanh"),
            tf.keras.layers.Dense(1)
            ])

    def call(self, x):
        return self.critic(x)


class SVPGParticleActor(tf.keras.Model):
    def __init__(self, in_size, h,out_size):
        super(SVPGParticleActor, self).__init__()
        
        self.actor_base=tf.keras.models.Sequential(layers=[
            tf.keras.layers.Input(in_size),
            tf.keras.layers.Dense(h,activation="tanh"),
            tf.keras.layers.Dense(h,activation="tanh"), 
            tf.keras.layers.Dense(out_size,activation="tanh")
            ])
        
        self.logstd = tf.Variable(initial_value=np.log(1.0)* tf.keras.initializers.Ones()(shape=[out_size,1])-1,dtype=tf.float32)

    def call(self, x):
        mean= self.actor_base(x)
        
        std= tf.math.exp(tf.math.maximum(self.logstd, np.log(1e-6)))
        
        dist=tfp.distributions.Normal(mean,std)
        
        return dist     
    

class SVPG:
    """
    Input: current randomization settings
    Output: either a direction to move in (Discrete - for 1D/2D) or a delta across all parameters (Continuous)
    """
    def __init__(self, n_particles, dr, h, delta_max, T_svpg, H_svpg, temperature, lr_svpg, svpg_kernel_mode,gamma_svpg=0.99, tau=0.005, epochs=30, batch_size=256, T_init=100,base_alg="a2c"):
        
        self.svpg_kernel_mode=svpg_kernel_mode
        self.temperature = temperature 
        self.n_particles = n_particles
        self.T_svpg = T_svpg
        self.dr = dr
        self.H_svpg = H_svpg
        self.delta_max = delta_max
        self.gamma = gamma_svpg
        self.tau=tau
        self.base_alg=base_alg
        
        self.last_states = np.random.uniform(0, 1, (self.n_particles, self.dr))
        self.timesteps = np.zeros(self.n_particles)
        
        self.epochs = epochs
        self.batch_size = batch_size
        self.T_init = T_init
        
        self.particles_actor = []
        self.particles_critic = []
        self.particles_log_probs = [[] for _ in range(n_particles)]
        
        self.particles_actor_target = []
        self.particles_critic_target = []
        
        self.optimizers_actor = []
        self.optimizers_critic = []
        

        for i in range(self.n_particles):
            
            # Initialize each of the individual particles
            actor = SVPGParticleActor(in_size=dr, h=h,out_size=dr) if base_alg=="a2c" else Actor(in_size=dr, h1=h, h2=h, out_size=dr, max_action=delta_max)
            actor_optimizer = Adam(learning_rate=lr_svpg)
            critic = SVPGParticleCritic(in_size=dr, h=h) if base_alg=="a2c" else Critic(in_size=2*dr, h1=h, h2=h)
            critic_optimizer = Adam(learning_rate=lr_svpg*10.)
            
            self.particles_actor.append(actor)
            self.particles_critic.append(critic)
            self.optimizers_actor.append(actor_optimizer)
            self.optimizers_critic.append(critic_optimizer)
            
            if base_alg=="ddpg":
                actor_target = Actor(in_size=dr, h1=h, h2=h, out_size=dr, max_action=delta_max)
                actor_target.set_weights(actor.get_weights())
                critic_target = Critic(in_size=2*dr, h1=h, h2=h) 
                critic_target.set_weights(critic.get_weights())
                
                self.particles_actor_target.append(actor_target)
                self.particles_critic_target.append(critic_target)

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
        
        if self.svpg_kernel_mode==1:
            grad_k= grad_k + tf.matmul(tf.expand_dims(sum_k,0),X)
            
            # grad_k_new=[]
            # for i in range(X.shape[1]):
            #     grad_k_new.append(grad_k[:, i] + tf.tensordot(X[:, i],sum_k,1))
            
            # grad_k=tf.stack(grad_k_new,1)
            
        elif self.svpg_kernel_mode==2:
            grad_k= grad_k + tf.multiply(X,tf.expand_dims(sum_k,1))
        else:
            RuntimeError("unknown svpg kernel mode")

        grad_k = grad_k / (h ** 2)

        return k, grad_k

    def select_action(self, policy_idx, state):
        
        if self.base_alg=="a2c":
            state = tf.expand_dims(tf.convert_to_tensor(state,dtype=tf.float32),0)
            dist = self.particles_actor[policy_idx](state)
        
        
            value = self.particles_critic[policy_idx](state)
    
            action = tf.stop_gradient(dist.sample())
            self.particles_log_probs[policy_idx].append(dist.log_prob(action))
        
            action = tf.squeeze(action).numpy()
            
            return action, value
            
        elif self.base_alg=="ddpg":
            state=tf.convert_to_tensor(state,dtype=tf.float32)
            action=self.particles_actor[policy_idx](state).numpy()

            return action

    def compute_returns(self, next_value, rewards, masks):
        return_ = next_value 
        returns = []
        for step in reversed(range(len(rewards))):
            # Eq. 80: https://arxiv.org/abs/1704.06440
            return_ = self.gamma * masks[step] * return_ + rewards[step]
            returns.insert(0, return_)

        return returns

    def step(self,RB=None):
        """Rollout trajectories, starting from random initializations of randomization settings (i.e. current_sim_params), each of T_svpg size
        Then, send it to agent for further training and reward calculation
        """
        self.simulation_instances = np.zeros((self.n_particles, self.T_svpg, self.dr))

        # Store the values of each state - for advantage estimation
        self.values=[[] for _ in range(self.n_particles)]
        self.tapes=[]
        RB_samples=[[] for i in range(self.n_particles)] if RB is not None else None
        # Store the last states for each particle (calculating rewards)
        self.masks = np.ones((self.n_particles, self.T_svpg))

        for i in range(self.n_particles):
            with tf.GradientTape(persistent=True) as tape:
                
                self.particles_log_probs[i] = []
                current_sim_params = self.last_states[i]
                
                for t in range(self.T_svpg):
                    self.simulation_instances[i][t] = current_sim_params
                    
                    if self.base_alg=="a2c":
                        action, value = self.select_action(i, current_sim_params)
                        action = self.delta_max*  np.array(np.clip(action, -1, 1))
                        next_params = np.clip(current_sim_params + action, 0, 1) #step
                        
                        self.values[i].append(value)
                        
                        done_bool = True if np.array_equal(next_params, current_sim_params) or self.timesteps[i] + 1 == self.H_svpg else False
                        
                    elif self.base_alg=="ddpg":
                        if len(RB.storage) > self.T_init:
                            action = self.select_action(i, current_sim_params)
                        elif self.delta_max > 0:
                            action = self.delta_max * np.random.uniform(-1, 1, (self.dr,))
                        else:
                            action = np.random.uniform(-1, 1, (self.dr,))
                        
                        next_params = current_sim_params + action
                        
                        reward = 1.
                        done=True if next_params < 0. or next_params > 1. else False
                        done_bool = 0 if self.timesteps[i] + 1 == self.H_svpg else float(done)
                        
                        RB_samples[i].append([current_sim_params, next_params, action, reward, done_bool])
    
                    if done_bool:
                        current_sim_params = np.random.uniform(0, 1, (self.dr,))                
                        self.timesteps[i] = 0
                    else:
                        current_sim_params = next_params
                        self.timesteps[i] += 1

                self.last_states[i] = current_sim_params
            self.tapes.append(tape)

        return np.array(self.simulation_instances), RB_samples
    
    
    def train(self,arg):
        if self.base_alg=="a2c":
            return self._train_a2c(arg)
        elif self.base_alg=="ddpg":
            return self._train_ddpg(arg)
    
    def _train_a2c(self, simulator_rewards):
        
        policy_grads = []
        parameters = []
        
        for i in range(self.n_particles):
            
            # policy_grad_particle = []
            
            # Calculate the value of last state - for Return Computation
            _, next_value = self.select_action(i, self.last_states[i]) 

            particle_rewards = tf.convert_to_tensor(simulator_rewards[i],dtype=tf.float32)
            masks = tf.convert_to_tensor(self.masks[i],dtype=tf.float32)
            
            # Calculate entropy-augmented returns, advantages
            returns = self.compute_returns(next_value, particle_rewards, masks)
            returns = tf.stop_gradient(tf.concat(returns,0))
            
            with self.tapes[i] as tape:
                advantages = returns - tf.concat(self.values[i],0)
                
                # Compute value loss, update critic
                critic_loss = 0.5 * tf.reduce_mean(tf.square(advantages))
                
                # Store policy gradients for SVPG update
                policy_grad = tf.reduce_mean(tf.concat(self.particles_log_probs[i][:-1],0) * tf.stop_gradient(advantages))
                # for log_prob, advantage in zip(self.particles[i].saved_log_probs, advantages):
                #     policy_grad_particle.append(log_prob * tf.stop_gradient(advantage))
                # policy_grad = tf.reduce_mean(tf.concat(policy_grad_particle,0))
                
            gradients_c = tape.gradient(critic_loss, self.particles_critic[i].trainable_variables) #calculate gradient
            self.optimizers_critic[i].apply_gradients(zip(gradients_c, self.particles_critic[i].trainable_variables))
            
            gradients_p = tape.gradient(policy_grad, self.particles_actor[i].trainable_variables)

            # Vectorize parameters and PGs
            vec_param, vec_policy_grad = params2vec(self.particles_actor[i].trainable_variables, gradients_p)

            policy_grads.append(tf.expand_dims(vec_policy_grad,0))
            parameters.append(tf.expand_dims(vec_param,0))

        # calculating the kernel matrix and its gradients
        parameters = tf.concat(parameters,0)
        k, grad_k = self.compute_kernel(parameters)

        policy_grads = 1. / self.temperature * tf.concat(policy_grads,0)
        grad_logp = tf.linalg.matmul(k, policy_grads)

        grad_theta = - (grad_logp + grad_k) / self.n_particles

        # update param gradients
        for i in range(self.n_particles):
            grads=vec2params(grad_theta[i],self.particles_actor[i].trainable_variables)
            self.optimizers_actor[i].apply_gradients(zip(grads, self.particles_actor[i].trainable_variables))

    def _train_ddpg(self, RB):
        
        for _ in range(self.epochs):
    
            policy_grads = []
            parameters = []
    
            for i in range(self.n_particles):
                
                x, y, u, r, d = RB.sample(self.batch_size)
                state = tf.convert_to_tensor(x,dtype=tf.float32) 
                action = tf.convert_to_tensor(u,dtype=tf.float32) 
                next_state = tf.convert_to_tensor(y,dtype=tf.float32) 
                done = tf.convert_to_tensor(1-d,dtype=tf.float32) 
                reward = tf.convert_to_tensor(r,dtype=tf.float32)  
                
                # Compute the target Q value
                target_Q = self.particles_critic_target[i](next_state, self.particles_actor_target[i](next_state))
                target_Q = reward + (done * self.gamma * target_Q)
    
                with tf.GradientTape() as tape:
                    # Get current Q estimate
                    current_Q = self.particles_critic[i](state, action)
        
                    # Compute critic loss
                    critic_loss = tf.keras.losses.MeanSquaredError()(current_Q, target_Q)
                    
                gradients = tape.gradient(critic_loss, self.particles_critic[i].trainable_variables) #calculate gradient
                self.optimizers_critic[i].apply_gradients(zip(gradients, self.particles_critic[i].trainable_variables)) #backpropagate
                
                # Compute actor loss
                with tf.GradientTape() as tape:
                    policy_grad = tf.reduce_mean(self.particles_critic[i](state, self.particles_actor[i](state)))
                
                # Optimize the actor
                gradients = tape.gradient(policy_grad, self.particles_actor[i].trainable_variables) #calculate gradient
                
                # Vectorize parameters and PGs
                vec_param, vec_policy_grad = params2vec(self.particles_actor[i].trainable_variables, gradients)
    
                policy_grads.append(tf.expand_dims(vec_policy_grad,0))
                parameters.append(tf.expand_dims(vec_param,0))
    
            # calculating the kernel matrix and its gradients
            parameters = tf.concat(parameters,0)
            k, grad_k = self.compute_kernel(parameters)
    
            policy_grads = 1. / self.temperature * tf.concat(policy_grads,0)
            grad_logp = tf.linalg.matmul(k, policy_grads)
    
            grad_theta = - (grad_logp + grad_k) / self.n_particles
    
            # update param gradients
            for i in range(self.n_particles):
                grads=vec2params(grad_theta[i],self.particles_actor[i].trainable_variables)
                self.optimizers_actor[i].apply_gradients(zip(grads, self.particles_actor[i].trainable_variables))
    
                # Update the frozen target models
                for param, target_param in zip(self.particles_critic[i].trainable_variables, self.particles_critic_target[i].trainable_variables):
                    target_param.assign(self.tau * param + (1 - self.tau) * target_param)
    
                for param, target_param in zip(self.particles_actor[i].trainable_variables, self.particles_actor_target[i].trainable_variables):
                    target_param.assign(self.tau * param + (1 - self.tau) * target_param)

