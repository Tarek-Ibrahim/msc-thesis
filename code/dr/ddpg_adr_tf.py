# %% TODOs


#%% Imports
#General
import numpy as np
import os
# os.environ["OMP_NUM_THREADS"] = "1"
import pandas as pd

#visualization
import matplotlib.pyplot as plt
import tqdm

#multiprocessing
import multiprocessing as mp

#env
from baselines.common.vec_env import VecEnv, CloudpickleWrapper
import gym
#------only for spyder IDE
for env in gym.envs.registration.registry.env_specs.copy():
     if 'custom' in env:
         print('Remove {} from registry'.format(env))
         del gym.envs.registration.registry.env_specs[env]
#------
import gym_custom

#utils
from scipy.spatial.distance import squareform, pdist
import decimal

#ML
import tensorflow as tf
import tensorflow_probability as tfp
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

#%% Utils

progress=lambda x: tqdm.trange(x, leave=True) #for visualizing/monitoring training progress

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


def rollout(n_particles, env, policy_agent, RB, eps_rollout_agent, T_env, T_agent_init, b_agent, gamma_agent, freeze_agent=True, add_noise=False, noise_scale=0.1): 
    
    states = [[] for _ in range(n_particles)]
    actions = [[] for _ in range(n_particles)]
    next_states = [[] for _ in range(n_particles)]
    rewards = [[] for _ in range(n_particles)]
    ep_rewards = []

    for ep in range(eps_rollout_agent):
        rewards_sum = np.zeros(n_particles)
        state = env.reset()

        done = [False] * n_particles
        add_to_buffer = [True] * n_particles
        t_env = 0 #env timestep
        training_iters = 0

        while not all(done) and t_env <= T_env:
            action = policy_agent.select_action(np.array(state))

            if add_noise:
                action = action + np.random.normal(0, noise_scale, size=action.shape)
                action = action.clip(-1, 1)

            next_state, reward, done, info = env.step(action)

            #Add samples to replay buffer
            for i, st in enumerate(state):
                if add_to_buffer[i]:
                    states[i].append(st)
                    actions[i].append(action[i])
                    next_states[i].append(next_state[i])
                    rewards[i].append(reward[i])
                    rewards_sum[i] += reward[i]
                    training_iters += 1

                    if RB is not None:
                        done_bool = 0 if t_env + 1 == T_env else float(done[i])
                        RB.add((state[i], next_state[i], action[i], reward[i], done_bool))

                if done[i]:
                    # Avoid duplicates
                    add_to_buffer[i] = False

            state = next_state
            t_env += 1

        # Train agent policy
        if not freeze_agent and len(RB.storage) > T_agent_init: #if it has enough samples
            # policy_agent.train(RB=RB, eps=training_iters,batch_size=b_agent,gamma=gamma_agent)
            policy_agent.train(RB=RB, eps=int(T_env/10),batch_size=b_agent,gamma=gamma_agent)

        ep_rewards.append(rewards_sum)

    #concatenate rollouts
    trajs = []
    for i in range(n_particles):
        trajs.append(np.concatenate(
            [
                np.array(states[i]),
                np.array(actions[i]),
                np.array(next_states[i])
            ], axis=-1))

    return trajs, ep_rewards

#%% Environments

def envworker(child_conn, parent_conn, env_func):
    parent_conn.close()
    env = env_func.x()
    while True:
        func, arg = child_conn.recv()
        
        if func == 'step':
            ob, reward, done, info = env.step(arg)
            child_conn.send((ob, reward, done, info))
        elif func == 'reset':
            ob = env.reset()
            child_conn.send(ob)
        elif func == 'close':
            child_conn.close()
            break
        elif func == 'randomize':
            randomized_val = arg
            env.randomize(randomized_val)
            child_conn.send(None)


class SubprocVecEnv(VecEnv):
    def __init__(self,env_funcs,ds,da):
        
        self.parent_conns, self.child_conns = zip(*[mp.Pipe() for _ in env_funcs])
        self.workers = [mp.Process(target=envworker,args=(child_conn, parent_conn, CloudpickleWrapper(env_func))) for (child_conn, parent_conn, env_func) in zip(self.child_conns, self.parent_conns, env_funcs)]
        
        for worker in self.workers:
            worker.daemon = True #making child processes daemonic to not continue running when master process exists
            worker.start()
        for child_conn in self.child_conns:
            child_conn.close()
        
        self.waiting = False
        self.closed = False
        
        VecEnv.__init__(self, len(env_funcs), ds, da)
        
    def step_async(self, actions):
        #step through each env asynchronously
        for parent_conn, action in zip(self.parent_conns, actions):
            parent_conn.send(('step',action))
        self.waiting = True
        
    def step_wait(self):
        #wait for all envs to finish stepping and then collect results
        results = [parent_conn.recv() for parent_conn in self.parent_conns]
        self.waiting = False
        states, rewards, dones, infos = zip(*results)
        
        return np.stack(states), np.stack(rewards), np.stack(dones), infos
    
    def randomize(self, randomized_values):
        for parent_conn, val in zip(self.parent_conns, randomized_values):
            parent_conn.send(('randomize', val))
        results = [parent_conn.recv() for parent_conn in self.parent_conns]
        self.waiting = False
    
    def reset(self):
        for parent_conn in self.parent_conns:
            parent_conn.send(('reset',None))
        results = [parent_conn.recv() for parent_conn in self.parent_conns]
        return np.stack(results)
    
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


def make_env(env_name,seed=None, rank=None):
    def _make_env():
        env = gym.make(env_name)
        if seed is not None and rank is not None:
            env.seed(seed+rank)
        return env
    return _make_env


def make_vec_envs(env_name, seed, n_workers):
    envs=[make_env(env_name,seed,rank) for rank in range(n_workers)]
    envs=SubprocVecEnv(envs, ds, da)
    return envs

#%% Agent's Policy (any model-free RL algorithm. here: DDPG)

class Actor(tf.keras.Model):
    def __init__(self, in_size, h1, h2, out_size, max_action):
        super(Actor, self).__init__()
        
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
    def __init__(self, ds, da, h1, h2, lr_agent, a_max=1.):
        self.actor = Actor(in_size=ds, h1=h1, h2=h2, out_size=da, max_action=a_max)
        self.actor_target = Actor(in_size=ds, h1=h1, h2=h2, out_size=da, max_action=a_max)
        self.actor_target.set_weights(self.actor.get_weights())
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_agent)

        self.critic = Critic(in_size=ds+da, h1=h1, h2=h2) 
        self.critic_target = Critic(in_size=ds+da, h1=h1, h2=h2) 
        self.critic_target.set_weights(self.critic.get_weights())
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_agent)
    
    def select_action(self, state):
        state = tf.convert_to_tensor(state,dtype=tf.float32) 
        return self.actor(state).numpy()

    def train(self, RB, eps, batch_size, gamma, tau=0.005):
        for ep in range(eps):
            # Sample replay buffer 
            x, y, u, r, d = RB.sample(batch_size)
            state = tf.convert_to_tensor(x,dtype=tf.float32) 
            action = tf.convert_to_tensor(u,dtype=tf.float32) 
            next_state = tf.convert_to_tensor(y,dtype=tf.float32) 
            done = tf.convert_to_tensor(1-d,dtype=tf.float32) 
            reward = tf.convert_to_tensor(r,dtype=tf.float32)  

            # Compute the target Q value
            target_Q = self.critic_target(next_state, self.actor_target(next_state))
            target_Q = reward + (done * gamma * target_Q)

            
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
                target_param.assign(tau * param + (1 - tau) * target_param)

            for param, target_param in zip(self.actor.trainable_variables, self.actor_target.trainable_variables):
                target_param.assign(tau * param + (1 - tau) * target_param)





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


class Discriminator(object): #TIP: reward of discriminator (score) is in the interval [0,1] -> for the same env, starts high then reduces over time as agent gets better at that env, then the randomized parameters of the env are encouraged to vary to make the disc reward high again
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
    
#====================
# Adjust Torch Utils
#====================
    
def parameters_to_vector(parameters, grads):
    
        vec_params, vec_grads = [], []
        for param,grad in zip(parameters,grads):
            vec_params.append(tf.reshape(param,-1))
            vec_grads.append(tf.reshape(grad,-1))
        return tf.concat(vec_params,0), tf.concat(vec_grads,0)


def vector_to_parameters(vec, grads):
    grads_new=[]
    pointer = 0
    for grad in grads:
        numel = tf.size(grad).numpy()
        grads_new.append(tf.reshape(vec[pointer:pointer + numel],list(grad.shape)))

        pointer += numel
    return grads_new

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
    
                    # with tf.GradientTape(persistent=True) as tape:
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
            vec_param, vec_policy_grad = parameters_to_vector(self.particles[i].trainable_variables, gradients)

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
            grads_tot[i]=vector_to_parameters(grad_theta[i],grads_tot[i])
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
    
#%% Algorithm Implementation
if __name__ == '__main__':
    
    #%% Inputs
    
    #DDPG
    lr_agent=0.001 #0.01 #0.001 #learning rate
    h1_agent=100 #400 #64 #400
    h2_agent=100 #300 #64 #300
    gamma_agent=0.99 #discount factor
    T_agent_init=1000 #number of timesteps before any updates
    b_agent=1000 #100 #1000 #batch size
    eps_rollout_agent=1 #number of episodes to rollout the agent for per simulation instance
    
    #Discriminator
    r_disc_scale = 1. #reward scale
    h_disc=128 #32 #128
    lr_disc=0.002 #0.02 #0.002
    b_disc=128
    
    #SVPG
    n_particles=10
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
    
    #Env
    env_names=['cartpole_custom-v1', 'halfcheetah_custom-v1', 'halfcheetah_custom_norm-v1', 'halfcheetah_custom_rand-v1', 'halfcheetah_custom_rand-v2', 'lunarlander_custom_default_rand-v0']
    env_name=env_names[-2]
    
    #Evaluation
    evaluate=True
    log_ival=1
    eval_eps=3
    
    #general
    tr_eps=500
    file_name=os.path.basename(__file__).split(".")[0]
    common_name = "_"+file_name+"_"+env_name
    verbose=1 #or: False/True (False/0: display progress bar; True/1: display 1 log newline per episode) 
    
    #Seed
    # seeds=[None,1,2,3,4,5]
    seeds=[1,2,3]
    # seed = seeds[1]
    
    plot_disc_rewards_all=[]
    plot_rewards_all=[]
    plot_eval_rewards_all=[]
    
    for seed in seeds:
        
        print(f"For Seed: {seed} \n")
        
        #%% Initializations
        #Env
        env=gym.make(env_name)
        set_seed(seed,env)
        T_env=env._max_episode_steps #task horizon / max env timesteps
        ds=env.observation_space.shape[0] #state dims
        da=env.action_space.shape[0] #action dims
        a_max=env.action_space.high[0]
        dr=env.unwrapped.randomization_space.shape[0] #N_rand (no. of randomization params)
        n_workers=n_particles
        
        env_ref=make_vec_envs(env_name, seed, n_workers)
        env_rand=make_vec_envs(env_name, seed, n_workers)
        
        #models
        policy_agent=DDPG(ds, da, h1_agent, h2_agent, lr_agent, a_max)
        discriminator=Discriminator(ds, da, h_disc, b_disc, r_disc_scale, lr_disc)
        svpg = SVPG(n_particles, dr, h_svpg, delta_max, T_svpg, T_svpg_reset, temp, kld_coeff, lr_svpg, gamma_svpg, type_particles)
        
        #memory
        RB=ReplayBuffer()
        
        # t_svpg=0 #SVPG timesteps
        # T_agent=int(1e6) #max agent timesteps
        
        #Results
        plot_rewards=[]
        plot_eval_rewards=[]
        plot_disc_rewards=[]
        total_timesteps=[]
        sampled_regions = [[] for _ in range(dr)]
        best_reward=-1e6
        t_agent=0
        
        #Evaluation
        eval_rewards_mean=0
        eval_freq = T_env * n_particles
        t_eval=0 # agent timesteps since eval 

        episodes=progress(tr_eps) if not verbose else range(tr_eps)
        # with tqdm.tqdm(total=T_agent) as pbar:
        for episode in episodes:
        # while t_agent < T_agent:
            #get sim instances from SVPG policy if current timestep is greater than the specified initial, o.w. create completely randomized env
            simulation_instances = svpg.step() if episode >= T_svpg_init else -1 * np.ones((n_particles,T_svpg,dr))
            
            # Create placeholders
            rewards_disc = np.zeros(simulation_instances.shape[:2])
            scores_disc=np.zeros(simulation_instances.shape[:2])
        
            # Reshape to work with vectorized environments
            simulation_instances = np.transpose(simulation_instances, (1, 0, 2))
            
            for t in range(T_svpg):
                T_disc_eps = 0 #agent timesteps in the current iteration/episode
                # create ref and randomized instances of the env, rollout the agent in both, and train the agent
                ref_traj, _=rollout(n_particles,env_ref,policy_agent,None,eps_rollout_agent,T_env,T_agent_init,b_agent,gamma_agent)
                env_rand.randomize(simulation_instances[t])
                rand_traj, rewards_agent =rollout(n_particles,env_rand,policy_agent,RB,eps_rollout_agent,T_env,T_agent_init,b_agent,gamma_agent,freeze_agent=False,add_noise=True)
                
                for i in range(n_particles):
                    T_disc_eps += len(rand_traj[i])
                    t_agent += len(rand_traj[i])
                    t_eval += len(rand_traj[i])
                    
                    r_disc, score_disc = discriminator.calculate_rewards(rand_traj[i])
                    rewards_disc[i][t]= r_disc
                    scores_disc[i][t]=score_disc
                    
                #train discriminator (with set of all ref and rand trajs for all agents at the current svpg timestep)
                flattened_rand = [rand_traj[i] for i in range(n_particles)]
                flattened_rand = np.concatenate(flattened_rand)
        
                flattened_ref= [ref_traj[i] for i in range(n_particles)]
                flattened_ref = np.concatenate(flattened_ref)
                
                discriminator.train(ref_traj=flattened_ref, rand_traj=flattened_rand, eps=T_disc_eps)
                        
            plot_disc_rewards.append(scores_disc.mean())
            
            #update svpg particles (ie. train their policies)
            if episode >= T_svpg_init:
                svpg.train(rewards_disc)
                
                #log sampled regions only once svpg particles start training (i.e. once adr starts)
                for dim in range(dr):
                    low=env.unwrapped.dimensions[dim].range_min
                    high=env.unwrapped.dimensions[dim].range_max
                    scaled_instances=low + (high-low) * simulation_instances[:, :, dim]
                    sampled_regions[dim]=np.concatenate([sampled_regions[dim],scaled_instances.flatten()])
            
            #evaluate
            if evaluate and t_eval>eval_freq:
                t_eval %= eval_freq
                eval_rewards = []
                for _ in range(eval_eps):
                    env.randomize(["random"]*dr)
                    s=env.reset()
                    done=False
                    R=0
                    while not done:
                        a = policy_agent.select_action(np.expand_dims(s,0))
                        s, r, done, _ = env.step(a)
                        R+=r
                    eval_rewards.append(R)
                
                eval_rewards_mean=np.mean(np.array(eval_rewards).flatten())
                plot_eval_rewards.append(eval_rewards_mean)
            
            #compute & log results
            #save best running model [params]
            if eval_rewards_mean>best_reward: 
                best_reward=eval_rewards_mean
                policy_agent.actor.save_weights(f"saved_models/actor{common_name}")
                policy_agent.critic.save_weights(f"saved_models/critic{common_name}")
            #save plot data
            plot_rewards.append(np.array(rewards_agent).mean())
            total_timesteps.append(t_agent)
            #log iteration results & statistics
            # if t_agent % 1== 0:
            if t_agent % log_ival == 0:
                log_msg="Rewards Agent: {:.2f}, Rewards Disc: {:.2f}, Rewards Eval: {:.2f}, Total Timesteps: {}".format(np.array(rewards_agent).mean(), scores_disc.mean(), eval_rewards_mean, t_agent)
                if verbose:
                    print(log_msg+f" episode:{episode} \n")
                else:
                    episodes.set_description(desc=log_msg); episodes.refresh()
                    # pbar.update(); pbar.set_description(desc=log_msg); pbar.refresh()
    
            # t_svpg += 1
            
        plot_rewards_all.append(plot_rewards)
        plot_eval_rewards_all.append(plot_eval_rewards)
        plot_disc_rewards_all.append(plot_disc_rewards)
        
        env.close()
        env_ref.close()
        env_rand.close()
    
    
    #%% Results & Plots
    #process results
    plot_rewards_mean = np.stack(plot_rewards_all).mean(0)
    plot_eval_rewards_mean = np.stack(plot_eval_rewards_all).mean(0)
    plot_disc_rewards_mean = np.stack(plot_disc_rewards_all).mean(0)
    
    plot_rewards_max= np.maximum.reduce(plot_rewards_all)
    plot_eval_rewards_max = np.maximum.reduce(plot_eval_rewards_all)
    plot_disc_rewards_max = np.maximum.reduce(plot_disc_rewards_all)
    
    plot_rewards_min = np.minimum.reduce(plot_rewards_all)
    plot_eval_rewards_min = np.minimum.reduce(plot_eval_rewards_all)
    plot_disc_rewards_min = np.minimum.reduce(plot_disc_rewards_all)
    
    #save results to df
    df = pd.DataFrame(list(zip(plot_rewards_mean,
                               plot_rewards_max,
                               plot_rewards_min,
                               plot_eval_rewards_mean,
                               plot_eval_rewards_max,
                               plot_eval_rewards_min,
                               plot_disc_rewards_mean,
                               plot_disc_rewards_max,
                               plot_disc_rewards_min,
                               total_timesteps)),
                      columns =['Rewards_Tr', 'Rewards_Tr_Max', 'Rewards_Tr_Min', 'Rewards_Eval', 'Rewards_Eval_Max', 'Rewards_Eval_Min', 'Rewards_Disc', 'Rewards_Disc_Max', 'Rewards_Disc_Min', 'Total_Timesteps'])
    df.to_pickle(f"plots/results{common_name}.pkl")
    
    #plot results
    title="Training Rewards"
    plt.figure(figsize=(16,8))
    plt.grid(1)
    plt.plot(plot_rewards_mean)
    plt.fill_between(range(tr_eps), plot_rewards_max, plot_rewards_min,alpha=0.2)
    # plt.axhline(y = env.spec.reward_threshold, color = 'r', linestyle = '--',label='Solved')
    plt.title(title)
    plt.legend(loc="upper right")
    plt.savefig(f'plots/tr{common_name}.png')
    
    title="Evaluation Rewards"
    plt.figure(figsize=(16,8))
    plt.grid(1)
    plt.plot(plot_eval_rewards_mean)
    plt.fill_between(range(tr_eps), plot_eval_rewards_max, plot_eval_rewards_min,alpha=0.2)
    # plt.axhline(y = env.spec.reward_threshold, color = 'r', linestyle = '--',label='Solved')
    plt.title(title)
    plt.legend(loc="upper right")
    plt.savefig(f'plots/ts{common_name}.png')
    
    title="Discriminator Rewards"
    plt.figure(figsize=(16,8))
    plt.grid(1)
    plt.plot(plot_disc_rewards_mean)
    plt.fill_between(range(tr_eps), plot_disc_rewards_max, plot_disc_rewards_min,alpha=0.2)
    plt.title(title)
    plt.savefig(f'plots/disc{common_name}.png')
    
    #TODO: plot control actions (of which episode(s)?)
    
    eps_step=int((tr_eps-T_svpg_init)/4) #100
    region_step=eps_step*T_svpg*n_particles
    df2=pd.DataFrame()
    for dim, regions in enumerate(sampled_regions):
        
        low=env.unwrapped.dimensions[dim].range_min
        high=env.unwrapped.dimensions[dim].range_max
        
        dim_name=env.unwrapped.dimensions[dim].name
        
        d = decimal.Decimal(str(low))
        step_exp=d.as_tuple().exponent-1
        step=10**step_exp

        x=np.arange(low,high+step,step)
        
        title=f"Sampled Regions for Randomization Dim = {dim_name} Over Time"
        plt.figure(figsize=(16,8))
        plt.grid(1)
        plt.hist((regions[region_step*0:region_step*1],regions[region_step*1:region_step*2],regions[region_step*2:region_step*3], regions[region_step*3:]), np.arange(min(x),max(x)+2*step,step), histtype='barstacked', label=[f'{eps_step*1} eps',f'{eps_step*2} eps', f'{eps_step*3} eps', f'{eps_step*4} eps'],color=["lightskyblue","blueviolet","hotpink","lightsalmon"])
        plt.xlim(min(x), max(x)+step)
        plt.legend()
        plt.title(title)
        #save results
        plt.savefig(f'plots/sampled_regions_dim_{dim_name}{common_name}.png')
        df2[f'Sampled_Regions_{dim_name}'] = list(regions)
    
    df2.to_pickle(f"plots/sampled_regions{common_name}.pkl")