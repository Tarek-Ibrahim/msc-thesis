# %% TODOs

#TODO: handle normalization/no-normalization situations of inputs (to model)
#TODO: adjust core alg
#TODO: use replay buffer instead of internal storage in model train
#TODO: make sure alg runs/learns for non-terminating envs
#TODO: combine with adr
#TODO: make sure alg runs/learns for non-terminating envs
#TODO: adjust for terminating envs
#TODO: make sure alg runs/learns for terminating envs
#TODO: integrate into all codes (trpo and ppo -based)
#TODO: make sure alg runs/learns 

# %% Imports
#general
import numpy as np
import pandas as pd
import os
# os.environ["OMP_NUM_THREADS"] = "1"
import yaml
from common import set_seed, progress , parameters_to_vector, ValueNetwork, surrogate_loss, HVP, conjugate_gradients, line_search, make_vec_envs,compute_advantages,weighted_mean, adapt #, ReplayBuffer #,PolicyNetwork
import timeit
from collections import OrderedDict

#env
import gym
import gym_custom
from baselines.common.vec_env import CloudpickleWrapper

#visualization
import matplotlib.pyplot as plt

#ML
import tensorflow as tf
import tensorflow_probability as tfp
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import queue as Q

#multiprocessing
import multiprocessing as mp
# import multiprocess as mp

#%% Utils

def collect_rollout_batch(envs, ds, da, policy, T, b, n_workers, queue, params=None,add_noise=False,noise_scale=0.1, initial=False,models=None,RB=None,vec_model=None,state_dict=None): # a batch of rollouts

    # if params_set is None:
    #     params_set = [None] * b
    
    state_dicts=[state_dict]*b

    states=[[] for _ in range(b)]
    rewards = [[] for _ in range(b)]
    actions = [[] for _ in range(b)]
    next_states = [[] for _ in range(b)]
    
    for rollout_idx in range(b):
        queue.put(rollout_idx)
    for _ in range(n_workers):
        queue.put(None)
    
    #each rollout in the batch is the history of stepping through the environment once till termination
    
    if models is not None:
        
        # s=np.stack([envs.reset() for _ in range(b)])
        # rollout_idxs = vec_model.reset()
        s,rollout_idxs = vec_model.reset()
        # state_dicts=[models[idx].get_weights() for idx in rollout_idxs]
    else:
        s, rollout_idxs=envs.reset() 
    dones=[False]
    
    while (not all(dones)) or (not queue.empty()):
        s=s.astype(np.float32)
        state=tf.convert_to_tensor(s)
        if initial:
            a=np.random.uniform(0,1,(b,da)).astype(np.float32)
        else:
            a=policy(state,params).sample().numpy()
        # if add_noise:
        #     a = a + np.random.normal(0, noise_scale, size=a.shape)
        #     # a = a.clip(-1, 1)
        
        if models is not None:
            # inputs=tf.concat([s, a], axis=-1)
            s_dash, r, dones, rollout_idxs_new, _ = vec_model.step(state_dicts,s,a) #(inputs) #(models,inputs)
        else:
            s_dash, r, dones, rollout_idxs_new, _ = envs.step(a)
        
        #append to batch
        for state, next_state, action, reward, rollout_idx, done in zip(s,s_dash,a,r,rollout_idxs,dones):
            if rollout_idx is not None:
                states[rollout_idx].append(state.astype(np.float32))
                next_states[rollout_idx].append(next_state.astype(np.float32))
                actions[rollout_idx].append(action.astype(np.float32))
                rewards[rollout_idx].append(reward.astype(np.float32))
                if RB is not None:
                    done_bool = 0 if len(states[rollout_idx]) + 1 == T else float(done)
                    RB.add((state, next_state, action, reward, done_bool))

        #reset
        s, rollout_idxs = s_dash, rollout_idxs_new
    
    T_max=max(map(len,rewards))
    states_mat=np.zeros((T_max,b,ds),dtype=np.float32)
    actions_mat=np.zeros((T_max,b,da),dtype=np.float32)
    rewards_mat=np.zeros((T_max,b),dtype=np.float32)
    masks_mat=np.zeros((T_max,b),dtype=np.float32)
    # next_states_mat=np.zeros((T_max,b,ds),dtype=np.float32)
    
    for rollout_idx in range(b):
        T_rollout=len(rewards[rollout_idx])
        states_mat[:T_rollout,rollout_idx]= np.stack(states[rollout_idx])
        # next_states_mat[:T_rollout,rollout_idx]= np.stack(next_states[rollout_idx])
        actions_mat[:T_rollout,rollout_idx]= np.stack(actions[rollout_idx])
        rewards_mat[:T_rollout,rollout_idx]= np.stack(rewards[rollout_idx])
        masks_mat[:T_rollout,rollout_idx]=1.0
    
    # D=[states_mat, actions_mat, rewards_mat, masks_mat, next_states_mat]
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


class ReplayBuffer(object):
    def __init__(self, in_size, max_size=1e6):
        self.storage = []
        self.max_size = int(max_size)
        self.next_idx = 0
        self.mu=np.zeros((1,in_size))
        self.sigma=np.ones((1,in_size))

    # Expects tuples of (state, next_state, action, reward, done)
    def add(self, data):
        if self.next_idx >= len(self.storage):
            self.storage.append(data)
        else:
            self.storage[self.next_idx] = data

        self.next_idx = (self.next_idx + 1) % self.max_size
    
    
    def fit_input_stats(self):
        
        state, _, action, _, _ = zip(*self.storage)
        
        inputs= np.concatenate([state,action],axis=-1)
        
        self.mu = np.mean(inputs, axis=0, keepdims=True) #over cols (each observation/action col of input) and keeping same col size --> result has size = (1,input_size)
        sigma = np.std(inputs, axis=0, keepdims=True)
        sigma[sigma < 1e-12] = 1.0 #???: why 1 (and not 1e-12 e.g.)??
        
        self.sigma=sigma

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
        
        self.nonlinearity=tf.nn.tanh
            
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

#%% Model

class DynamicsModel(tf.keras.Model):
    
    def __init__(self, ds, da, n, h, epochs, lr, b_train):
        super().__init__()
        
        self.epochs=epochs
        self.optimizer=tf.keras.optimizers.Adam(learning_rate=lr)
        # self.env=env
        self.ds=ds #env.observation_space.shape[0] #state dims
        self.da=da #env.action_space.shape[0] #action dims
        self.b_train=b_train
        
        # self.inputs=np.empty((0,in_size))
        # self.targets=np.empty((0,out_size // 2))
        
        self.n=n
        self.in_size=ds+da
        self.out_size=ds*2
        
        self.max_logvar = tf.Variable(0.5 * tf.ones([1, self.out_size // 2]))
        self.min_logvar = tf.Variable(-10.0 * tf.ones([1, self.out_size // 2]))
        
        # self.mu = tf.Variable(initial_value=tf.keras.initializers.Zeros()(shape=[1,self.in_size]), trainable=False)
        # self.sigma = tf.Variable(initial_value=tf.keras.initializers.Ones()(shape=[1,self.in_size]), trainable=False)
        
        self.w, self.b = [], []
        for l in range(n+1):
            ip=self.in_size if l==0 else h
            op=self.out_size if l==n else h
            w, b = self.initialize(ip,op)
            self.w.append(w)
            self.b.append(b)
    
    
    def initialize(self,in_size,out_size):
        #truncated normal for weights (i.e. draw from normal distribution with samples outside 2*std from mean discarded and resampled)
        #zeros for biases
        
        w=tf.Variable(initial_value=tf.keras.initializers.TruncatedNormal(mean=0.0,stddev=1.0/(2.0*np.sqrt(in_size)))(shape=[in_size,out_size]))
        b=tf.Variable(initial_value=tf.keras.initializers.Zeros()(shape=[1, out_size]))
        
        return w, b
    
    
    def call(self,inputs,normalize=True):
        
        # print(f"inputs shape before: {inputs.shape} \n ")
       
        #fwd pass
        for l in range(self.n+1):
            inputs = tf.matmul(inputs,self.w[l]) + self.b[l] #after size (till before last layer) = [B,input samples,h]; after NN size=[B,input samples,out_size]
            if l<self.n: #skips for last iteration/layer
                # inputs = tf.keras.activations.swish(inputs)
                inputs = tf.nn.relu(inputs)
        
        # print(f"inputs shape after: {inputs.shape} \n ")
        
        #extract mean and log(var) from network output
        mean = inputs[:, :self.out_size // 2]
        logvar = inputs[:, self.out_size // 2:]
        
        #bounding variance (becase network gives arbitrary variance for OOD points --> could lead to numerical problems)
        logvar = self.max_logvar - tf.nn.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + tf.nn.softplus(logvar - self.min_logvar)
        
        return mean, logvar
    
    
#     def normalize_action(self, action):
#         # Clip the action in [-1, 1]
#         action = np.clip(action, -1.0, 1.0)
# 		# Map the normalized action to original action space
#         lb, ub = self.env.action_space.low, self.env.action_space.high
#         action = lb + 0.5 * (action + 1.0) * (ub - lb)
#         return action
    
    
    def train(self,RB): #Train the policy with rollouts
    
        states, next_states, actions, _, _ = RB.sample(self.b_train)
        # dones = 1 - ds
        
        inputs=tf.convert_to_tensor(np.concatenate([states, actions], axis=-1).astype(np.float32))
        inputs = (inputs - RB.mu) / RB.sigma
        
        targets = tf.convert_to_tensor(next_states - states,dtype=tf.float32)
        
        for _ in range(self.epochs): #for each epoch
            # Operate on batches:
            with tf.GradientTape() as tape:
                mean, logvar = self(inputs) #fwd pass
                var = tf.math.exp(-logvar)
                # Calculate loss
                loss = tf.math.reduce_mean(tf.math.reduce_mean(tf.math.square(mean - targets) * var, axis=-1), axis=-1) #MSE losses #???: why does mean over target dimension make sense?
                loss += tf.math.reduce_mean(tf.math.reduce_mean(logvar, axis=-1), axis=-1) #var losses #???: why does mean over target dimension make sense?
                loss = tf.math.reduce_sum(loss) 
                loss += 0.01 * (tf.math.reduce_sum(self.max_logvar) - tf.math.reduce_sum(self.min_logvar)) # a constant
                # loss += self.model.compute_decays() #L2 regularization
            
            gradients = tape.gradient(loss, self.trainable_variables) #calculate gradient
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables)) #backpropagate
   

class DynamicsModelWrapper(object):
    
    def __init__(self, ds, da, T_env, reward_func, H_max,n,h,epochs,lr,b_train,env_name):
        super().__init__()
        
        self.T_env=T_env #task horizon
        self.ds=ds #state dims
        self.da=da #action dims
        self.reward_func=reward_func
        self.timesteps=0
        # self.model=DynamicsModel(ds,da,n,h,epochs,lr,b_train)
        # self.env=env
        # self.rf=rf
        self.env_name=env_name
        self.n=n
        self.h=h
        self.epochs=epochs
        self.lr=lr
        self.b_train=b_train
        
        self.H=min(self.T_env,H_max)
        
    
    def reset(self):
        self.timesteps = 0
        env=gym.make(self.env_name)
        return env.reset()
    
    
    def step(self,state_dict,s,a):
        
        if s.ndim <2:
            s=np.expand_dims(s, 0)
            a=np.expand_dims(a, 0)
        
        inputs=tf.convert_to_tensor(np.concatenate([s, a], axis=-1).astype(np.float32))
        
        model=DynamicsModel(self.ds,self.da,self.n,self.h,self.epochs,self.lr,self.b_train)
        # self.actor_target.load_state_dict(self.actor.state_dict())
        model.set_weights(state_dict)
        
        info = None
        done=False
        
        # mean, logvar = model(inputs)
        mean, logvar = model(inputs,normalize=False)
        var = tf.math.exp(logvar)
        delta_s_dash = mean + tf.random.normal(shape=tf.shape(mean)) * tf.sqrt(var)
        # print(f"delta_s_dash_shape: {delta_s_dash.shape} \n")
        # delta_s_dash=tf.reshape(delta_s_dash,[-1, self.ds])
        s_dash= (s+delta_s_dash).numpy().squeeze()
        # print(f"s_dash_shape: {s_dash.shape} \n")
        r=self.reward_func(s,a,s_dash).squeeze()
        
        self.timesteps +=1
        if self.timesteps >= self.H:
            done = True
        
        return s_dash, r, done, info


class SubprocVecModel(object):
    def __init__(self,model_funcs,ds,da,queue_model,lock_model):
        
        self.parent_conns, self.child_conns = zip(*[mp.Pipe() for _ in model_funcs])
        self.workers = [ModelWorker(child_conn, model_func, queue_model, lock_model) for (child_conn, model_func) in zip(self.child_conns, model_funcs)]
        
        for worker in self.workers:
            worker.daemon = True #making child processes daemonic to not continue running when master process exists
            worker.start()
        for child_conn in self.child_conns:
            child_conn.close()
        
        self.waiting = False
        self.closed = False
        
    def step(self, state_dicts, states, actions):
        
        #step through each env asynchronously
        for parent_conn, state_dict, state, action in zip(self.parent_conns, state_dicts, states, actions):
            parent_conn.send(('step',(state_dict, state, action)))
        self.waiting = True
        
        #wait for all envs to finish stepping and then collect results
        results = [parent_conn.recv() for parent_conn in self.parent_conns]
        self.waiting = False
        states, rewards, dones, rollouts_idxs, infos = zip(*results)
        
        return np.stack(states), np.stack(rewards), np.stack(dones), rollouts_idxs, infos
        
    # def step(self, models, inputs):
        
    #     #step through each env asynchronously
    #     for parent_conn, model, inp in zip(self.parent_conns, models, inputs):
    #         parent_conn.send(('step',model,inp))
    #     self.waiting = True
        
    #     #wait for all envs to finish stepping and then collect results
    #     results = [parent_conn.recv() for parent_conn in self.parent_conns]
    #     self.waiting = False
    #     states, rewards, dones, rollouts_idxs, infos = zip(*results)
        
    #     return np.stack(states), np.stack(rewards), np.stack(dones), rollouts_idxs, infos
    
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
        
        
class ModelWorker(mp.Process):
    def __init__(self, child_conn, model_func, queue_model, lock_model):
        super().__init__()
        
        self.child_conn = child_conn
        self.model = model_func()
        self.queue_model = queue_model
        self.lock = lock_model
        self.rollout_idx = None
        self.done = False
        self.ds = self.model.ds
        # self.dyn_model=DynamicsModel(self.ds,self.model.da,self.model.n,self.model.h,self.model.epochs,self.model.lr,self.model.b_train)
    
    def try_reset(self):
        with self.lock:
            try:
                self.rollout_idx = self.queue_model.get()
                self.done = (self.rollout_idx is None)
            except Q.Empty:
                self.done = True
            # self.model.reset()
            if self.done:
                state = np.zeros(self.ds, dtype=np.float32)
            else:
                state = self.model.reset()
            return state
    
    def run(self):
        while True:
            func, args = self.child_conn.recv()
            
            if func == 'step':
                if self.done:
                    state, reward, done, info = np.zeros(self.ds, dtype=np.float32), 0.0, True, {}
                else:
                    # print(f"idx: {self.rollout_idx}, state_shape: {args[1].shape}, action_shape: {args[2].shape} \n")
                    state, reward, done, info = self.model.step(args[0],args[1],args[2])
                if done and not self.done:
                    state = self.try_reset()
                self.child_conn.send((state,reward,done,self.rollout_idx,info))
                
            elif func == 'reset':
                # self.try_reset()
                # self.child_conn.send(self.rollout_idx)
                state = self.try_reset()
                self.child_conn.send((state,self.rollout_idx))
                
            elif func == 'close':
                self.child_conn.close()
                break

# def make_dyn_model(ds,da,n,h_model,epochs,lr,b_model):
#     def _make_dyn_model():
#         dyn_model = DynamicsModel(ds,da, n, h_model, epochs, lr, b_model)
#         return dyn_model
#     return _make_dyn_model


def make_model(ds, da, T_env, reward_func, H_max, n,h,epochs,lr,b_train,env_name,seed=None, rank=None):
    def _make_model():
        vec_model = DynamicsModelWrapper(ds, da, T_env, reward_func, H_max, n,h,epochs,lr,b_train,env_name)
        return vec_model
    return _make_model


def make_vec_models(T_env, reward_func, H_max, seed, n_workers, ds, da, n,h,epochs,lr,b_train,env_name, queue_model, lock_model):
    models=[make_model(ds, da, T_env, reward_func, H_max,n,h,epochs,lr,b_train,env_name,seed,rank) for rank in range(n_workers)]
    models=SubprocVecModel(models,ds, da, queue_model, lock_model)
    return models

#%% Main Func
if __name__ == '__main__':
    
    #%% Inputs
    
    modes=["debug_mode","run_mode"]
    mode=modes[0]
    
    with open("config.yaml", 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    config=config[mode]
    
    #MAML
    h=config["h_mbmpo"]
    alpha=config["lr_maml"]
    b = config["b_maml"]
    gamma = config["gamma_maml"]
    
    #TRPO
    max_grad_kl=config["max_grad_kl"]
    max_backtracks=config["max_backtracks"]
    zeta=config["zeta"]
    rdotr_tol=config["rdotr_tol"]
    nsteps=config["nsteps"]
    damping=config["damping"]
    
    #MB-MPO
    n=config["n_layers"]
    B=config["n_models"]
    lr=config["lr_mbmpo"]
    epochs=config["epochs"]
    b_model=config["b_mbmpo"]
    h_model=config["h_models"]
    
    #Env
    env_name='halfcheetah_custom_rand-v2' #config["env_name"]
    n_workers=config["n_workers"] 
    
    #Evaluation
    evaluate=config["evaluate"]
    log_ival=config["log_ival"]
    eval_eps=config["eval_eps"]
    eval_step = 1.0 / eval_eps
    
    #general
    tr_eps=config["tr_eps"]
    file_name=os.path.basename(__file__).split(".")[0]
    common_name = "_"+file_name+"_"+env_name
    verbose=config["verbose"]
    T_rand_rollout=1 #config["T_rand_rollout"]
    load_policy=False
    H_max=200
    
    #Seed
    seeds=config["seeds"]
    
    plot_tr_rewards_all=[]
    plot_val_rewards_all=[]
    plot_eval_rewards_all=[]
    total_timesteps_all=[]
    best_reward=-1e6
    
    for seed in seeds:
        
        print(f"For Seed: {seed} \n")
        start_time=timeit.default_timer()
  
        #%% Initializations
        #multiprocessing
        queue = mp.Queue()
        lock = mp.Lock()
        
        queue_model = mp.Queue()
        lock_model = mp.Lock()
        
        # queue = mp.Manager().Queue()
        # lock = mp.Manager().Lock()
        
        # queue_model = mp.Manager().Queue()
        # lock_model = mp.Manager().Lock()
        
        #environment
        env=gym.make(env_name)
        set_seed(seed,env)
        T_env=env._max_episode_steps #task horizon
        ds=env.observation_space.shape[0] #state dims
        da=env.action_space.shape[0] #action dims
        dr=env.unwrapped.randomization_space.shape[0] #N_rand (no. of randomization params)
        reward_func = env.reward
        env_rand=make_vec_envs(env_name, seed, n_workers, ds, da, queue, lock)
        
        #models
        in_size=ds
        out_size=da
        policy = PolicyNetwork(in_size,h,out_size) #dynamics model
        if load_policy:
            policy.load_weights(f"saved_models/model{common_name}")
        value_net=ValueNetwork(in_size,gamma)
        
        # models = [DynamicsModel(da+ds, n, h_model, ds*2, epochs, lr, b_model, env) for _ in range(B)]
        models = [DynamicsModel(ds,da,n,h_model,epochs,lr,b_model) for _ in range(B)]
        # models = [make_dyn_model(ds,da,n,h_model,epochs,lr,b_model) for _ in range(B)]
        vec_model = make_vec_models(T_env, reward_func, H_max, seed, n_workers, ds, da,n,h_model,epochs,lr,b_model,env_name,queue_model, lock_model)
        
        RB=ReplayBuffer(da+ds)
        
        #results 
        plot_tr_rewards=[]
        total_timesteps=[]
        t_agent = 0
        
        #evaluation
        eval_rewards_mean=0
        eval_freq = n_workers * T_env
        t_eval=0 # agent timesteps since eval
        plot_eval_rewards=[]
        
        theta_dashes=[None] * B
 
        #%% Implementation
            
        episodes=progress(tr_eps) if not verbose else range(tr_eps)
        for episode in episodes:
            
            initial = True if episode == 0 else False
            
            #rollout svpg particles (each particle represents a different rand env and each timestep in that rollout represents different values for its randomization params)
            simulation_instances = -1 * np.ones((n_workers,T_rand_rollout,dr))
            
            #create empty storages
            rewards_tr_ep = []
            Ds, D_dashes=[], []
            
            # Reshape to work with vectorized environments
            simulation_instances = np.transpose(simulation_instances, (1, 0, 2))
            
            #randomize rand envs #!!!: this only works if transitions within the same rollout is collected in the same environment (i.e. by the same [env] worker) #that's why we choose: n_particles=b(rollout batch size)=n_workers/n_envs
            # env_rand.randomize(simulation_instances[t_rand_rollout])
            env_rand.randomize(simulation_instances[0])
            
            for theta_dash in theta_dashes:
                
                rollout_batch,_=collect_rollout_batch(env_rand, ds, da, policy, T_env, b, n_workers, queue,params=theta_dash,initial=initial,RB=RB)
                
                _, _, rewards,_= rollout_batch
                rewards_tr_ep.append(rewards)
                t_agent += rewards.size
                t_eval += rewards.size
            
            # for i in range(B):
            #     models[i].train(RB)
            
            theta_dashes=[]
            for model in models:
                
                # model().train(RB)
                model.train(RB)
                state_dict=model.get_weights() 
                
                #collect pre-adaptation rollout batch in rand envs (one rollout for each svpg particle)
                # D, _= collect_rollout_batch(vec_model, ds, da, policy, T_env, b, n_workers, queue_model, model=model)
                D, _= collect_rollout_batch(env, ds, da, policy, T_env, b, n_workers, queue_model, models=models,vec_model=vec_model,state_dict=state_dict)
                Ds.append(D)
                
                #adapt agent [meta-]parameters (via VPG w/ baseline)
                theta_dash=adapt(D,value_net,policy,alpha)
                theta_dashes.append(theta_dash)
                
                #collect post-adaptation rollout batch in rand envs
                # D_dash,_= collect_rollout_batch(vec_model, ds, da, policy, T_env, b, n_workers, queue_model, params=theta_dash,model=model)
                D_dash,_= collect_rollout_batch(env, ds, da, policy, T_env, b, n_workers, queue_model, params=theta_dash,models=models, vec_model=vec_model,state_dict=state_dict)
                D_dashes.append(D_dash)
    
            #outer loop: update meta-params (via: TRPO) #!!!: since MAML uses TRPO it is on-policy, so care should be taken that order of associated transitions is preserved
            with tf.GradientTape() as tape:
                prev_loss, _, prev_pis = surrogate_loss(D_dashes,policy,value_net,gamma,alpha=alpha,Ds=Ds)
            grads = parameters_to_vector(tape.gradient(prev_loss, policy.trainable_variables),policy.trainable_variables)
            prev_loss=tf.identity(prev_loss)
            hvp=HVP(D_dashes,policy,value_net,damping,alpha=alpha,Ds=Ds)
            search_step_dir=conjugate_gradients(hvp, grads, rdotr_tol=rdotr_tol,nsteps=nsteps)
            max_length=np.sqrt(2.0 * max_grad_kl / np.dot(search_step_dir, hvp(search_step_dir)))
            full_step=search_step_dir*max_length        
            prev_params = parameters_to_vector(policy.trainable_variables)
            line_search(policy, prev_loss, prev_pis, value_net, gamma, b, D_dashes, full_step, prev_params, max_grad_kl, max_backtracks, zeta,alpha=alpha,Ds=Ds)
            
            #evaluation
            if evaluate and t_eval>eval_freq:
                t_eval %= eval_freq
                eval_rewards=[]
                lower=0.
    
                for _ in range(eval_eps):
                    rand_value = np.around(np.random.uniform(low=lower, high=min(lower+eval_step,1.0)),3)
                    lower += eval_step
                    
                    env.randomize([rand_value]*dr)
                    
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
                        masks=tf.expand_dims(tf.expand_dims(tf.convert_to_tensor(np.array(1.0),dtype=np.float32),0),0)
                        D=[states, actions, rewards, masks]
                                            
                        theta_dash=adapt(D,value_net,policy,alpha)
                        
                        dist=policy(state,params=theta_dash)
                        a=tf.squeeze(dist.sample()).numpy()
                        s, r, done, _ = env.step(a)
                                                
                        R+=r
                        
                    eval_rewards.append(R)
                
                eval_rewards_mean=np.mean(np.array(eval_rewards).flatten())
            plot_eval_rewards.append(eval_rewards_mean)
        
            #compute & log results
            # compute rewards
            reward_ep = (tf.math.reduce_mean(tf.stack([tf.math.reduce_mean(tf.math.reduce_sum(rewards, axis=0)) for rewards in rewards_tr_ep], axis=0))).numpy() #sum over T, mean over b, stack horiz one reward per task, mean of tasks
            #save best running model [params]
            if eval_rewards_mean>best_reward: 
                best_reward=eval_rewards_mean
                policy.save_weights(f"saved_models/model{common_name}")
            #save plot data
            plot_tr_rewards.append(reward_ep)
            # plot_val_rewards.append(reward_val)
            total_timesteps.append(t_agent)
            #log iteration results & statistics
            if episode % log_ival == 0:
                log_msg="Rewards Tr: {:.2f}, Rewards Eval: {:.2f}, Total Timesteps: {}".format(reward_ep, eval_rewards_mean, t_agent)
                if verbose:
                    print(log_msg+f" episode:{episode} \n")
                else:
                    episodes.set_description(desc=log_msg); episodes.refresh()
                    
        
        plot_tr_rewards_all.append(plot_tr_rewards)
        plot_eval_rewards_all.append(plot_eval_rewards)
        total_timesteps_all.append(total_timesteps)
        
    
        #%% Results & Plot
        #process results
        plot_tr_rewards_mean = np.stack(plot_tr_rewards_all).mean(0)
        plot_eval_rewards_mean = np.stack(plot_eval_rewards_all).mean(0)
        total_timesteps_mean = np.stack(total_timesteps_all).mean(0)
        
        plot_tr_rewards_std= np.stack(plot_tr_rewards_all).std(0)
        plot_eval_rewards_std = np.stack(plot_eval_rewards_all).std(0)
        
        #save results to df
        df = pd.DataFrame(list(zip(plot_tr_rewards_mean,
                                   plot_tr_rewards_std,
                                   plot_eval_rewards_mean,
                                   plot_eval_rewards_std,
                                   total_timesteps_mean)),
                          columns =['Rewards_Tr_Mean', 'Rewards_Tr_Std', 'Rewards_Eval_Mean', 'Rewards_Eval_Std', 'Total_Timesteps'])
        
        df.to_pickle(f"plots/results{common_name}.pkl")
        
        #plot results
        title="Meta-Training Training Rewards"
        plt.figure(figsize=(16,8))
        plt.grid(1)
        plt.plot(plot_tr_rewards_mean)
        plt.fill_between(range(tr_eps), plot_tr_rewards_mean + plot_tr_rewards_std, plot_tr_rewards_mean - plot_tr_rewards_std,alpha=0.2)
        # plt.axhline(y = env.spec.reward_threshold, color = 'r', linestyle = '--',label='Solved')
        plt.title(title)
        plt.savefig(f'plots/mtr_tr{common_name}.png')
        
        title="Meta-Testing Rewards"
        plt.figure(figsize=(16,8))
        plt.grid(1)
        plt.plot(plot_eval_rewards_mean)
        plt.fill_between(range(len(plot_eval_rewards_mean)), plot_eval_rewards_mean + plot_eval_rewards_std, plot_eval_rewards_mean - plot_eval_rewards_std,alpha=0.2)
        # plt.axhline(y = env.spec.reward_threshold, color = 'r', linestyle = '--',label='Solved')
        plt.title(title)
        plt.savefig(f'plots/mts{common_name}.png')    

        
        #record elapsed time and close envs
        end_time=timeit.default_timer()
        print("Elapsed Time: {:.1f} minutes \n".format((end_time-start_time)/60.0))
        
        env.close()
        env_rand.close()