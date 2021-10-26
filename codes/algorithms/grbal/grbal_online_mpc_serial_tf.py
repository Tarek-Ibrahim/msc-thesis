# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 00:59:53 2021

"""

# %% TODOs
# TODO: upgrade to parallel data/rollout batch collection
# TODO: upgrade to parallel cost computation (create a vectorized controller abstraction like the env)
# TODO: upgrade from MPC to PETS
# TODO: upgrade to tasks batch
# TODO: include noise
# TODO: modularize
# TODO: code clean-up


#%% Imports

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
from scipy.stats import truncnorm
from collections import OrderedDict, namedtuple

#ML
import tensorflow as tf
import tensorflow_probability as tfp

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

def collect_rollouts(env,controller,model,T,M,loss_func,b):
    #sample a rollout batch from the agent
    
    rollout_batch = []

    for _ in range(b):
        controller.reset() #amounts to resetting CEM optimizer's prev sol to its initial value (--> array of size H with all values = avg of action space value range)
        o=env.reset()
        O, A, rewards, O_dash= [], [], [], []
        for t in range(T):
            
            #adapt parameters to last M timesteps
            # if A:
            if len(A)>=M:
                with tf.GradientTape() as tape:
                    #construct inputs and targets from previous M timesteps
                    obs=np.array(O[-M:])
                    acs=np.array(A[-M:])
                    obs_dash=np.array(O_dash[-M:])
                    inputs=np.concatenate([obs, acs], axis=-1)
                    targets = obs_dash - obs
                    inputs=tf.convert_to_tensor(inputs,dtype=tf.float32)
                    targets=tf.convert_to_tensor(targets,dtype=tf.float32)
                    
                    #calculate adapted parameters
                    mean, logvar, dist = model(inputs)
                    loss= compute_loss(loss_func,mean,targets,logvar,dist,model.var_type,model)
                grads = tape.gradient(loss, model.trainable_variables)
                theta_dash=model.update_params(grads)
            else:
                theta_dash=None
                
            a=controller.act(o,params=theta_dash) #use controller to plan and choose optimal action as first action in sequence
    
            o_dash, r, done, _ = env.step(a) #execute first action from optimal actions
            
            A.append(a)
            O.append(o)
            rewards.append(r)
            O_dash.append(o_dash)
            
            o=o_dash
            
            if done:
                break
        
        rollout_batch.append([np.array(O), np.array(A), np.array(rewards), np.array(O_dash)])

    return rollout_batch


def compute_loss(loss_func,mean,targets,logvar,dist,var_type,model):
    if loss_func=="nll":
        loss= - dist.log_prob(targets)
    else: #mse
        if var_type=="out":
            var=tf.math.exp(-logvar)
            loss= tf.math.square(mean - targets)*var+logvar
        else:
            loss= tf.math.square(mean - targets)*dist.scale-tf.math.log(dist.scale)
    loss=tf.math.reduce_mean(tf.math.reduce_sum(loss,-1))
    if var_type=="out":
        loss += 0.01 * (tf.math.reduce_sum(model.max_logvar) - tf.math.reduce_sum(model.min_logvar))
    return loss

#--------
# Models
#--------

class MLP(tf.keras.Model):
    def __init__(self, in_size, n, h, out_size,alpha,var,var_type):
        super().__init__()
        
        self.n=n
        self.in_size=in_size
        self.out_size=out_size
        self.alpha=alpha
        self.var_type = var_type
        
        self.logstd= tf.Variable(initial_value=np.log(1.0)* tf.keras.initializers.Ones()(shape=[1,out_size]),dtype=tf.float32,name='logstd') if var_type=="param" else np.log(var)*tf.ones(self.out_size,dtype=tf.float32)
        
        if var_type=="out":
            self.max_logvar = tf.Variable(initial_value=0.5 * tf.keras.initializers.Ones()(shape=[1,out_size//2]),dtype=tf.float32,name='max_logvar')
            self.min_logvar = tf.Variable(initial_value=-10.0 * tf.keras.initializers.Ones()(shape=[1,out_size//2]),dtype=tf.float32,name='min_logvar')
        
        self.mu = tf.Variable(initial_value=tf.keras.initializers.Zeros()(shape=[1,self.in_size]), trainable=False) #tf.zeros(shape=[1,self.in_size],dtype=tf.float32) #
        self.sigma = tf.Variable(initial_value=tf.keras.initializers.Ones()(shape=[1,self.in_size]), trainable=False) #tf.ones(shape=[1,self.in_size],dtype=tf.float32) #
        
        self.w1= tf.Variable(initial_value=tf.keras.initializers.glorot_uniform()(shape=(in_size, h)), dtype=tf.float32, name='layer1/weight')
        self.b1=tf.Variable(initial_value=tf.keras.initializers.Zeros()(shape=(h,)), dtype=tf.float32, name='layer1/bias')
        
        self.w2= tf.Variable(initial_value=tf.keras.initializers.glorot_uniform()(shape=(h, h)), dtype=tf.float32, name='layer2/weight')
        self.b2=tf.Variable(initial_value=tf.keras.initializers.Zeros()(shape=(h,)), dtype=tf.float32, name='layer2/bias')
        
        self.w3= tf.Variable(initial_value=tf.keras.initializers.glorot_uniform()(shape=(h, h)), dtype=tf.float32, name='layer3/weight')
        self.b3=tf.Variable(initial_value=tf.keras.initializers.Zeros()(shape=(h,)), dtype=tf.float32, name='layer3/bias')
        
        self.w4= tf.Variable(initial_value=tf.keras.initializers.glorot_uniform()(shape=(h, out_size)), dtype=tf.float32, name='layer4/weight')
        self.b4=tf.Variable(initial_value=tf.keras.initializers.Zeros()(shape=(out_size,)), dtype=tf.float32, name='layer4/bias')
                
        self.nonlinearity=tf.nn.relu
    
    def fit_input_stats(self, inputs):
        #get mu and sigma of [all] input data fpr later normalization of model [batch] inputs

        mu = np.mean(inputs, axis=0, keepdims=True) #over cols (each observation/action col of input) and keeping same col size --> result has size = (1,input_size)
        sigma = np.std(inputs, axis=0, keepdims=True)
        sigma[sigma < 1e-12] = 1.0 #???: why 1 (and not 1e-12 e.g.)??

        self.mu.assign(mu)
        self.sigma.assign(sigma)
            
    def update_params(self, grads):
        named_parameters = [(param.name, param) for param in self.trainable_variables]
        new_params = OrderedDict()
        for (name,param), grad in zip(named_parameters, grads):
            new_params[name]= param - self.alpha * grad
        return new_params
    
    def call(self, inputs, params=None, normalize=False):
        if params is None:
            params=OrderedDict((param.name, param) for param in self.trainable_variables)
            
        #normalize inputs
        if normalize: inputs = (inputs - self.mu) / self.sigma
 
        inputs=self.nonlinearity(tf.math.add(tf.linalg.matmul(inputs,params['layer1/weight:0']),params['layer1/bias:0']))
        inputs=self.nonlinearity(tf.math.add(tf.linalg.matmul(inputs,params['layer2/weight:0']),params['layer2/bias:0']))
        inputs=self.nonlinearity(tf.math.add(tf.linalg.matmul(inputs,params['layer3/weight:0']),params['layer3/bias:0']))
        inputs=tf.math.add(tf.linalg.matmul(inputs,params['layer4/weight:0']),params['layer4/bias:0'])
        
        if self.var_type == "out":
            #extract mean and log(var) from network output
            mean = inputs[..., :self.out_size // 2]
            logvar = inputs[..., self.out_size // 2:]
        
            #bounding variance (becase network gives arbitrary variance for OOD points --> could lead to numerical problems)
            logvar = params['max_logvar:0'] - tf.nn.softplus(params['max_logvar:0'] - logvar) # self.max_logvar - tf.nn.softplus(self.max_logvar - logvar)
            logvar = params['min_logvar:0'] + tf.nn.softplus(logvar - params['min_logvar:0']) #  self.min_logvar + tf.nn.softplus(logvar - self.min_logvar)
            
            std=tf.math.exp(-logvar)

        else:
            mean=inputs
            std= tf.math.exp(tf.math.maximum(params["logstd:0"], np.log(1e-6))) if self.var_type=="param" else tf.math.exp(self.logstd )
            logvar=tf.math.log(std)

        #dist=tfp.distributions.MultivariateNormalDiag(mean,tf.linalg.diag(std))
        #dist=tfp.distributions.Independent(tfp.distributions.Normal(mean,std),1)

        return mean, logvar, tfp.distributions.Normal(mean,std)

    
class MPC:
    def __init__(self,env,model,H,pop_size,opt_max_iters):
        self.H=H
        self.pop_size=pop_size
        self.opt_max_iters=opt_max_iters
        self.model=model
        
        self.ds=env.observation_space.shape[0] #state/observation dims
        self.da=env.action_space.shape[0] #action dims
        self.ac_lb= env.action_space.low #env.ac_lb
        self.ac_ub= env.action_space.high #env.ac_ub
        self.cost_obs= env.cost_o
        self.cost_acs= env.cost_a
        self.reset() #sol's initial mu/mean
        self.init_var= np.tile(((self.ac_ub - self.ac_lb) / 4.0)**2, [self.H]) #sol's intial variance

    def reset(self):
        self.prev_sol = np.tile((self.ac_lb + self.ac_ub) / 2.0, [self.H])
    
    def act(self, ob, params=None):
        self.params=params
        sol=self.CEM(ob) #get CEM optimizer's sol
        self.prev_sol=np.concatenate([np.copy(sol)[self.da:], np.zeros(self.da)])
        action = sol[:self.da] #has the first action in the sequence = optimal action
        return action
        
    def CEM(self, ob):
        sol_dim=self.H*self.da #dimension of an action sequence
        epsilon=0.001 #termination condition representing min variance allowable
        alpha=0.25 #0.1 #controls how much of mean & variance is used for next iteration
        lb=np.tile(self.ac_lb,[self.H])
        ub=np.tile(self.ac_ub,[self.H])
        
        mean, var, t = self.prev_sol, self.init_var, 0.0
        X = truncnorm(-2, 2, loc=np.zeros_like(mean), scale=np.ones_like(mean))

        while (t < self.opt_max_iters) and np.max(var) > epsilon:
            lb_dist = mean - lb
            ub_dist = ub - mean
            constrained_var = np.minimum(np.minimum((lb_dist / 2.0)**2, (ub_dist / 2.0)**2), var)
            
            #1- generate sols
            samples = X.rvs(size=[self.pop_size, sol_dim]) * np.sqrt(constrained_var) + mean #sample from destandardized/denormalized distributiion #???: this method of random action seq generation is a bit weird because actions in an action sequence are not really related and does it also assume that all actions are available from all states?
            samples = samples.astype(np.float32)

            #2- propagate state & evaluate actions
            costs = self.get_costs(ob,samples)

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
        
    def get_costs(self,ob,ac_seqs):
                
        #reshape ac_seqs --> from (pop_size,sol_dim[H*da]) to (H,pop_size,da)
        ac_seqs = tf.transpose(tf.reshape(ac_seqs, [-1, self.H, self.da]),[1, 0, 2])

        #reshape obs --> from (ds) to (pop_size,ds)
        obs=tf.tile(ob[None].astype(np.float32), [self.pop_size, 1])
        
        costs = tf.zeros(self.pop_size)
        
        for t in range(self.H):
            curr_acs=ac_seqs[t]
            
            #predict next observations
            inputs=tf.concat([obs, curr_acs], axis=-1)
            mean, logvar, dist  = self.model(inputs,self.params)
            if self.model.var_type=="out":
                var=tf.math.exp(logvar)
                delta_obs_next = mean + tf.random.normal(mean.shape) * tf.math.sqrt(var)
            else:
                delta_obs_next = dist.sample()
            obs_next=obs + delta_obs_next
            
            cost=self.cost_obs(obs_next)+self.cost_acs(curr_acs) #evaluate actions
            costs += cost
            obs = obs_next
        
        costs=tf.where(tf.math.is_nan(costs), 1e6 * tf.ones_like(costs), costs) #replace NaNs with a high cost
        
        
        return costs

# %% Main Function
def main():
    
    #%% Inputs
    #model / policy
    n=3 #no. of NN layers
    h=100 #512 #size of hidden layers
    var=0.5 #1. #NN variance if using a fixed varaince
    var_types=[None,"out","param"] #out=output of the network; param=a learned parameter; None=fixed variance
    var_type=var_types[1]
    normalize=False #whether to normalize model inputs
    
    #optimizer
    alpha=0.01 #adaptation step size / learning rate
    beta=0.001 #meta step size / learning rate
    
    #general
    # trials=30 #no. of trials
    tr_eps=20 #50 #no. of training episodes/iterations
    # eval_eps=1
    log_ival=1 #logging interval
    b=1 #4 #32 #batch size: Number of rollouts to sample from each task
    # meta_b=16 #32 #number of tasks sampled
    seed=1
    
    #controller
    H=8 #planning horizon
    # epochs=5 #propagation method epochs
    pop_size=60 #1000 #CEM population size: number of candidate solutions to be sampled every iteration 
    opt_max_iters=5 #5 #CEM's max iterations (used as a termination condition)
    
    #algorithm
    M=16 #no. of prev timesteps
    K=M #no. of future timesteps (adaptation horizon)
    N=16 #no. of sampled tasks (fluid definition)
    ns=1 #3 #10 #task sampling frequency
    loss_funcs=["nll","mse"] #nll: negative log loss; mse: mean squared error
    loss_func= loss_funcs[1]
    traj_b=b #trajectory batch size (no. of trajectories sampled per sampled task)
    # gamma= 1. #discount factor
    
    #%% Initializations
    #common
    D=[] #dataset / experience (a list of rollouts [where each rollout is a list of arrays (s,a,r,s')])
    
    #environment
    env_names=['cartpole_custom-v2','halfcheetah_custom-v2']
    env_name=env_names[1]
    env=gym.make(env_name)
    task_name="cripple"
    T=env._max_episode_steps #task horizon
    ds=env.observation_space.shape[0] #state dims
    da=env.action_space.shape[0] #action dims
    
    #models
    in_size=ds+da
    out_size=ds*2 if var_type=="out" else ds
    model = MLP(in_size,n,h,out_size,alpha,var,var_type) #dynamics model
    controller = MPC(env,model,H,pop_size,opt_max_iters)
    optimizer = tf.keras.optimizers.Adam(learning_rate=beta) #model optimizer
    
    #results 
    plot_tr_rewards=[]
    best_reward=-1e6
    
    set_seed(seed,env)
    
    #%% Sanity Checks
    assert H<K, "planning horizon H should be smaller than the adaptation horizon K, since the adapted model is only valid within the current context"
    assert T>=M+K, "rollout length (task horizon / max no. of env timesteps) T has to at least be equal to the prev + future adaptation timesteps"
    assert H<=T, "planning horizon must be at most the total no. of environment timesteps"
    
    #%% Algorithm
    
    episodes=progress(tr_eps)
    for episode in episodes:
        
        if episode==0 or episode % ns == 0:
            task = env.sample_task()
            env.reset_task(task,task_name)
            rollout_batch=collect_rollouts(env, controller, model, T,M,loss_func,b)
            reward_ep=np.mean([np.sum(rollout[2]) for rollout in rollout_batch])
            D.extend(rollout_batch)
        
        model.fit_input_stats(np.concatenate([np.concatenate((rollout[0],rollout[1]),-1) for rollout in D]))
        
        # for _ in range(epochs):
        #adaptation
        with tf.GradientTape() as outer_tape:
            losses=[]
            for i in range(N):
                
                #sample [a batch of] trajectories randomly from the dataset
                m_trajs, k_trajs = [], []
                for _ in range(traj_b):
                    rollout = D[np.random.choice(len(D))] #pick a rollout at random
                    m_start_idx = np.random.choice(len(rollout[0]) + 1 - M - K)
                    m_traj=[r[m_start_idx: m_start_idx + M] for r in rollout]
                    k_traj=[r[m_start_idx + M : m_start_idx + M + K] for r in rollout]
                    m_trajs = m_traj if m_trajs==[] else [np.concatenate((m_trajs[dim], m_traj[dim])) for dim in range(len(m_traj))]
                    k_trajs = k_traj if k_trajs==[] else [np.concatenate((k_trajs[dim], k_traj[dim])) for dim in range(len(k_traj))]
                
                #adapt params
                #construct inputs and targets from m_trajs
                obs=m_trajs[0]
                acs=m_trajs[1]
                obs_dash=m_trajs[-1]
                inputs=np.concatenate([obs, acs], axis=-1)
                targets = obs_dash - obs
                inputs=tf.convert_to_tensor(inputs,dtype=tf.float32)
                targets=tf.convert_to_tensor(targets,dtype=tf.float32)
                #compute adapted parameters
                with tf.GradientTape() as inner_tape:
                    mean, logvar, dist  = model(inputs,normalize=normalize)
                    loss= compute_loss(loss_func,mean,targets,logvar,dist,model.var_type,model)
                grads = inner_tape.gradient(loss, model.trainable_variables)
                theta_dash = model.update_params(grads)
                
                #compute loss
                #construct inputs and targets from k_trajs
                obs=k_trajs[0]
                acs=k_trajs[1]
                obs_dash=k_trajs[-1]
                inputs=np.concatenate([obs, acs], axis=-1)
                targets = obs_dash - obs
                inputs=tf.convert_to_tensor(inputs,dtype=tf.float32)
                targets=tf.convert_to_tensor(targets,dtype=tf.float32)
                #compute task loss
                mean, logvar, dist  = model(inputs,params=theta_dash,normalize=normalize)
                loss= compute_loss(loss_func,mean,targets,logvar,dist,model.var_type,model)
                losses.append(loss)
            meta_loss=tf.math.reduce_mean(tf.stack(losses, axis=0))
            
        #meta update
        grads = outer_tape.gradient(meta_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        
        #save best running model [params]
        if reward_ep>best_reward: 
            best_reward=reward_ep
            model.save_weights("saved_models/"+env_name)
            
        #log iteration results & statistics
        plot_tr_rewards.append(reward_ep)
        if episode % log_ival == 0:
            log_msg="Rewards Tr: {:.2f}".format(reward_ep)
            episodes.set_description(desc=log_msg); episodes.refresh()
        
    #%% Results & Plot
    title="Meta-Training Rewards (Learning Curve)"
    plt.figure(figsize=(16,8))
    plt.grid(1)
    plt.plot(plot_tr_rewards)
    plt.title(title)
    plt.show()

#%%
if __name__ == '__main__':
    main()