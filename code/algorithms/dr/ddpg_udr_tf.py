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

#ML
import tensorflow as tf
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


def rollout(n_envs, env, policy_agent, RB, T_env, T_agent_init, b_agent, gamma_agent, noise_scale=0.1): 

    rewards_sum = np.zeros(n_envs)
    state = env.reset()

    done = [False] * n_envs
    add_to_buffer = [True] * n_envs
    t_env = 0 #env timestep
    rollouts_len = 0

    while not all(done) and t_env <= T_env:
        action = policy_agent.select_action(np.array(state))
        
        action = action + np.random.normal(0, noise_scale, size=action.shape)
        action = action.clip(-1, 1)

        next_state, reward, done, info = env.step(action)

        #Add samples to replay buffer
        for i, st in enumerate(state):
            if add_to_buffer[i]:
                rewards_sum[i] += reward[i]
                rollouts_len += 1

                done_bool = 0 if t_env + 1 == T_env else float(done[i])
                RB.add((state[i], next_state[i], action[i], reward[i], done_bool))

            if done[i]:
                # Avoid duplicates
                add_to_buffer[i] = False

        state = next_state
        t_env += 1

    # Train agent policy
    if len(RB.storage) > T_agent_init: #if it has enough samples
        # policy_agent.train(RB=RB, eps=rollouts_len,batch_size=b_agent,gamma=gamma_agent)
        policy_agent.train(RB=RB, eps=int(T_env/10),batch_size=b_agent,gamma=gamma_agent)

    return rewards_sum.mean(), rollouts_len

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
    
    #Env
    env_names=['cartpole_custom-v1', 'halfcheetah_custom-v1', 'halfcheetah_custom_norm-v1', 'halfcheetah_custom_rand-v1', 'halfcheetah_custom_rand-v2', 'lunarlander_custom_default_rand-v0']
    env_name=env_names[-2]
    n_workers=10
    
    #Evaluation
    evaluate=True
    log_ival=1
    eval_eps=3
    
    #general
    tr_eps=500
    file_name=os.path.basename(__file__).split(".")[0]
    common_name = "_"+file_name+"_"+env_name
    verbose=1 #or: False/True (False/0: display progress bar; True/1: display 1 log newline per episode) 
    T_rand_rollout=5
    
    #Seed
    # seeds=[None,1,2,3,4,5]
    seeds=[1,2,3]
    # seed = seeds[1]
    
    plot_rewards_all=[]
    plot_eval_rewards_all=[]
    total_timesteps_all=[]
    
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
        
        env_rand=make_vec_envs(env_name, seed, n_workers)
        
        #models
        policy_agent=DDPG(ds, da, h1_agent, h2_agent, lr_agent, a_max)
        
        #memory
        RB=ReplayBuffer()
        
        #Results
        plot_rewards=[]
        plot_eval_rewards=[]
        total_timesteps=[]
        best_reward=-1e6
        t_agent=0
        
        #Evaluation
        eval_rewards_mean=0
        eval_freq = T_env * n_workers
        t_eval=0 # agent timesteps since eval 

        episodes=progress(tr_eps) if not verbose else range(tr_eps)
        # with tqdm.tqdm(total=T_agent) as pbar:
        for episode in episodes:
        # while t_agent < T_agent:
            #get sim instances from SVPG policy if current timestep is greater than the specified initial, o.w. create completely randomized env
            simulation_instances = -1 * np.ones((n_workers,T_rand_rollout,dr))
        
            rewards_agent=np.zeros(T_rand_rollout)
        
            # Reshape to work with vectorized environments
            simulation_instances = np.transpose(simulation_instances, (1, 0, 2))
            
            for t_rand_rollout in range(T_rand_rollout):
                # create ref and randomized instances of the env, rollout the agent in both, and train the agent
                env_rand.randomize(simulation_instances[t_rand_rollout])
                reward_agent, rollouts_len =rollout(n_workers,env_rand,policy_agent,RB,T_env,T_agent_init,b_agent,gamma_agent)
                
                rewards_agent[t_rand_rollout]=reward_agent
                t_agent += rollouts_len
                t_eval += rollouts_len
            
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
                policy_agent.actor.save_weights(f"saved_models/model{common_name}")
            #save plot data
            plot_rewards.append(rewards_agent.mean())
            total_timesteps.append(t_agent)
            #log iteration results & statistics
            # if t_agent % 1== 0:
            if t_agent % log_ival == 0:
                log_msg="Rewards Agent: {:.2f}, Rewards Eval: {:.2f}, Total Timesteps: {}".format(rewards_agent.mean(), eval_rewards_mean, t_agent)
                if verbose:
                    print(log_msg+f" episode:{episode} \n")
                else:
                    episodes.set_description(desc=log_msg); episodes.refresh()
                    # pbar.update(); pbar.set_description(desc=log_msg); pbar.refresh()
                
        plot_rewards_all.append(plot_rewards)
        plot_eval_rewards_all.append(plot_eval_rewards)
        total_timesteps_all.append(total_timesteps)
        
        env.close()
        env_rand.close()
    
    #%% Results & Plots
    #process results
    plot_rewards_mean = np.stack(plot_rewards_all).mean(0)
    plot_eval_rewards_mean = np.stack(plot_eval_rewards_all).mean(0)
    total_timesteps_mean = np.stack(total_timesteps_all).mean(0)
    
    plot_rewards_max= np.maximum.reduce(plot_rewards_all)
    plot_eval_rewards_max = np.maximum.reduce(plot_eval_rewards_all)
    
    plot_rewards_min = np.minimum.reduce(plot_rewards_all)
    plot_eval_rewards_min = np.minimum.reduce(plot_eval_rewards_all)
    
    #save results to df
    df = pd.DataFrame(list(zip(plot_rewards_mean,
                               plot_rewards_max,
                               plot_rewards_min,
                               plot_eval_rewards_mean,
                               plot_eval_rewards_max,
                               plot_eval_rewards_min,
                               total_timesteps_mean)),
                      columns =['Rewards_Tr', 'Rewards_Tr_Max', 'Rewards_Tr_Min', 'Rewards_Eval', 'Rewards_Eval_Max', 'Rewards_Eval_Min', 'Total_Timesteps'])
    df.to_pickle(f"plots/results{common_name}.pkl")
    
    #plot results
    title="Training Rewards"
    plt.figure(figsize=(16,8))
    plt.grid(1)
    plt.plot(plot_rewards_mean)
    plt.fill_between(range(tr_eps), plot_rewards_max, plot_rewards_min,alpha=0.2)
    # plt.axhline(y = env.spec.reward_threshold, color = 'r', linestyle = '--',label='Solved')
    plt.title(title)
    plt.savefig(f'plots/tr{common_name}.png')
    
    title="Evaluation Rewards"
    plt.figure(figsize=(16,8))
    plt.grid(1)
    plt.plot(plot_eval_rewards_mean)
    plt.fill_between(range(len(plot_eval_rewards_max)), plot_eval_rewards_max, plot_eval_rewards_min,alpha=0.2)
    # plt.axhline(y = env.spec.reward_threshold, color = 'r', linestyle = '--',label='Solved')
    plt.title(title)
    plt.savefig(f'plots/ts{common_name}.png')
        