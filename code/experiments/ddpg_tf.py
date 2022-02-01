# %% TODOs


#%% Imports
#General
import numpy as np
import os
# os.environ["OMP_NUM_THREADS"] = "1"
import pandas as pd
import yaml
from common import ReplayBuffer, set_seed, DDPG, progress

#visualization
import matplotlib.pyplot as plt

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

def rollout(n_envs, env, policy_agent, RB, T_env, T_agent_init, b_agent, gamma_agent): 

    rewards_sum = np.zeros(n_envs)
    state = env.reset()

    done = [False] * n_envs
    add_to_buffer = [True] * n_envs
    t_env = 0 #env timestep
    rollouts_len = 0

    while not all(done) and t_env <= T_env:
        action = policy_agent.select_action(np.array(state))

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


def make_vec_envs(env_name, seed, n_workers, ds, da):
    envs=[make_env(env_name,seed,rank) for rank in range(n_workers)]
    envs=SubprocVecEnv(envs, ds, da)
    return envs

    
#%% Algorithm Implementation
if __name__ == '__main__':
    
    #%% Inputs
    
    modes=["debug_mode","run_mode"]
    mode=modes[1]
    
    with open("config.yaml", 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    config=config[mode]
    
    #DDPG
    lr_agent=config["lr_ddpg"]
    h1_agent=config["h1_ddpg"] 
    h2_agent=config["h2_ddpg"]
    gamma_agent=config["gamma_ddpg"]
    T_agent_init=config["T_ddpg_init"]
    b_agent=config["b_ddpg"]
    
    #Env
    env_name=config["env_name"]
    n_workers=config["n_workers"] 
    
    #Evaluation
    evaluate=config["evaluate"]
    log_ival=config["log_ival"]
    eval_eps=config["eval_eps"]
    
    #general
    tr_eps=config["tr_eps"]
    file_name=os.path.basename(__file__).split(".")[0]
    common_name = "_"+file_name+"_"+env_name
    verbose=config["verbose"]
    T_rand_rollout=config["T_rand_rollout"]
    
    #Seed
    seeds=config["seeds"]
    
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
        
        envs=make_vec_envs(env_name, seed, n_workers, ds, da)
        
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
        for episode in episodes:
            rewards_agent=np.zeros(T_rand_rollout)
            #rollout the agent in env, and train the agent
            for t_rand_rollout in range(T_rand_rollout):
                reward_agent, rollouts_len =rollout(n_workers,envs,policy_agent,RB,T_env,T_agent_init,b_agent,gamma_agent)
                rewards_agent[t_rand_rollout]=reward_agent
                t_agent += rollouts_len
                t_eval += rollouts_len
        
            #evaluate
            if evaluate and t_eval>eval_freq:
                t_eval %= eval_freq
                eval_rewards = []
                for _ in range(eval_eps):
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
        envs.close()
    
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
        
