# %% TODOs


# %% Imports
#general
import numpy as np
import pandas as pd
import os
# os.environ["OMP_NUM_THREADS"] = "1"
import yaml
from common import set_seed, progress , parameters_to_vector, PolicyNetwork, ValueNetwork, surrogate_loss, HVP, conjugate_gradients, line_search, adapt

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

#ML
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

#multiprocessing
import multiprocessing as mp
import queue as Q

#%% Utils
        
def collect_rollout_batch(envs, ds, da, policy, T, b, n_workers, queue, params=None): # a batch of rollouts
    states=[[] for _ in range(b)]
    rewards = [[] for _ in range(b)]
    actions = [[] for _ in range(b)]
    
    
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
        for state, next_state, action, reward, rollout_idx in zip(s,s_dash,a,r,rollout_idxs):
            if rollout_idx is not None:
                states[rollout_idx].append(state.astype(np.float32))
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
    
    return D

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


#%% Main Func
if __name__ == '__main__':
    
    #%% Inputs
    
    modes=["debug_mode","run_mode"]
    mode=modes[1]
    
    with open("config.yaml", 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    config=config[mode]
    
    #MAML
    h=config["h_maml"]
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
    
    #Env
    env_name='hopper_custom_rand-v1' #config["env_name"]
    n_workers=config["n_workers"] 
    
    #Evaluation
    evaluate=config["evaluate"]
    log_ival=config["log_ival"]
    eval_eps=config["eval_eps"]
    
    #general
    tr_eps=config["tr_eps"]
    file_name=os.path.basename(__file__).split(".")[0]
    common_name = "_"+file_name+"_"+env_name
    verbose=0 #config["verbose"]
    T_rand_rollout=config["T_rand_rollout"]
    
    #Seed
    seeds=config["seeds"]
    
    plot_tr_rewards_all=[]
    plot_val_rewards_all=[]
    plot_eval_rewards_all=[]
    total_timesteps_all=[]
    best_reward=-1e6
    
    for seed in seeds:
        
        print(f"For Seed: {seed} \n")
  
        #%% Initializations
        #multiprocessing
        queue = mp.Queue()
        lock = mp.Lock()
        
        #environment
        env=gym.make(env_name)
        set_seed(seed,env)
        T_env=env._max_episode_steps #task horizon
        ds=env.observation_space.shape[0] #state dims
        da=env.action_space.shape[0] #action dims
        
        envs=make_vec_envs(env_name, seed, n_workers, ds, da, queue, lock)
        
        #models
        in_size=ds
        out_size=da
        policy = PolicyNetwork(in_size,h,out_size) #dynamics model
        value_net=ValueNetwork(in_size,gamma)
        
        #results 
        plot_tr_rewards=[]
        plot_val_rewards=[]
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
            
            #create empty storages
            rewards_tr_ep, rewards_val_ep = [], []
            Ds, D_dashes=[], []
            
            #inner/adaptation loop
            for t_rand_rollout in range(T_rand_rollout):
                #collect pre-adaptation rollout batch in rand envs (one rollout for each svpg particle)
                D=collect_rollout_batch(envs, ds, da, policy, T_env, b, n_workers, queue)
                Ds.append(D)
                _, _, rewards,_ = D
                rewards_tr_ep.append(rewards)
                t_agent += rewards.size
                
                #adapt agent [meta-]parameters (via VPG w/ baseline)
                theta_dash=adapt(D,value_net,policy,alpha)
                
                #collect post-adaptation rollout batch in rand envs
                D_dash=collect_rollout_batch(envs, ds, da, policy, T_env, b, n_workers, queue, params=theta_dash)
                D_dashes.append(D_dash)
                _, _, rewards,_ = D_dash
                rewards_val_ep.append(rewards)
                t_eval += rewards.size
                t_agent += rewards.size
    
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
    
                for _ in range(eval_eps):                    
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
        total_timesteps_all.append(total_timesteps)
        
        env.close()
        envs.close()
    
    #%% Results & Plot
    #process results
    plot_tr_rewards_mean = np.stack(plot_tr_rewards_all).mean(0)
    plot_val_rewards_mean = np.stack(plot_val_rewards_all).mean(0)
    plot_eval_rewards_mean = np.stack(plot_eval_rewards_all).mean(0)
    total_timesteps_mean = np.stack(total_timesteps_all).mean(0)
    
    plot_tr_rewards_std= np.stack(plot_tr_rewards_all).std(0)
    plot_val_rewards_std = np.stack(plot_val_rewards_all).std(0)
    plot_eval_rewards_std = np.stack(plot_eval_rewards_all).std(0)
    
    #save results to df
    df = pd.DataFrame(list(zip(plot_tr_rewards_mean,
                               plot_tr_rewards_std,
                               plot_val_rewards_mean,
                               plot_val_rewards_std,
                               plot_eval_rewards_mean,
                               plot_eval_rewards_std,
                               total_timesteps_mean)),
                      columns =['Rewards_Tr_Mean', 'Rewards_Tr_Std', 'Rewards_Val_Mean', 'Rewards_Val_Std', 'Rewards_Eval_Mean', 'Rewards_Eval_Std', 'Total_Timesteps'])
    
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
    
    title="Meta-Training Testing Rewards"
    plt.figure(figsize=(16,8))
    plt.grid(1)
    plt.plot(plot_val_rewards_mean)
    plt.fill_between(range(tr_eps), plot_val_rewards_mean + plot_val_rewards_std, plot_val_rewards_mean - plot_val_rewards_std,alpha=0.2)
    # plt.axhline(y = env.spec.reward_threshold, color = 'r', linestyle = '--',label='Solved')
    plt.title(title)
    plt.savefig(f'plots/mtr_ts{common_name}.png')
    
    title="Meta-Testing Rewards"
    plt.figure(figsize=(16,8))
    plt.grid(1)
    plt.plot(plot_eval_rewards_mean)
    plt.fill_between(range(len(plot_eval_rewards_mean)), plot_eval_rewards_mean + plot_eval_rewards_std, plot_eval_rewards_mean - plot_eval_rewards_std,alpha=0.2)
    # plt.axhline(y = env.spec.reward_threshold, color = 'r', linestyle = '--',label='Solved')
    plt.title(title)
    plt.savefig(f'plots/mts{common_name}.png')    
            
