# %% TODOs


# %% Imports
#general
import numpy as np
import pandas as pd
import os
# os.environ["OMP_NUM_THREADS"] = "1"
import yaml
from common import set_seed, progress , parameters_to_vector, PolicyNetwork, ValueNetwork, surrogate_loss, HVP, conjugate_gradients, line_search, adapt, SVPG, Discriminator

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

#utils
import decimal

#ML
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


#multiprocessing
import multiprocessing as mp
import queue as Q


#%% Utils
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
    
    #Discriminator
    r_disc_scale = config["r_disc_scale"] 
    h_disc=config["h_disc"] 
    lr_disc=config["lr_disc"] 
    b_disc=config["b_disc"] 
    
    #SVPG
    n_particles=config["n_particles"] 
    temp=config["temp"]
    type_particles=config["type_particles"] 
    kld_coeff=config["kld_coeff"] 
    T_svpg_reset=config["T_svpg_reset"] 
    delta_max=config["delta_max"] 
    T_svpg_init=config["T_svpg_init"] 
    T_svpg=config["T_svpg"]
    lr_svpg=config["lr_svpg"] 
    gamma_svpg=config["gamma_svpg"]
    h_svpg=config["h_svpg"]
    
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
    
    plot_disc_rewards_all=[]
    plot_tr_rewards_all=[]
    plot_val_rewards_all=[]
    plot_eval_rewards_all=[]
    total_timesteps_all=[]
    
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
        dr=env.unwrapped.randomization_space.shape[0] #N_rand (no. of randomization params)
        
        env_ref=make_vec_envs(env_name, seed, n_workers, ds, da, queue, lock)
        env_rand=make_vec_envs(env_name, seed, n_workers, ds, da, queue, lock)

        #models
        in_size=ds
        out_size=da
        policy = PolicyNetwork(in_size,h,out_size) #dynamics model
        value_net=ValueNetwork(in_size,gamma)
        discriminator=Discriminator(ds, da, h_disc, b_disc, r_disc_scale, lr_disc)
        svpg = SVPG(n_particles, dr, h_svpg, delta_max, T_svpg, T_svpg_reset, temp, kld_coeff, lr_svpg, gamma_svpg, type_particles)
        
        #results 
        plot_tr_rewards=[]
        plot_val_rewards=[]
        plot_eval_rewards=[]
        plot_disc_rewards=[]
        total_timesteps=[]
        t_agent = 0
        best_reward=-1e6
        sampled_regions = [[] for _ in range(dr)]
        
        #evaluation
        eval_rewards_mean=0
        eval_freq = T_env * n_particles
        t_eval=0 #agent timesteps since eval 
        
        #%% Implementation
        episodes=progress(tr_eps) if not verbose else range(tr_eps)
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
                D,_=collect_rollout_batch(env_rand, ds, da, policy, T_env, b, n_workers, queue,add_noise=True)
                Ds.append(D)
                _, _, rewards,_ = D
                rewards_tr_ep.append(rewards)
                t_agent += rewards.size
                
                #adapt agent [meta-]parameters (via VPG w/ baseline)
                theta_dash=adapt(D,value_net,policy,alpha)
                
                #collect post-adaptaion rollout batch in ref envs
                _,ref_traj=collect_rollout_batch(env_ref, ds, da, policy, T_env, b, n_workers, queue, params=theta_dash)
                D_dashes_ref.append(ref_traj)
                
                #collect post-adaptation rollout batch in rand envs
                D_dash,rand_traj=collect_rollout_batch(env_rand, ds, da, policy, T_env, b, n_workers, queue, params=theta_dash,add_noise=True)
                D_dashes_rand.append(rand_traj)
                D_dashes.append(D_dash)
                _, _, rewards,_ = D_dash
                rewards_val_ep.append(rewards)
                t_agent += rewards.size
            
            #ADR updates
            for t, (ref_traj, rand_traj) in enumerate(zip(D_dashes_ref,D_dashes_rand)):
                T_disc_eps=0 #agent timesteps in the current iteration/episode
    
                #calculate discriminator reward
                for i in range(n_particles):
                    T_disc_eps += len(rand_traj[i])
                    t_eval += len(rand_traj[i])
                    
                    r_disc, score_disc = discriminator.calculate_rewards(rand_traj[i])
                    rewards_disc[i][t]= r_disc
                    scores_disc[i][t]=score_disc
                
                #train discriminator
                flattened_rand = np.concatenate(rand_traj)
                flattened_ref = np.concatenate(ref_traj)
                discriminator.train(ref_traj=flattened_ref, rand_traj=flattened_rand, eps=T_disc_eps)
            
            plot_disc_rewards.append(scores_disc.mean())
            
            #update svpg particles' params (ie. train their policies)
            if episode >= T_svpg_init:
                svpg.train(rewards_disc)
                
                #log sampled regions only once svpg particles start training (i.e. once adr starts)
                for dim in range(dr):
                    low=env.unwrapped.dimensions[dim].range_min
                    high=env.unwrapped.dimensions[dim].range_max
                    scaled_instances=low + (high-low) * simulation_instances[:, :, dim]
                    sampled_regions[dim]=np.concatenate([sampled_regions[dim],scaled_instances.flatten()])
    
            #outer loop: update meta-params (via: TRPO) #!!!: since MAML uses TRPO it is on-policy, so care should be taken that order of associated transitions is preserved
            with tf.GradientTape() as tape:
                prev_loss, _, prev_pis = surrogate_loss(Ds, D_dashes,policy,value_net,gamma,alpha)
            grads = parameters_to_vector(tape.gradient(prev_loss, policy.trainable_variables),policy.trainable_variables)
            prev_loss=tf.identity(prev_loss)
            hvp=HVP(Ds,D_dashes,policy,value_net,alpha,damping)
            search_step_dir=conjugate_gradients(hvp, grads, rdotr_tol=rdotr_tol,nsteps=nsteps)
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
                log_msg="Rewards Tr: {:.2f}, Rewards Val: {:.2f}, Rewards Disc: {:.2f}, Rewards Eval: {:.2f}, Total Timesteps: {}".format(reward_ep, reward_val, scores_disc.mean(), eval_rewards_mean, t_agent)
                if verbose:
                    print(log_msg+f" episode:{episode} \n")
                else:
                    episodes.set_description(desc=log_msg); episodes.refresh()
        
        plot_tr_rewards_all.append(plot_tr_rewards)
        plot_val_rewards_all.append(plot_val_rewards)
        plot_eval_rewards_all.append(plot_eval_rewards)
        plot_disc_rewards_all.append(plot_disc_rewards)
        total_timesteps_all.append(total_timesteps)
        
        env.close()
        env_ref.close()
        env_rand.close()
        
    #%% Results & Plot
    #process results
    plot_tr_rewards_mean = np.stack(plot_tr_rewards_all).mean(0)
    plot_val_rewards_mean = np.stack(plot_val_rewards_all).mean(0)
    plot_eval_rewards_mean = np.stack(plot_eval_rewards_all).mean(0)
    plot_disc_rewards_mean = np.stack(plot_disc_rewards_all).mean(0)
    total_timesteps_mean = np.stack(total_timesteps_all).mean(0)
    
    plot_tr_rewards_max= np.maximum.reduce(plot_tr_rewards_all)
    plot_val_rewards_max = np.maximum.reduce(plot_val_rewards_all)
    plot_eval_rewards_max = np.maximum.reduce(plot_eval_rewards_all)
    plot_disc_rewards_max = np.maximum.reduce(plot_disc_rewards_all)
    
    plot_tr_rewards_min = np.minimum.reduce(plot_tr_rewards_all)
    plot_val_rewards_min = np.minimum.reduce(plot_val_rewards_all)
    plot_eval_rewards_min = np.minimum.reduce(plot_eval_rewards_all)
    plot_disc_rewards_min = np.minimum.reduce(plot_disc_rewards_all)
    
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
                               plot_disc_rewards_mean,
                               plot_disc_rewards_max,
                               plot_disc_rewards_min,
                               total_timesteps_mean)),
                      columns =['Rewards_Tr', 'Rewards_Tr_Max', 'Rewards_Tr_Min', 'Rewards_Val', 'Rewards_Val_Max','Rewards_Val_Min', 'Rewards_Eval', 'Rewards_Eval_Max', 'Rewards_Eval_Min', 'Rewards_Disc', 'Rewards_Disc_Max', 'Rewards_Disc_Min', 'Total_Timesteps'])
    
    df.to_pickle(f"plots/results{common_name}.pkl")
    
    #plot results
    title="Meta-Training Training Rewards"
    plt.figure(figsize=(16,8))
    plt.grid(1)
    plt.plot(plot_tr_rewards_mean)
    plt.fill_between(range(tr_eps), plot_tr_rewards_max, plot_tr_rewards_min,alpha=0.2)
    # plt.axhline(y = env.spec.reward_threshold, color = 'r', linestyle = '--',label='Solved')
    plt.title(title)
    plt.savefig(f'plots/mtr_tr{common_name}.png')
    
    title="Meta-Training Testing Rewards"
    plt.figure(figsize=(16,8))
    plt.grid(1)
    plt.plot(plot_val_rewards_mean)
    plt.fill_between(range(tr_eps), plot_val_rewards_max, plot_val_rewards_min,alpha=0.2)
    # plt.axhline(y = env.spec.reward_threshold, color = 'r', linestyle = '--',label='Solved')
    plt.title(title)
    plt.savefig(f'plots/mtr_ts{common_name}.png')
    
    title="Meta-Testing Rewards"
    plt.figure(figsize=(16,8))
    plt.grid(1)
    plt.plot(plot_eval_rewards_mean)
    plt.fill_between(range(len(plot_eval_rewards_max)), plot_eval_rewards_max, plot_eval_rewards_min,alpha=0.2)
    # plt.axhline(y = env.spec.reward_threshold, color = 'r', linestyle = '--',label='Solved')
    plt.title(title)
    plt.savefig(f'plots/mts{common_name}.png')
    
    title="Discriminator Rewards"
    plt.figure(figsize=(16,8))
    plt.grid(1)
    plt.plot(plot_disc_rewards_mean)
    plt.fill_between(range(tr_eps), plot_disc_rewards_max, plot_disc_rewards_min,alpha=0.2)
    plt.title(title)
    plt.savefig(f'plots/disc{common_name}.png')
        
    eps_step=int((tr_eps-T_svpg_init)/4)
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
        
        title=f"Sampled Regions for Randomization Dim = {dim_name} {env.rand} Over Time"
        plt.figure(figsize=(16,8))
        plt.grid(1)
        plt.hist((regions[region_step*0:region_step*1],regions[region_step*1:region_step*2],regions[region_step*2:region_step*3], regions[region_step*3:]), np.arange(min(x),max(x)+2*step,step), histtype='barstacked', label=[f'{eps_step*1} eps',f'{eps_step*2} eps', f'{eps_step*3} eps', f'{eps_step*4} eps'],color=["lightskyblue","blueviolet","hotpink","lightsalmon"])
        plt.xlim(min(x), max(x)+step)
        plt.legend()
        plt.title(title)
        #save results
        plt.savefig(f'plots/sampled_regions_dim_{dim_name}_{env.rand}{common_name}.png')
        df2[f'Sampled_Regions_{dim_name}_{env.rand}'] = list(regions)
    
    df2.to_pickle(f"plots/sampled_regions{common_name}.pkl")
        
