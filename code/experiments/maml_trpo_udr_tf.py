# %% TODOs


# %% Imports
#general
import numpy as np
import pandas as pd
import os
# os.environ["OMP_NUM_THREADS"] = "1"
import yaml
from common import set_seed, progress , parameters_to_vector, PolicyNetwork, ValueNetwork, surrogate_loss, HVP, conjugate_gradients, line_search, adapt, make_vec_envs, collect_rollout_batch
import timeit

#env
import gym
import gym_custom

#visualization
import matplotlib.pyplot as plt

#ML
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

#multiprocessing
import multiprocessing as mp


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
        
        #environment
        env=gym.make(env_name)
        set_seed(seed,env)
        T_env=env._max_episode_steps #task horizon
        ds=env.observation_space.shape[0] #state dims
        da=env.action_space.shape[0] #action dims
        dr=env.unwrapped.randomization_space.shape[0] #N_rand (no. of randomization params)
        
        env_rand=make_vec_envs(env_name, seed, n_workers, ds, da, queue, lock)
        
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
                D,_=collect_rollout_batch(env_rand, ds, da, policy, T_env, b, n_workers, queue,add_noise=True)
                Ds.append(D)
                _, _, rewards,_ = D
                rewards_tr_ep.append(rewards)
                t_agent += rewards.size
                
                #adapt agent [meta-]parameters (via VPG w/ baseline)
                theta_dash=adapt(D,value_net,policy,alpha)
                
                #collect post-adaptation rollout batch in rand envs
                D_dash,_=collect_rollout_batch(env_rand, ds, da, policy, T_env, b, n_workers, queue, params=theta_dash,add_noise=True)
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
                log_msg="Rewards Tr: {:.2f}, Rewards Val: {:.2f}, Rewards Eval: {:.2f}, Total Timesteps: {}".format(reward_ep, reward_val, eval_rewards_mean, t_agent)
                if verbose:
                    print(log_msg+f" episode:{episode} \n")
                else:
                    episodes.set_description(desc=log_msg); episodes.refresh()
                    
        
        plot_tr_rewards_all.append(plot_tr_rewards)
        plot_val_rewards_all.append(plot_val_rewards)
        plot_eval_rewards_all.append(plot_eval_rewards)
        total_timesteps_all.append(total_timesteps)
        
    
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

        
        #record elapsed time and close envs
        end_time=timeit.default_timer()
        print("Elapsed Time: {:.1f} minutes \n".format((end_time-start_time)/60.0))
        
        env.close()
        env_rand.close()