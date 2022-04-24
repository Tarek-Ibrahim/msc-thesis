# %% TODOs


# %% Imports
#general
import numpy as np
import pandas as pd
# import os
# os.environ["OMP_NUM_THREADS"] = "1"
import yaml
from common import set_seed, progress , parameters_to_vector, PolicyNetwork, ValueNetwork, surrogate_loss, HVP, conjugate_gradients, line_search, SVPG, Discriminator, collect_rollout_batch, make_vec_envs

#env
import gym
import gym_custom

#visualization
import matplotlib.pyplot as plt

#utils
# import decimal
import timeit

#ML
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


#multiprocessing
import multiprocessing as mp

#%% Main Func
if __name__ == '__main__':
    
    #%% Inputs
    
    modes=["debug_mode","run_mode"]
    mode=modes[0]
    
    with open("config.yaml", 'r') as f:
        config_file = yaml.load(f, Loader=yaml.FullLoader)
    
    config=config_file[mode]
    
    #MAML
    h=config["h_maml"]
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
    env_key="lunarlander" #"hopper_friction"
    env_name=config_file["env_names"][env_key] #config["env_name"]
    n_workers=config["n_workers"] 
    
    #Evaluation
    evaluate=config["evaluate"]
    log_ival=config["log_ival"]
    eval_eps=config["eval_eps"]
    e=0.1
    
    #general
    tr_eps=config["tr_eps"]
    file_name="trpo_adr" #os.path.basename(__file__).split(".")[0]
    common_name = "_"+file_name+"_"+env_key
    verbose=config["verbose"]
    T_rand_rollout=config["T_rand_rollout"]
    load_policy=False
    # writer = tf.summary.create_file_writer("logs")
    # tf.summary.trace_on(graph=True, profiler=True)
    
    #Seed
    seeds=[1] #config["seeds"]
    
    plot_disc_rewards_all=[]
    plot_tr_rewards_all=[]
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
        
        env_ref=make_vec_envs(env_name, seed, n_workers, ds, da, queue, lock)
        env_rand=make_vec_envs(env_name, seed, n_workers, ds, da, queue, lock)
        
        #models
        in_size=ds
        out_size=da
        policy = PolicyNetwork(in_size,h,out_size) #dynamics model
        if load_policy:
            policy.load_weights(f"saved_models/model{common_name}")
        value_net=ValueNetwork(in_size,gamma)
        discriminator=Discriminator(ds, da, h_disc, b_disc, r_disc_scale, lr_disc)
        svpg = SVPG(n_particles, dr, h_svpg, delta_max, T_svpg, T_svpg_reset, temp, kld_coeff, lr_svpg, gamma_svpg, type_particles)
        
        #results 
        plot_tr_rewards=[]
        plot_eval_rewards=[]
        plot_disc_rewards=[]
        total_timesteps=[]
        t_agent = 0
        sampled_regions = [[] for _ in range(dr)]
        
        #evaluation
        eval_rewards_mean=0.
        eval_freq = 1. #T_env * n_particles
        t_eval=0 #agent timesteps since eval
        eval_step = 1.0 / eval_eps
        running_reward = 0.
        
        #%% Implementation
        episodes=progress(tr_eps) if not verbose else range(tr_eps)
        for episode in episodes:
            
            #rollout svpg particles (each particle represents a different rand env and each timestep in that rollout represents different values for its randomization params)
            simulation_instances = svpg.step() if episode >= T_svpg_init else -1 * np.ones((n_particles,T_svpg,dr))
            
            #create empty storages
            rewards_tr_ep = []
            D_dashes, D_dashes_ref, D_dashes_rand=[], [], []
            rewards_disc = np.zeros(simulation_instances.shape[:2])
            scores_disc=np.zeros(simulation_instances.shape[:2])
            
            # Reshape to work with vectorized environments
            simulation_instances = np.transpose(simulation_instances, (1, 0, 2))
            
            #inner/adaptation loop
            for t_svpg in range(T_svpg):
                
                #collect post-adaptaion rollout batch in ref envs
                _,ref_traj=collect_rollout_batch(env_ref, ds, da, policy, T_env, b, n_workers, queue)
                D_dashes_ref.append(ref_traj)
                
                #randomize rand envs (with svpg particle [randomization parameter] values [at the current svpg timestep]) #!!!: this only works if transitions within the same rollout is collected in the same environment (i.e. by the same [env] worker) #that's why we choose: n_particles=b(rollout batch size)=n_workers/n_envs
                env_rand.randomize(simulation_instances[t_svpg])
                
                #collect rollout batch in rand envs
                D_dash,rand_traj=collect_rollout_batch(env_rand, ds, da, policy, T_env, b, n_workers, queue,add_noise=True)
                D_dashes_rand.append(rand_traj)
                D_dashes.append(D_dash)
                _, _, rewards,_ = D_dash
                rewards_tr_ep.append(rewards)
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
                prev_loss, _, prev_pis = surrogate_loss(D_dashes,policy,value_net,gamma)
            grads = parameters_to_vector(tape.gradient(prev_loss, policy.trainable_variables),policy.trainable_variables)
            prev_loss=tf.identity(prev_loss)
            hvp=HVP(D_dashes,policy,value_net,damping)
            search_step_dir=conjugate_gradients(hvp, grads, rdotr_tol=rdotr_tol,nsteps=nsteps)
            max_length=np.sqrt(2.0 * max_grad_kl / np.dot(search_step_dir, hvp(search_step_dir)))
            full_step=search_step_dir*max_length        
            prev_params = parameters_to_vector(policy.trainable_variables)
            line_search(policy, prev_loss, prev_pis, value_net, gamma, b, D_dashes, full_step, prev_params, max_grad_kl, max_backtracks, zeta)
            
            #evaluation
            if evaluate and t_eval>eval_freq:
                t_eval %= eval_freq
                eval_rewards=[]
                lower = 0.
    
                for eval_ep in range(eval_eps):
                    
                    rand_value = np.around(np.random.uniform(low=lower, high=min(lower+eval_step,1.0)),3)
                    lower += eval_step
                    
                    env.randomize([rand_value]*dr)                    
                    s=env.reset()
                    R = 0
                    done= False
                    while not done:
                        
                        state=tf.expand_dims(tf.convert_to_tensor(s,dtype=tf.float32),0)
                                                                    
                        dist=policy(state)
                        a=tf.squeeze(dist.sample()).numpy()
                        s, r, done, _ = env.step(a)
                                                
                        R+=r
                        
                    eval_rewards.append(R)
                
                eval_rewards_mean=np.mean(np.array(eval_rewards).flatten())
                running_reward=e*eval_rewards_mean+(1-e)*running_reward; #print(np.around(running_reward,2))
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
            total_timesteps.append(t_agent)
            #log iteration results & statistics
            if episode % log_ival == 0:
                log_msg="Rewards Tr: {:.2f}, Rewards Disc: {:.2f}, Rewards Eval: {:.2f}, Total Timesteps: {}".format(reward_ep, scores_disc.mean(), eval_rewards_mean, t_agent)
                if verbose:
                    print(log_msg+f" episode:{episode} \n")
                else:
                    episodes.set_description(desc=log_msg); episodes.refresh()
                    
        # with writer.as_default():
        #     tf.summary.trace_export(name="my_func_trace", step=0,profiler_outdir="logs")

        
        plot_tr_rewards_all.append(plot_tr_rewards)
        plot_eval_rewards_all.append(plot_eval_rewards)
        plot_disc_rewards_all.append(plot_disc_rewards)
        total_timesteps_all.append(total_timesteps)
        
        #%% Results & Plot
        #process results
        plot_tr_rewards_mean = np.stack(plot_tr_rewards_all).mean(0)
        plot_eval_rewards_mean = np.stack(plot_eval_rewards_all).mean(0)
        plot_disc_rewards_mean = np.stack(plot_disc_rewards_all).mean(0)
        total_timesteps_mean = np.stack(total_timesteps_all).mean(0)
        
        plot_tr_rewards_std = np.stack(plot_tr_rewards_all).std(0)
        plot_eval_rewards_std = np.stack(plot_eval_rewards_all).std(0)
        plot_disc_rewards_std = np.stack(plot_disc_rewards_all).std(0)
        
        #save results to df
        df = pd.DataFrame(list(zip(plot_tr_rewards_mean,
                                   plot_tr_rewards_std,
                                   plot_eval_rewards_mean,
                                   plot_eval_rewards_std,
                                   plot_disc_rewards_mean,
                                   plot_disc_rewards_std,
                                   total_timesteps_mean)),
                          columns =['Rewards_Tr_Mean', 'Rewards_Tr_Std', 'Rewards_Eval_Mean', 'Rewards_Eval_Std', 'Rewards_Disc_Mean', 'Rewards_Disc_Std', 'Total_Timesteps'])
        
        df.to_pickle(f"plots/results{common_name}.pkl")
        
        #plot results
        title="Training Rewards"
        plt.figure(figsize=(16,8))
        plt.grid(1)
        plt.plot(plot_tr_rewards_mean)
        plt.fill_between(range(tr_eps), plot_tr_rewards_mean + plot_tr_rewards_std, plot_tr_rewards_mean - plot_tr_rewards_std, alpha=0.2)
        # plt.axhline(y = env.spec.reward_threshold, color = 'r', linestyle = '--',label='Solved')
        plt.title(title)
        plt.savefig(f'plots/tr{common_name}.png')
        
        title="Evaluation Rewards"
        plt.figure(figsize=(16,8))
        plt.grid(1)
        plt.plot(plot_eval_rewards_mean)
        plt.fill_between(range(len(plot_eval_rewards_mean)), plot_eval_rewards_mean + plot_eval_rewards_std, plot_eval_rewards_mean - plot_eval_rewards_std,alpha=0.2)
        # plt.axhline(y = env.spec.reward_threshold, color = 'r', linestyle = '--',label='Solved')
        plt.title(title)
        plt.savefig(f'plots/ts{common_name}.png')
        
        title="Discriminator Rewards"
        plt.figure(figsize=(16,8))
        plt.grid(1)
        plt.plot(plot_disc_rewards_mean)
        plt.fill_between(range(tr_eps),plot_disc_rewards_mean + plot_disc_rewards_std, plot_disc_rewards_mean - plot_disc_rewards_std,alpha=0.2)
        plt.title(title)
        plt.savefig(f'plots/disc{common_name}.png')
            
        eps_step=int((tr_eps-T_svpg_init)/4)
        rand_step=0.1
        region_step=eps_step*T_svpg*n_particles
        df2=pd.DataFrame()
        for dim, regions in enumerate(sampled_regions):
            
            low=env.unwrapped.dimensions[dim].range_min
            high=env.unwrapped.dimensions[dim].range_max
            
            dim_name=env.unwrapped.dimensions[dim].name
            
            # d = decimal.Decimal(str(low))
            # step_exp=d.as_tuple().exponent-1
            # step=10**step_exp
    
            x=np.arange(low,high+rand_step,rand_step)
            
            title=f"Sampled Regions for Randomization Dim = {dim_name} {env.rand} Over Time"
            plt.figure(figsize=(16,8))
            plt.grid(1)
            plt.hist((regions[region_step*0:region_step*1],regions[region_step*1:region_step*2],regions[region_step*2:region_step*3], regions[region_step*3:]), np.arange(min(x),max(x)+2*rand_step,rand_step), histtype='barstacked', label=[f'{eps_step*1} eps',f'{eps_step*2} eps', f'{eps_step*3} eps', f'{eps_step*4} eps'],color=["lightskyblue","blueviolet","hotpink","lightsalmon"])
            plt.xlim(min(x), max(x)+rand_step)
            plt.legend()
            plt.title(title)
            #save results
            plt.savefig(f'plots/sampled_regions_dim_{dim_name}_{env.rand}{common_name}.png')
            df2[f'Sampled_Regions_{dim_name}_{env.rand}'] = list(regions)
        
        df2.to_pickle(f"plots/sampled_regions{common_name}.pkl")
        
        
        #record elapsed time and close envs
        end_time=timeit.default_timer()
        print("Elapsed Time: {:.1f} minutes \n".format((end_time-start_time)/60.0))
        
        env.close()
        env_ref.close()
        env_rand.close()
            
