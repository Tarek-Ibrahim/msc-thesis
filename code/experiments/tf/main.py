# %% TODOs

# TODO: add to comparison: disc between pre and post adaptation maml (instead of post adaptation ref and rand)
# TODO: investigate problem with cuda (and multiprocessing) on start of second seed in active_dr mode

# %% Imports
#general
import numpy as np
import pandas as pd
# import os
# os.environ["OMP_NUM_THREADS"] = "1"
import yaml
from utils import set_seed, progress, parameters_to_vector, PolicyNetwork, ValueNetwork, SVPG, Discriminator, make_vec_envs, collect_rollout_batch, adapt, surrogate_loss, HVP, conjugate_gradients, line_search, map_rewards, ReplayBuffer, DDPG, SAC

#env
import gym
import gym_custom

#visualization
import matplotlib.pyplot as plt

#utils
# import decimal
import timeit
import argparse
from copy import copy, deepcopy

#ML
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


#multiprocessing
import multiprocessing as mp


#%% Main Func
if __name__ == '__main__':
    
    #%% Inputs
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_policy", "-l", action='store_true', help="Whether to start training from saved policy")
    parser.add_argument("--dr_type", "-t", type=str, default="", help="Type of domain randomization (default: baseline)", choices=["","uniform_dr","active_dr","auto_dr","oracle"])
    parser.add_argument("--active_dr_opt", "-o", type=str, default="svpg_a2c", help="Type of randomization space optimization method (in case of active_dr)", choices=["svpg_a2c","svpg_ddpg","ddpg","sac"])
    parser.add_argument("--active_dr_rewarder", "-r", type=str, default="disc", help="Type of reward to randomization space optimization method (in case of active_dr)", choices=["disc","map_neg","map_delta"])
    parser.add_argument("--sac_entropy_tuning_method", "-s", type=str, default="", help="(In case --active_dr_opt=sac) Which entropy tuning method to use, if any", choices=["","learn","anneal"])
    parser.add_argument("--agent_alg", "-a", type=str, default="trpo", help="RL algorithm of the agent", choices=["trpo","ppo"])
    parser.add_argument("--maml", "-M", action='store_true', help="Whether to use MAML algorithm (defaults to base RL algorithm)")
    parser.add_argument("--oracle_rand_value", "-O", type=float, default=0., help="Randomization dimension value used for oracle") #TODO: accomodate dr>1
    parser.add_argument("--mode", "-m", type=int, default=1, help="0: debug mode; 1: run mode")
    parser.add_argument("--verbose", "-v", type=int, default=1, help="0: progress bar; 1: line print")
    parser.add_argument("--env_key", "-e", type=str, default="hopper_friction", help="Environment key")
    args = parser.parse_args()

    
    modes=["debug_mode","run_mode"]
    mode=modes[args.mode]
    
    with open("config.yaml", 'r') as f:
        config_file = yaml.load(f, Loader=yaml.FullLoader)
    
    config=config_file[mode]  
    
    #Env
    env_key=args.env_key
    env_name=config_file["env_names"][env_key]
    n_workers=config["n_workers"]
    
    if args.maml:
        #MAML
        alpha=config["lr_agent"]
        plot_val_rewards_all=[]
    else:
        alpha = None
    
    #TRPO / PPO
    h=config["h_agent"]
    b = config["b_maml"]
    gamma = config["gamma"]
    #TRPO
    max_grad_kl=config["max_grad_kl"]
    max_backtracks=config["max_backtracks"]
    zeta=config["zeta"]
    rdotr_tol=config["rdotr_tol"]
    nsteps=config["nsteps"]
    damping=config["damping"]
    #PPO
    clip=config["clip"]
    lr=config["lr_agent"]
    
    if args.dr_type=="active_dr":
        if args.active_dr_rewarder=="disc":
            #Discriminator
            r_disc_scale = config["r_disc_scale"] 
            h_disc=config["h_disc"] 
            lr_disc=config["lr_disc"] 
            b_disc=config["b_disc"]
        elif "map" in args.active_dr_rewarder:
            r_map_scale = config["r_map_scale"]
            
        plot_disc_rewards_all=[]
        
        #SVPG
        n_particles=config["n_particles"] 
        H_svpg=config["H_svpg"] if "a2c" in args.active_dr_opt else config["H_dr"]
        delta_max=config["delta_max"] 
        T_svpg_init=config["T_svpg_init"]
        T_dr_init=config["T_dr_init"]
        T_svpg=config["T_svpg"] if "a2c" in args.active_dr_opt else config["T_dr"]
        T_rollout=copy(T_svpg)
        lr_svpg=config["lr_dr"] 
        gamma_svpg=config["gamma"]
        h_svpg=config["h_dr"]
        if "svpg" in args.active_dr_opt:
            svpg_kernel_mode=config["svpg_kernel_mode"]
            temp=config["temp_svpg"]
            svpg_base_alg = "a2c" if "a2c" in args.active_dr_opt else "ddpg"
        elif "sac" in args.active_dr_opt:
            temp=config["temp_sac"] if args.sac_entropy_tuning_method else config["temp_min"]
            temp_min=config["temp_min"]
            temp_discount=config["temp_discount"]
            T_temp_init=config["T_temp_init"]
        # if "a2c" not in args.active_dr_opt:
        b_svpg=config["b_dr"]
        epochs_svpg=config["epochs_dr"]
          
    else:
        T_svpg_init=0
        T_rand_rollout=config["T_rand_rollout"]
        T_rollout=copy(T_rand_rollout)
        if args.dr_type=="auto_dr":
            thr_high=config["thr_high"]
            thr_low=config["thr_low"]
            adr_delta=config["delta"]
            m=config["perf_buff_len"]
            pb=0.5
            assert thr_low < thr_high <= n_workers
    
    #Evaluation
    evaluate=config["evaluate"]
    log_ival=config["log_ival"]
    eval_eps=config["eval_eps"]
    e=0.1
    
    #general
    tr_eps=config["tr_eps"]
    file_name=("maml_" if args.maml else "")+args.agent_alg+("_" if args.dr_type else "")+args.dr_type+(f"_{args.oracle_rand_value}" if "oracle" in args.dr_type else "")+(f"_{args.active_dr_rewarder}_{args.active_dr_opt}" if args.dr_type=="active_dr" else "")+(f"_{args.sac_entropy_tuning_method}" if args.dr_type=="active_dr" and args.active_dr_opt=="sac" and args.sac_entropy_tuning_method else "")
    common_name = "_"+file_name+"_"+env_key
    plots_tr_dir=config_file["plots_tr_dir"]
    models_dir=config_file["models_dir"]
    verbose=args.verbose
    add_noise=True if args.dr_type else False
    figsize=tuple(config_file["figsize"])
    
    #Seed
    seeds=config["seeds"]
    
    plot_tr_rewards_all=[]
    plot_eval_rewards_all=[]
    total_timesteps_all=[]
    best_reward=-1e6
    best_running_reward=-1e6
    
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
        thr_r=env.spec.reward_threshold if args.mode else -1e6
        
        if args.dr_type=="active_dr":
            env_ref=make_vec_envs(env_name, seed, n_workers, ds, da, queue, lock)
            if args.active_dr_rewarder=="disc":
                discriminator=Discriminator(ds, da, h_disc, b_disc, r_disc_scale, lr_disc)
            elif "map" in args.active_dr_rewarder:
                default_solved=False
            if "svpg" in args.active_dr_opt:
                svpg = SVPG(n_particles, dr, h_svpg, delta_max, T_svpg, H_svpg, temp, lr_svpg, svpg_kernel_mode, gamma_svpg, epochs=epochs_svpg, batch_size=b_svpg, T_init=T_dr_init,base_alg=svpg_base_alg)
            elif "ddpg" in args.active_dr_opt:
                svpg=DDPG(dr, dr, h_svpg, h_svpg, lr_svpg, b_svpg, epochs_svpg, a_max=delta_max, gamma=gamma_svpg)
            elif "sac" in args.active_dr_opt:
                svpg=SAC(dr, dr,h_svpg, lr_svpg,b_svpg,epochs_svpg,temp,delta_max,args.sac_entropy_tuning_method, T_temp_init, gamma=gamma_svpg, alpha_min=temp_min, alpha_discount=temp_discount)
            plot_disc_rewards=[]
            sampled_regions = [[] for _ in range(dr)]
            RB=ReplayBuffer() if "a2c" not in args.active_dr_opt else None
            RB_samples=None
        elif args.dr_type=="auto_dr":
            D_autodr={str(env.unwrapped.dimensions[dim].name):{"low":[],"high":[]} for dim in range(dr)} #Performance buffers/queues
            phis={str(env.unwrapped.dimensions[dim].name):{"low":env.unwrapped.dimensions[dim].default_value,"high":env.unwrapped.dimensions[dim].default_value} for dim in range(dr)}
            sampled_regions = [[] for _ in range(dr)]
            phis_plot={str(env.unwrapped.dimensions[dim].name):{"low":[env.unwrapped.dimensions[dim].default_value],"high":[env.unwrapped.dimensions[dim].default_value]} for dim in range(dr)}
            phis_plot_all={str(env.unwrapped.dimensions[dim].name):{"low":{"data": [], "mean": 0., "std": 1.},"high":{"data": [], "mean": 0., "std": 1.}} for dim in range(dr)}
            lambda_vec=np.zeros(dr)
            # p_bar=0.
            bounds_reached={str(env.unwrapped.dimensions[dim].name):{"low":0,"high":0} for dim in range(dr)}
            
        env_rand=make_vec_envs(env_name, seed, n_workers, ds, da, queue, lock)

        #models
        in_size=ds
        out_size=da
        policy = PolicyNetwork(in_size,h,out_size) #dynamics model
        if args.agent_alg=="ppo": optimizer=tf.keras.optimizers.Adam(learning_rate=lr)
        if args.load_policy:
            policy.load_weights(f"{models_dir}model{common_name}")
        value_net=ValueNetwork(in_size,gamma)
        
        #results
        plot_tr_rewards=[]
        plot_eval_rewards=[]
        total_timesteps=[]
        if args.maml: plot_val_rewards=[]
        t_agent = 0
        
        #evaluation
        eval_rewards_mean=0. #best_reward
        eval_freq = T_env * n_workers if args.mode else 1
        t_eval=0 #agent timesteps since eval
        eval_step = 1.0 / eval_eps
        running_reward = 0.
        
        #%% Implementation
        episodes=progress(tr_eps) if not verbose else range(tr_eps)
        for episode in episodes:
            
            if args.dr_type:
                #rollout svpg particles (each particle represents a different rand env and each timestep in that rollout represents different values for its randomization params)
                if args.dr_type=="active_dr":
                    if "map" in args.active_dr_rewarder and not default_solved:
                        simulation_instances = -2 * np.ones((n_workers,T_svpg,dr))
                    elif episode >= T_svpg_init:
                        simulation_instances, RB_samples = svpg.step(RB) if "svpg" in args.active_dr_opt else svpg.step(RB,T_svpg,T_dr_init,H_svpg)
                        simulation_instances = np.clip(simulation_instances,0,1)
                        if "svpg" not in args.active_dr_opt: simulation_instances = np.tile(simulation_instances,(n_particles,1,1))
                    else:
                        simulation_instances = -1 * np.ones((n_particles,T_svpg,dr))
                        
                    rewards_disc = np.zeros(simulation_instances.shape[:2])
                    scores_disc=np.zeros(simulation_instances.shape[:2])
                    D_dashes_ref, D_dashes_rand=[], []
                    if "map" in args.active_dr_rewarder: rewards_ref=[]
                else:
                    simulation_instances = -1 * np.ones((n_workers,T_rand_rollout,dr))
                            
                # Reshape to work with vectorized environments
                simulation_instances = np.transpose(simulation_instances, (1, 0, 2))
            
            #create empty storages
            rewards_tr_ep=[]
            D_dashes = []
            lambda_norms=[]
            if args.maml:
                rewards_val_ep = []
                Ds = []
            else:
                Ds = None
                
            #inner/adaptation loop
            for t_rollout in range(T_rollout):
                
                if args.dr_type:
                    #randomize rand envs (with svpg particle [randomization parameter] values [at the current svpg timestep]) #!!!: this only works if transitions within the same rollout is collected in the same environment (i.e. by the same [env] worker) #that's why we choose: n_particles=b(rollout batch size)=n_workers/n_envs
                    if args.dr_type=="auto_dr":
                        
                        lambda_norm = np.zeros_like(lambda_vec)
                        for i in range(len(lambda_vec)):
                            dim_name=env.unwrapped.dimensions[i].name
                            low=env.unwrapped.dimensions[i].range_min
                            high=env.unwrapped.dimensions[i].range_max
                            lambda_vec[i]=np.random.uniform(phis[dim_name]["low"],phis[dim_name]["high"])
                            lambda_norm[i]=(lambda_vec[i]-low)/(high-low)
                        
                        pb_ep=np.random.rand()
                        if pb_ep < pb:
                            i = np.random.randint(0,dr)
                            if np.random.rand() < 0.5:
                                boundary="low"
                                other_boundary="high"
                            else:
                                boundary="high"
                                other_boundary="low"
                            dim_name=env.unwrapped.dimensions[i].name
                            lambda_vec[i]=phis[dim_name][boundary]
                            
                            low=env.unwrapped.dimensions[i].range_min
                            high=env.unwrapped.dimensions[i].range_max
                            default=env.unwrapped.dimensions[i].default_value
                            lambda_norm[i] = (lambda_vec[i]-low)/(high-low)
                        
                        simulation_instances[t_rollout]=np.tile(lambda_norm,(n_workers,1))
                        lambda_norms.append(lambda_norm)
                    elif args.dr_type=="oracle":
                        simulation_instances[t_rollout]=np.ones((n_workers,dr))*args.oracle_rand_value
                            
                    env_rand.randomize(simulation_instances[t_rollout])
                
                if args.maml:
                    #collect pre-adaptation rollout batch in rand envs (one rollout for each svpg particle)
                    D,_=collect_rollout_batch(env_rand, ds, da, policy, T_env, b, n_workers, queue,add_noise=add_noise)
                    Ds.append(D)
                    _, _, rewards,_ = D
                    rewards_tr_ep.append(rewards)
                    t_agent += rewards.size
                
                    #adapt agent [meta-]parameters (via VPG w/ baseline)
                    theta_dash=adapt(D,value_net,policy,alpha)
                else:
                    theta_dash=None
                
                if args.dr_type=="active_dr":
                    #collect post-adaptaion rollout batch in ref envs
                    D_ref,ref_traj=collect_rollout_batch(env_ref, ds, da, policy, T_env, b, n_workers, queue, params=theta_dash)
                    D_dashes_ref.append(ref_traj)
                    if "map" in args.active_dr_rewarder:
                        _, _, rewards,_ = D_ref
                        rewards_ref.append(rewards)
                
                #collect post-adaptation rollout batch in rand envs
                D_dash,rand_traj=collect_rollout_batch(env_rand, ds, da, policy, T_env, b, n_workers, queue, params=theta_dash,add_noise=add_noise)
                if args.dr_type=="active_dr": D_dashes_rand.append(rand_traj)
                D_dashes.append(D_dash)
                _, _, rewards,_ = D_dash
                t_agent += rewards.size
                t_eval += rewards.size
                if args.maml:
                    rewards_val_ep.append(rewards)
                else:
                    rewards_tr_ep.append(rewards)
                    
                if args.dr_type=="auto_dr" and pb_ep < pb:
                    p=(rewards.sum(0)>=thr_r).astype(int).sum()
                    D_autodr[dim_name][boundary].append(p)
                    
                    if len(D_autodr[dim_name][boundary])>=m:
                        p_bar=np.mean(D_autodr[dim_name][boundary])
                        D_autodr[dim_name][boundary]=[]
                        if p_bar>=thr_high: #expand bounds
                            if boundary=="low": 
                                phis[dim_name][boundary]=np.maximum(phis[dim_name][boundary] - adr_delta, low) #decrease lower bound
                                if phis[dim_name]["low"] <= low:
                                    bounds_reached[dim_name]["low"]+=1
                            else:
                                phis[dim_name][boundary]=np.minimum(phis[dim_name][boundary] + adr_delta, high) #increase upper bound
                                if phis[dim_name]["high"] >= high:
                                    bounds_reached[dim_name]["high"]+=1
                            phis_plot[dim_name][boundary].append(phis[dim_name][boundary])
                            phis_plot[dim_name][other_boundary].append(phis[dim_name][other_boundary])
    
                        elif p_bar<=thr_low: #tighten bounds
                            if boundary=="high":    
                                phis[dim_name][boundary]=np.maximum(phis[dim_name][boundary] - adr_delta, default) #decrease upper bound
                            else:
                                phis[dim_name][boundary]=np.minimum(phis[dim_name][boundary] + adr_delta, default) #increase lower bound
                            phis_plot[dim_name][boundary].append(phis[dim_name][boundary])
                            phis_plot[dim_name][other_boundary].append(phis[dim_name][other_boundary])
            
            if args.dr_type=="active_dr":
                #active_dr updates
                for t, (ref_traj, rand_traj) in enumerate(zip(D_dashes_ref,D_dashes_rand)):
                    T_disc_eps=0 #agent timesteps in the current iteration/episode
        
                    #calculate discriminator reward
                    for i in range(n_particles):
                        if args.active_dr_rewarder=="disc":
                            T_disc_eps += len(rand_traj[i])
                            r_disc, score_disc = discriminator.calculate_rewards(rand_traj[i])
                        elif "map" in args.active_dr_rewarder:
                            randomized_rewards = deepcopy(rewards_val_ep) if args.maml else deepcopy(rewards_tr_ep)
                            randomized_reward = np.sum(randomized_rewards[t][:,i])
                            reference_reward = np.sum(rewards_ref[t][:,i])
                            r_disc = map_rewards(args.active_dr_rewarder,r_map_scale,randomized_reward,reference_reward)
                            score_disc = deepcopy(r_disc)
                        
                        rewards_disc[i][t]= r_disc
                        scores_disc[i][t]=score_disc
                        
                        if args.active_dr_opt=="svpg_ddpg" and RB_samples is not None:
                            RB_samples[i][t][3]=r_disc
                            RB.add(RB_samples[i][t])
                            
                    if "svpg" not in args.active_dr_opt and RB_samples is not None:
                        RB_samples[t][3]=rewards_disc[:,t].mean()
                        RB.add(RB_samples[t])
                    
                    if args.active_dr_rewarder=="disc":
                        #train discriminator
                        flattened_rand = np.concatenate(rand_traj)
                        flattened_ref = np.concatenate(ref_traj)
                        discriminator.train(ref_traj=flattened_ref, rand_traj=flattened_rand, eps=T_disc_eps)
                
                plot_disc_rewards.append(scores_disc.mean())
                
                #update svpg particles' params (ie. train their policies)
                if episode >= T_svpg_init and ("map" not in args.active_dr_rewarder and ("a2c" in args.active_dr_opt or len(RB.storage) > T_dr_init) or "map" in args.active_dr_rewarder and default_solved):
                    # arg= deepcopy(rewards_disc) if "a2c" in args.active_dr_opt else deepcopy(RB)
                    svpg.train(rewards_disc if "a2c" in args.active_dr_opt else RB)
                    
                    #log sampled regions only once svpg particles start training (i.e. once active_dr starts)
                    for dim in range(dr):
                        low=env.unwrapped.dimensions[dim].range_min
                        high=env.unwrapped.dimensions[dim].range_max
                        scaled_instances=low + (high-low) * simulation_instances[:, :, dim]
                        sampled_regions[dim]=np.concatenate([sampled_regions[dim],scaled_instances.flatten()])
                
                
                if "map" in args.active_dr_rewarder and not default_solved:
                    default_solved = (randomized_reward >= thr_r)
            
            elif args.dr_type=="auto_dr":
                
                for dim in range(dr):
                    low=env.unwrapped.dimensions[dim].range_min
                    high=env.unwrapped.dimensions[dim].range_max
                    scaled_instances=low + (high-low) * np.array(lambda_norms)
                    sampled_regions[dim]=np.concatenate([sampled_regions[dim],scaled_instances.flatten()])
                    
                #check stop condition (once you've covered the whole range of all randomized dims)
                n_covered_dims=0
                for dim in range(dr):
                    dim_name=env.unwrapped.dimensions[i].name
                    if bounds_reached[dim_name]["low"] > 1 and bounds_reached[dim_name]["high"] > 1:
                        n_covered_dims += 1
                
                if n_covered_dims == dr:
                    break
            
            #outer loop: update meta-params (via: TRPO) #!!!: since MAML uses TRPO it is on-policy, so care should be taken that order of associated transitions is preserved
            if args.agent_alg=="trpo":
                with tf.GradientTape() as tape:
                    prev_loss, _, prev_pis = surrogate_loss(D_dashes,policy,value_net,gamma,alpha=alpha,Ds=Ds,alg="trpo")
                grads = parameters_to_vector(tape.gradient(prev_loss, policy.trainable_variables),policy.trainable_variables)
                prev_loss=tf.identity(prev_loss)
                hvp=HVP(D_dashes,policy,value_net,damping,alpha=alpha,Ds=Ds)
                search_step_dir=conjugate_gradients(hvp, grads, rdotr_tol=rdotr_tol,nsteps=nsteps)
                max_length=np.sqrt(2.0 * max_grad_kl / np.dot(search_step_dir, hvp(search_step_dir)))
                full_step=search_step_dir*max_length        
                prev_params = parameters_to_vector(policy.trainable_variables)
                line_search(policy, prev_loss, prev_pis, value_net, gamma, b, D_dashes, full_step, prev_params, max_grad_kl, max_backtracks, zeta, alpha=alpha, Ds=Ds)
            elif args.agent_alg=="ppo":
                with tf.GradientTape() as tape:
                    loss,_,_ = surrogate_loss(D_dashes,policy,value_net,gamma,clip=clip,alpha=alpha,Ds=Ds,alg="ppo")
                gradients = tape.gradient(loss, policy.trainable_variables) #calculate gradient
                optimizer.apply_gradients(zip(gradients, policy.trainable_variables)) #backpropagate
            
            #evaluation
            if evaluate and t_eval>eval_freq:
                t_eval %= eval_freq
                eval_rewards=[]
                lower = 0.
    
                for _ in range(eval_eps):
                    if args.dr_type:
                        rand_value = args.oracle_rand_value if "oracle" in args.dr_type else np.around(np.random.uniform(low=lower, high=min(lower+eval_step,1.0)),3)
                        lower += eval_step
                        
                        env.randomize([rand_value]*dr)
                    
                    s=env.reset()
                    
                    if args.maml:
                        state=tf.expand_dims(tf.convert_to_tensor(s,dtype=tf.float32),0)
                        dist=policy(state,params=None)
                        a=tf.squeeze(dist.sample()).numpy()
                        s, r, done, _ = env.step(a)
                        R = r
                    else:
                        R=0
                        done=False
                    
                    while not done:
                        
                        state=tf.expand_dims(tf.convert_to_tensor(s,dtype=tf.float32),0)
                        
                        if args.maml:
                            states=tf.expand_dims(state,0)
                            actions=tf.expand_dims(tf.expand_dims(tf.convert_to_tensor(a),0),0)
                            rewards=tf.expand_dims(tf.expand_dims(tf.convert_to_tensor(np.array(r),dtype=np.float32),0),0)
                            masks=tf.expand_dims(tf.expand_dims(tf.convert_to_tensor(np.array(1.0),dtype=np.float32),0),0)
                            D=[states, actions, rewards, masks]
                                                
                            theta_dash=adapt(D,value_net,policy,alpha)
                        else:
                            theta_dash=None
                        
                        dist=policy(state,params=theta_dash)
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
            plot_tr_rewards.append(reward_ep)
            if args.maml:
                reward_val=(tf.math.reduce_mean(tf.stack([tf.math.reduce_mean(tf.math.reduce_sum(rewards, axis=0)) for rewards in rewards_val_ep], axis=0))).numpy()
                plot_val_rewards.append(reward_val)
            
            #save best running model [params]
            if eval_rewards_mean>best_reward: 
                best_reward=eval_rewards_mean
                policy.save_weights(f"{models_dir}model{common_name}")
                
            # #save best running model [params]
            # if running_reward>best_running_reward: 
            #     best_running_reward=running_reward
            #     policy.save_weights(f"saved_models/model_running{common_name}")
            
            #log iteration results & statistics
            total_timesteps.append(t_agent)
            if episode % log_ival == 0:
                log_msg="Rewards Tr: {:.2f}, ".format(reward_ep)
                log_msg = log_msg + ("Rewards Val: {:.2f}, ".format(reward_val) if args.maml else "") + ("Rewards Disc: {:.2f}, ".format(scores_disc.mean()) if args.dr_type=="active_dr" else "")
                log_msg = log_msg + "Rewards Eval: {:.2f}, Total Timesteps: {}".format(eval_rewards_mean, t_agent)
                if verbose:
                    print(log_msg+f" episode:{episode} \n")
                else:
                    episodes.set_description(desc=log_msg); episodes.refresh()
                    
        
        plot_tr_rewards_all.append(plot_tr_rewards)
        plot_tr_rewards_mean = np.stack(plot_tr_rewards_all).mean(0)
        plot_tr_rewards_std= np.stack(plot_tr_rewards_all).std(0)
        
        plot_eval_rewards_all.append(plot_eval_rewards)
        plot_eval_rewards_mean = np.stack(plot_eval_rewards_all).mean(0)
        plot_eval_rewards_std = np.stack(plot_eval_rewards_all).std(0)
        
        total_timesteps_all.append(total_timesteps)
        total_timesteps_mean = np.stack(total_timesteps_all).mean(0)
        
        if args.maml:
            plot_val_rewards_all.append(plot_val_rewards)
            plot_val_rewards_mean = np.stack(plot_val_rewards_all).mean(0)
            plot_val_rewards_std = np.stack(plot_val_rewards_all).std(0)
            
        if args.dr_type=="active_dr": 
            plot_disc_rewards_all.append(plot_disc_rewards)
            plot_disc_rewards_mean = np.stack(plot_disc_rewards_all).mean(0)
            plot_disc_rewards_std = np.stack(plot_disc_rewards_all).std(0)
        elif args.dr_type=="auto_dr":
            for dim in range(dr):
                dim_name=env.unwrapped.dimensions[dim].name
                for boundary in ["low", "high"]:
                    phis_plot_all[dim_name][boundary]["data"].append(phis_plot[dim_name][boundary])
                    phis_plot_all[dim_name][boundary]["mean"]=np.stack(phis_plot_all[dim_name][boundary]["data"]).mean(0)
                    phis_plot_all[dim_name][boundary]["std"]=np.stack(phis_plot_all[dim_name][boundary]["data"]).std(0)
        
        #%% Results & Plot
        #save results to df
        df = pd.DataFrame(list(zip(plot_tr_rewards_mean,
                                   plot_tr_rewards_std,
                                   plot_eval_rewards_mean,
                                   plot_eval_rewards_std,
                                   total_timesteps_mean)),
                          columns =['Rewards_Tr_Mean', 'Rewards_Tr_Std', 'Rewards_Eval_Mean', 'Rewards_Eval_Std', 'Total_Timesteps'])
        if args.maml:
            df['Rewards_Val_Mean']=plot_val_rewards_mean
            df['Rewards_Val_Std']=plot_val_rewards_std
        
        if args.dr_type=="active_dr":
            df['Rewards_Disc_Mean']=plot_disc_rewards_mean
            df['Rewards_Disc_Std']=plot_disc_rewards_std
        
        df.to_pickle(f"{plots_tr_dir}results{common_name}.pkl")
        
        #plot results
        title="Meta-Training Training Rewards" if args.maml else "Training Rewards"
        plt.figure(figsize=figsize)
        plt.grid(1)
        plt.plot(plot_tr_rewards_mean)
        plt.fill_between(range(tr_eps), plot_tr_rewards_mean + plot_tr_rewards_std, plot_tr_rewards_mean - plot_tr_rewards_std,alpha=0.2)
        # plt.axhline(y = env.spec.reward_threshold, color = 'r', linestyle = '--',label='Solved')
        plt.xlabel("Episodes")
        plt.ylabel("Rewards")
        plt.title(title)
        plt.savefig(f'{plots_tr_dir}tr{common_name}.png')
        
        if args.maml:
            title="Meta-Training Testing Rewards"
            plt.figure(figsize=figsize)
            plt.grid(1)
            plt.plot(plot_val_rewards_mean)
            plt.fill_between(range(tr_eps), plot_val_rewards_mean + plot_val_rewards_std, plot_val_rewards_mean - plot_val_rewards_std,alpha=0.2)
            # plt.axhline(y = env.spec.reward_threshold, color = 'r', linestyle = '--',label='Solved')
            plt.xlabel("Episodes")
            plt.ylabel("Rewards")
            plt.title(title)
            plt.savefig(f'{plots_tr_dir}val{common_name}.png')
        
        title="Meta-Testing Rewards" if args.maml else "Evaluation Rewards"
        plt.figure(figsize=figsize)
        plt.grid(1)
        plt.plot(plot_eval_rewards_mean)
        plt.fill_between(range(len(plot_eval_rewards_mean)), plot_eval_rewards_mean + plot_eval_rewards_std, plot_eval_rewards_mean - plot_eval_rewards_std,alpha=0.2)
        # plt.axhline(y = env.spec.reward_threshold, color = 'r', linestyle = '--',label='Solved')
        plt.xlabel("Episodes")
        plt.ylabel("Rewards")
        plt.title(title)
        plt.savefig(f'{plots_tr_dir}ts{common_name}.png')
        
        if "active" in args.dr_type or "auto" in args.dr_type:
            if "active" in args.dr_type:
                title="Discriminator Rewards"
                plt.figure(figsize=figsize)
                plt.grid(1)
                plt.plot(plot_disc_rewards_mean)
                plt.fill_between(range(tr_eps), plot_disc_rewards_mean + plot_disc_rewards_std, plot_disc_rewards_mean - plot_disc_rewards_std,alpha=0.2)
                plt.xlabel("Episodes")
                plt.ylabel("Rewards")
                plt.title(title)
                plt.savefig(f'{plots_tr_dir}disc{common_name}.png')
            elif "auto" in args.dr_type:
                for dim in range(dr):
                    title=f"Boundaries Change for Randomization Dim = {dim_name} {env.rand} at Episode = {episode}"
                    plt.figure(figsize=figsize)
                    plt.grid(1)
                    plt.plot(phis_plot_all[dim_name]["low"]["mean"],label="Lower Bound")
                    plt.fill_between(range(len(phis_plot_all[dim_name]["low"]["mean"])), phis_plot_all[dim_name]["low"]["mean"] + phis_plot_all[dim_name]["low"]["std"], phis_plot_all[dim_name]["low"]["mean"] - phis_plot_all[dim_name]["low"]["std"],alpha=0.2)
                    plt.plot(phis_plot_all[dim_name]["high"]["mean"],label="Upper Bound")
                    plt.fill_between(range(len(phis_plot_all[dim_name]["high"]["mean"])), phis_plot_all[dim_name]["high"]["mean"] + phis_plot_all[dim_name]["high"]["std"], phis_plot_all[dim_name]["high"]["mean"] - phis_plot_all[dim_name]["high"]["std"],alpha=0.2)
                    plt.xlabel("Episodes")
                    plt.ylabel("Randomized Parameter Boundary Values")
                    plt.title(title)
                    plt.legend()
                    plt.savefig(f'{plots_tr_dir}boundaries_change_dim_{dim_name}_{env.rand}{common_name}.png')
                
            eps_step=int((tr_eps-T_svpg_init)/4)
            rand_step = 0.1
            region_step=eps_step*T_rollout*n_workers
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
                plt.figure(figsize=figsize)
                plt.grid(1)
                plt.hist((regions[region_step*0:region_step*1],regions[region_step*1:region_step*2],regions[region_step*2:region_step*3], regions[region_step*3:]), np.arange(min(x),max(x)+2*rand_step,rand_step), histtype='barstacked', label=[f'{eps_step*1} eps',f'{eps_step*2} eps', f'{eps_step*3} eps', f'{eps_step*4} eps'],color=["lightskyblue","blueviolet","hotpink","lightsalmon"])
                plt.xlim(min(x), max(x)+rand_step)
                plt.xlabel("Randomization Range")
                plt.ylabel("Number of Samples (Sampling Frequency)")
                plt.legend()
                plt.title(title)
                #save results
                plt.savefig(f'{plots_tr_dir}sampled_regions_dim_{dim_name}_{env.rand}{common_name}.png')
                df2[f'Sampled_Regions_{dim_name}_{env.rand}'] = list(regions)
            
            df2.to_pickle(f"{plots_tr_dir}sampled_regions{common_name}.pkl")
        

        #record elapsed time and close envs
        end_time=timeit.default_timer()
        print("Elapsed Time: {:.1f} minutes \n".format((end_time-start_time)/60.0))
        
        env.close()
        if args.dr_type=="active_dr": env_ref.close()
        env_rand.close()
        