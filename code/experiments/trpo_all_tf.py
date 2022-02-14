# %% TODOs


# %% Imports
#general
import numpy as np
import pandas as pd
# import os
# os.environ["OMP_NUM_THREADS"] = "1"
import yaml
from common import set_seed, progress, parameters_to_vector, vector_to_parameters, PolicyNetwork, ValueNetwork, SVPG, Discriminator, make_vec_envs, collect_rollout_batch, weighted_mean, detach_dist, compute_advantages, adapt

#env
import gym
import gym_custom

#visualization
import matplotlib.pyplot as plt

#utils
import decimal
import timeit
import argparse

#ML
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


#multiprocessing
import multiprocessing as mp


#%% TRPO

def conjugate_gradients(Avp_f, b, rdotr_tol=1e-10, nsteps=10):
    """
    nsteps = max_iterations
    rdotr = residual
    """
    x = np.zeros_like(b)
    r = b.numpy().copy()
    p = b.numpy().copy() 
    rdotr = r.dot(r) 
    
    for i in range(nsteps):
        Avp = Avp_f(p).numpy()
        alpha = rdotr / p.dot(Avp) 
        x += alpha * p
        r -= alpha * Avp
        new_rdotr = r.dot(r)
        beta = new_rdotr / rdotr
        p = r + beta * p
        rdotr = new_rdotr
        if rdotr < rdotr_tol:
            break
        
    return x


def surrogate_loss(D_dashes,policy,value_net,gamma,alpha=None,prev_pis=None, Ds=None):
    
    kls, losses, pis =[], [], []
    if Ds is None:
        Ds = [None] * len(D_dashes)
        
    if prev_pis is None:
        prev_pis = [None] * len(D_dashes)
        
    for D, D_dash, prev_pi in zip(Ds,D_dashes,prev_pis):

        states, actions, rewards, masks = D_dash
        
        if D is not None:
            theta_dash=adapt(D,value_net,policy,alpha)
            pi=policy(states,params=theta_dash)
        else:
            value_net.fit_params(states,rewards,masks)
            pi=policy(states)
        
        pis.append(detach_dist(pi))
        
        if prev_pi is None:
            prev_pi = detach_dist(pi)
        
        advantages=compute_advantages(states,rewards,value_net,gamma,masks)
        
        ratio=pi.log_prob(actions)-prev_pi.log_prob(actions)
        if len(ratio.shape) > 2:
            ratio = tf.reduce_sum(ratio, axis=2)
        loss = - weighted_mean(tf.exp(ratio)*advantages,axis=0,weights=masks)
        losses.append(loss)
        
        if len(actions.shape) > 2:
            masks = tf.expand_dims(masks, axis=2)
        #???: which version is correct?
        # kl=tf.math.reduce_mean(pi.kl_divergence(prev_pi))
        # kl=tf.math.reduce_mean(prev_pi.kl_divergence(pi))
        kl= weighted_mean(prev_pi.kl_divergence(pi),axis=0,weights=masks)
        
        kls.append(kl)
    
    prev_loss=tf.math.reduce_mean(tf.stack(losses, axis=0))
    kl=tf.math.reduce_mean(tf.stack(kls, axis=0)) #???: why is it always zero?
    
    return prev_loss, kl, pis


def line_search(policy, prev_loss, prev_pis, value_net, gamma, b, D_dashes, step, prev_params, max_grad_kl, max_backtracks=10, zeta=0.5, alpha=None, Ds=None):
    """backtracking line search"""
    
    for step_frac in [zeta**x for x in range(max_backtracks)]:
        vector_to_parameters(prev_params - step_frac * step, policy.trainable_variables)
        
        loss, kl, _ = surrogate_loss(D_dashes,policy,value_net,gamma,alpha=alpha,prev_pis=prev_pis,Ds=Ds)
        
        #check improvement
        actual_improve = loss - prev_loss
        if not np.isfinite(loss):
            raise RuntimeError('NANs/Infs encountered in line search')
        if (actual_improve.numpy() < 0.0) and (kl.numpy() < max_grad_kl):
            break
    else:
        vector_to_parameters(prev_params, policy.trainable_variables)


def kl_div(D_dashes,value_net,policy,alpha=None,Ds=None):
    kls=[]
    
    if Ds is None:
        Ds = [None] * len(D_dashes)
                          
    for D, D_dash in zip(Ds,D_dashes):
        
        states, actions, _, masks = D_dash
        
        if D is not None:
            theta_dash=adapt(D,value_net,policy,alpha)
        else:
            theta_dash=None
        
        pi=policy(states,params=theta_dash)
        
        prev_pi = detach_dist(pi)
        
        if len(actions.shape) > 2:
            masks = tf.expand_dims(masks, axis=2)
        #???: which version is correct?
        # kl=tf.math.reduce_mean(pi.kl_divergence(prev_pi))
        # kl=tf.math.reduce_mean(prev_pi.kl_divergence(pi))
        kl= weighted_mean(prev_pi.kl_divergence(pi),axis=0,weights=masks)
        
        kls.append(kl)
    
    return tf.math.reduce_mean(tf.stack(kls, axis=0))


def HVP(D_dashes,policy,value_net,damping,alpha=None,Ds=None):
    def _HVP(v):
        with tf.GradientTape() as outer_tape:
             with tf.GradientTape() as inner_tape:
                 kl = kl_div(D_dashes,value_net,policy,alpha,Ds)
             grad_kl=parameters_to_vector(inner_tape.gradient(kl,policy.trainable_variables),policy.trainable_variables)
             dot=tf.tensordot(grad_kl, v, axes=1)
        return parameters_to_vector(outer_tape.gradient(dot, policy.trainable_variables),policy.trainable_variables) + damping * v            
    return _HVP

#%% Main Func
if __name__ == '__main__':
    
    #%% Inputs
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_policy", "-l", type=bool, default=False, help="Whether to start training from saved policy")
    parser.add_argument("--dr_type", "-r", type=str, default="", help="Type of domain randomization", choices=["","udr","adr"])
    parser.add_argument("--maml", "-M", type=bool, default=False, help="Whether to use MAML algorithm (defaults to base RL algorithm)")
    parser.add_argument("--mode", "-m", type=int, default=1, help="0: debug mode; 1: run mode")
    args = parser.parse_args()

    
    modes=["debug_mode","run_mode"]
    mode=modes[args.mode]
    
    with open("config.yaml", 'r') as f:
        config_file = yaml.load(f, Loader=yaml.FullLoader)
    
    config=config_file[mode]
    
    if args.maml:
        #MAML
        alpha=config["lr_maml"]
        plot_val_rewards_all=[]
    else:
        alpha = None
    
    #TRPO
    h=config["h_maml"]
    b = config["b_maml"]
    gamma = config["gamma_maml"]
    max_grad_kl=config["max_grad_kl"]
    max_backtracks=config["max_backtracks"]
    zeta=config["zeta"]
    rdotr_tol=config["rdotr_tol"]
    nsteps=config["nsteps"]
    damping=config["damping"]
        
    if args.dr_type=="adr":
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
        T_rollout=T_svpg
        lr_svpg=config["lr_svpg"] 
        gamma_svpg=config["gamma_svpg"]
        h_svpg=config["h_svpg"]
        plot_disc_rewards_all=[]
        
    else:
        T_rand_rollout=config["T_rand_rollout"]
        T_rollout=T_rand_rollout
    
    #Env
    env_key="hopper_friction"
    env_name=config_file["env_names"][env_key] #config["env_name"]
    n_workers=config["n_workers"] 
    
    #Evaluation
    evaluate=config["evaluate"]
    log_ival=config["log_ival"]
    eval_eps=config["eval_eps"]
    
    #general
    tr_eps=config["tr_eps"]
    file_name=("maml_" if args.maml else "")+"trpo"+("_" if args.dr_type else "")+args.dr_type
    common_name = "_"+file_name+"_"+env_key
    verbose=config["verbose"]
    add_noise=True if args.dr_type else False
    
    #Seed
    seeds=config["seeds"]
    
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
        
        if args.dr_type=="adr":
            env_ref=make_vec_envs(env_name, seed, n_workers, ds, da, queue, lock)
            discriminator=Discriminator(ds, da, h_disc, b_disc, r_disc_scale, lr_disc)
            svpg = SVPG(n_particles, dr, h_svpg, delta_max, T_svpg, T_svpg_reset, temp, kld_coeff, lr_svpg, gamma_svpg, type_particles)
            plot_disc_rewards=[]
            sampled_regions = [[] for _ in range(dr)]
        env_rand=make_vec_envs(env_name, seed, n_workers, ds, da, queue, lock)

        #models
        in_size=ds
        out_size=da
        policy = PolicyNetwork(in_size,h,out_size) #dynamics model
        if args.load_policy:
            policy.load_weights(f"saved_models/model{common_name}")
        value_net=ValueNetwork(in_size,gamma)
        
        #results
        plot_tr_rewards=[]
        plot_eval_rewards=[]
        total_timesteps=[]
        if args.maml: plot_val_rewards=[]
        t_agent = 0
        
        #evaluation
        eval_rewards_mean=0 #best_reward
        eval_freq = T_env * n_workers
        t_eval=0 #agent timesteps since eval 
        
        #%% Implementation
        episodes=progress(tr_eps) if not verbose else range(tr_eps)
        for episode in episodes:
            
            if args.dr_type:
                #rollout svpg particles (each particle represents a different rand env and each timestep in that rollout represents different values for its randomization params)
                if args.dr_type=="adr":
                    simulation_instances = svpg.step() if episode >= T_svpg_init else -1 * np.ones((n_particles,T_svpg,dr))
                    rewards_disc = np.zeros(simulation_instances.shape[:2])
                    scores_disc=np.zeros(simulation_instances.shape[:2])
                    D_dashes_ref, D_dashes_rand=[], []
                else:
                    simulation_instances = -1 * np.ones((n_workers,T_rand_rollout,dr))
                # Reshape to work with vectorized environments
                simulation_instances = np.transpose(simulation_instances, (1, 0, 2))
            
            #create empty storages
            rewards_tr_ep=[]
            D_dashes = []
            if args.maml:
                rewards_val_ep = []
                Ds = []
            else:
                Ds = None
                
            #inner/adaptation loop
            for t_rollout in range(T_rollout):
                
                if args.dr_type:
                    #randomize rand envs (with svpg particle [randomization parameter] values [at the current svpg timestep]) #!!!: this only works if transitions within the same rollout is collected in the same environment (i.e. by the same [env] worker) #that's why we choose: n_particles=b(rollout batch size)=n_workers/n_envs
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
                
                if args.dr_type=="adr":
                    #collect post-adaptaion rollout batch in ref envs
                    _,ref_traj=collect_rollout_batch(env_ref, ds, da, policy, T_env, b, n_workers, queue, params=theta_dash)
                    D_dashes_ref.append(ref_traj)
                
                #collect post-adaptation rollout batch in rand envs
                D_dash,rand_traj=collect_rollout_batch(env_rand, ds, da, policy, T_env, b, n_workers, queue, params=theta_dash,add_noise=add_noise)
                if args.dr_type=="adr": D_dashes_rand.append(rand_traj)
                D_dashes.append(D_dash)
                _, _, rewards,_ = D_dash
                t_agent += rewards.size
                t_eval += rewards.size
                if args.maml:
                    rewards_val_ep.append(rewards)
                else:
                    rewards_tr_ep.append(rewards)
            
            
            if args.dr_type=="adr":
                #ADR updates
                for t, (ref_traj, rand_traj) in enumerate(zip(D_dashes_ref,D_dashes_rand)):
                    T_disc_eps=0 #agent timesteps in the current iteration/episode
        
                    #calculate discriminator reward
                    for i in range(n_particles):
                        T_disc_eps += len(rand_traj[i])
                        
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
                prev_loss, _, prev_pis = surrogate_loss(D_dashes,policy,value_net,gamma,alpha=alpha,Ds=Ds)
            grads = parameters_to_vector(tape.gradient(prev_loss, policy.trainable_variables),policy.trainable_variables)
            prev_loss=tf.identity(prev_loss)
            hvp=HVP(D_dashes,policy,value_net,damping,alpha=alpha,Ds=Ds)
            search_step_dir=conjugate_gradients(hvp, grads, rdotr_tol=rdotr_tol,nsteps=nsteps)
            max_length=np.sqrt(2.0 * max_grad_kl / np.dot(search_step_dir, hvp(search_step_dir)))
            full_step=search_step_dir*max_length        
            prev_params = parameters_to_vector(policy.trainable_variables)
            line_search(policy, prev_loss, prev_pis, value_net, gamma, b, D_dashes, full_step, prev_params, max_grad_kl, max_backtracks, zeta, alpha=alpha, Ds=Ds)
            
            #evaluation
            if evaluate and t_eval>eval_freq:
                t_eval %= eval_freq
                eval_rewards=[]
    
                for _ in range(eval_eps):
                    if args.dr_type: env.randomize(["random"]*dr)
                    
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
                policy.save_weights(f"saved_models/model{common_name}")
            
            #log iteration results & statistics
            total_timesteps.append(t_agent)
            if episode % log_ival == 0:
                log_msg="Rewards Tr: {:.2f}, ".format(reward_ep)
                log_msg = log_msg + ("Rewards Val: {:.2f}, ".format(reward_val) if args.maml else "") + ("Rewards Disc: {:.2f}, ".format(scores_disc.mean()) if args.dr_type=="adr" else "")
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
            
        if args.dr_type=="adr": 
            plot_disc_rewards_all.append(plot_disc_rewards)
            plot_disc_rewards_mean = np.stack(plot_disc_rewards_all).mean(0)
            plot_disc_rewards_std = np.stack(plot_disc_rewards_all).std(0)
        
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
        
        if args.dr_type=="adr":
            df['Rewards_Disc_Mean']=plot_disc_rewards_mean
            df['Rewards_Disc_Std']=plot_disc_rewards_std
        
        df.to_pickle(f"plots/results{common_name}.pkl")
        
        #plot results
        title="Meta-Training Training Rewards" if args.maml else "Training Rewards"
        plt.figure(figsize=(16,8))
        plt.grid(1)
        plt.plot(plot_tr_rewards_mean)
        plt.fill_between(range(tr_eps), plot_tr_rewards_mean + plot_tr_rewards_std, plot_tr_rewards_mean - plot_tr_rewards_std,alpha=0.2)
        # plt.axhline(y = env.spec.reward_threshold, color = 'r', linestyle = '--',label='Solved')
        plt.title(title)
        plt.savefig(f'plots/tr{common_name}.png')
        
        if args.maml:
            title="Meta-Training Testing Rewards"
            plt.figure(figsize=(16,8))
            plt.grid(1)
            plt.plot(plot_val_rewards_mean)
            plt.fill_between(range(tr_eps), plot_val_rewards_mean + plot_val_rewards_std, plot_val_rewards_mean - plot_val_rewards_std,alpha=0.2)
            # plt.axhline(y = env.spec.reward_threshold, color = 'r', linestyle = '--',label='Solved')
            plt.title(title)
            plt.savefig(f'plots/val{common_name}.png')
        
        title="Meta-Testing Rewards" if args.maml else "Evaluation Rewards"
        plt.figure(figsize=(16,8))
        plt.grid(1)
        plt.plot(plot_eval_rewards_mean)
        plt.fill_between(range(len(plot_eval_rewards_mean)), plot_eval_rewards_mean + plot_eval_rewards_std, plot_eval_rewards_mean - plot_eval_rewards_std,alpha=0.2)
        # plt.axhline(y = env.spec.reward_threshold, color = 'r', linestyle = '--',label='Solved')
        plt.title(title)
        plt.savefig(f'plots/ts{common_name}.png')
        
        if args.dr_type=="adr":
            title="Discriminator Rewards"
            plt.figure(figsize=(16,8))
            plt.grid(1)
            plt.plot(plot_disc_rewards_mean)
            plt.fill_between(range(tr_eps), plot_disc_rewards_mean + plot_disc_rewards_std, plot_disc_rewards_mean - plot_disc_rewards_std,alpha=0.2)
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
        

        #record elapsed time and close envs
        end_time=timeit.default_timer()
        print("Elapsed Time: {:.1f} minutes \n".format((end_time-start_time)/60.0))
        
        env.close()
        if args.dr_type=="adr": env_ref.close()
        env_rand.close()