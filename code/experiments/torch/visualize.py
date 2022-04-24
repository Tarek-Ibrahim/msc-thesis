#%% TODOs
#TODO: include oracle in testing rewards plot
#TODO: accomodate dr>1 in testing

#%% Imports
#general
import numpy as np
import pandas as pd
import yaml
from utils import PolicyNetwork, ValueNetwork, adapt
import os
import argparse

#visualize
import matplotlib.pyplot as plt
# import seaborn as sns

#env
import gym
import gym_custom

#Utils
# import decimal

#ML
import torch
torch.cuda.empty_cache()
torch.set_default_tensor_type(torch.FloatTensor)

#%% Inputs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--agent_alg", "-a", type=str, default="trpo", help="RL algorithm of the agent", choices=["trpo","ppo"])
# parser.add_argument("--test_eps", "-t", type=int, default=20, help="Number of testing episodes per randomization value")
parser.add_argument("--mode", "-m", type=int, default=0, help="0: debug mode; 1: run mode")
parser.add_argument("--env_key", "-e", type=str, default="hopper_friction", help="Environment key")
parser.add_argument("--plot_tr_results", "-r", action='store_true', help="Whether to plot training results")
parser.add_argument("--plot_ts_results", "-p", action='store_true', help="Whether to plot test results")
parser.add_argument("--plot_sample_eff", "-E", action='store_true', help="Whether to plot sampling effeciency")
parser.add_argument("--plot_sampled_regs", "-S", action='store_true', help="Whether to plot sampled regions")
parser.add_argument("--plot_control_acs", "-c", action='store_true', help="Whether to plot control actions")
parser.add_argument("--save_results", "-s", action='store_true', help="Whether to save results")
parser.add_argument("--visualize", "-v", action='store_true', help="Whether to render the env")
parser.add_argument("--verbose", "-V", action='store_true', help="Whether to print log")
parser.add_argument("--test_random", "-R", action='store_true', help="Whether to use randomized envs in testing vs default/reference env")
args = parser.parse_args()

modes=["debug_mode","run_mode"]
mode=modes[args.mode] #modes[0]

with open("config.yaml", 'r') as f:
    config_file = yaml.load(f, Loader=yaml.FullLoader)

config=config_file[mode]

#Env
# env_keys=['hopper_friction','halfcheetah_friction',"lunarlander"]
env_key=args.env_key #env_keys[0]
env_name=config_file["env_names"][env_key]
env=gym.make(env_name)
ds=env.observation_space.shape[0] #state dims
da=env.action_space.shape[0] #action dims
dr=env.unwrapped.randomization_space.shape[0] #N_rand (no. of randomization params)
thr_r=env.spec.reward_threshold if args.mode else 0.
setting=env_name.split("_")[0]+f" {env.rand}"

#MAML
alpha=config["lr_agent"]

#TRPO/PPO
# algs=["trpo","ppo"]
alg=args.agent_alg #algs[0]
h=config["h_agent"]
gamma=config["gamma"]

#Active DR
T_svpg_init=config["T_svpg_init"]
T_svpg=config["T_svpg"]
T_dr=config["T_dr"]
n_particles=config["n_particles"]

#Models
in_size=ds
out_size=da

value_net = ValueNetwork(in_size,gamma).to(device)

#General
tr_eps=config["tr_eps"]
figsize=tuple(config_file["figsize"])

keys=['Rewards_Tr','Rewards_Val','Rewards_Eval','Rewards_Disc']

plot_tr_results=args.plot_tr_results
plot_sample_eff=args.plot_sample_eff
plot_sampled_regs=args.plot_sample_eff
plot_control_acs=args.plot_control_acs
plot_ts_results=args.plot_ts_results
save_results=args.save_results

includes_maml=[True,False]
dr_types=["","uniform_dr","active_dr","auto_dr"]
active_dr_rewarders=["disc","map_neg","map_delta"]
active_dr_opts=["svpg_a2c","svpg_ddpg","ddpg","sac"]
sac_entropy_tuning_methods=["","learn","anneal"]

current_dir=os.path.dirname(os.path.abspath(__file__))
plots_tr_dir=config_file["plots_tr_dir"] #os.path.join(current_dir,config_file["plots_tr_dir"])
plots_ts_dir=config_file["plots_ts_dir"]
models_dir=config_file["models_dir"]

filenames, filenames_sr, labels, labels_sr, policies, dfs, dfs_sr=[],[],[],[],[],[],[]
for include_maml in includes_maml:
    for dr_type in dr_types:
        for active_dr_rewarder in active_dr_rewarders:
            for active_dr_opt in active_dr_opts:
                for sac_entropy_tuning_method in sac_entropy_tuning_methods:
                    
                    filename=("maml_" if include_maml else "")+alg+("_" if dr_type else "")+dr_type+(f"_{active_dr_rewarder}_{active_dr_opt}" if dr_type=="active_dr" else "")+(f"_{sac_entropy_tuning_method}" if dr_type=="active_dr" and active_dr_opt=="sac" and sac_entropy_tuning_method else "")
                    label=("MAML" if include_maml else alg.upper())+(f" + {dr_type}" if dr_type else "")+(f" ({active_dr_rewarder} / {active_dr_opt}" if dr_type=="active_dr" else "")+(f" / {sac_entropy_tuning_method})" if dr_type=="active_dr" and active_dr_opt=="sac" and sac_entropy_tuning_method else ")" if "active_dr" in dr_type else "")
                    common_name = "_"+filename+"_"+env_key
                    if any("model"+common_name in name for name in os.listdir(models_dir)) and common_name not in filenames:
                        filenames.append(common_name)
                        labels.append(label)
                        if plot_sampled_regs and ("active_dr" in label.lower() or "auto_dr" in label.lower()): labels_sr.append(label)
                        if plot_tr_results or plot_sample_eff: dfs.append(pd.read_pickle(f"{plots_tr_dir}results{common_name}.pkl"))
                        policy=PolicyNetwork(in_size,h,out_size).to(device)
                        policy.load_state_dict(torch.load(f"{models_dir}model{common_name}"+".pt",map_location=device))
                        policy.eval()
                        policies.append(policy)
                        if plot_sampled_regs and ("active_dr" in common_name or "auto_dr" in common_name):
                            dfs_sr.append(pd.read_pickle(f"{plots_tr_dir}sampled_regions{common_name}.pkl"))
                            filenames_sr.append(common_name)

rand_step = 0.1
    
#Testing
visualize=args.visualize
test_eps=config["te_eps"] if not visualize else 1
test_random=args.test_random
test_step=0.1 if not visualize else 0.25

test_rewards=[[] for _ in range(len(policies))]
test_rewards_var=[[] for _ in range(len(policies))]
control_actions=[[] for _ in range(len(policies))]

#%% Training Results

#plot tr results
if plot_tr_results:
    for key in keys:
        
        key_mean=key+"_Mean"
        key_std=key+"_Std"
        
        title=f"{key} ({setting})"
        plt.figure(figsize=figsize)
        plt.grid(1)
        
        for i, df in enumerate(dfs):
            if key_mean in list(df.keys()):
                plt.plot(df[key_mean],label=labels[i])
                plt.fill_between(range(df[key_mean].size), df[key_mean]+df[key_std], df[key_mean]-df[key_std],alpha=0.2)
        
        if "disc" not in key.lower(): plt.axhline(y = thr_r, color = 'r', linestyle = '--',label='Solved')
        plt.xlabel("Training Episodes")
        plt.ylabel("Rewards")
        plt.title(title)
        plt.legend(loc="upper right")
        plt_name=key.split("_")[-1]
        if save_results:
            plt.savefig(f'{plots_ts_dir}{plt_name}_{env_key}.png')
            plt.close()
        else:
            plt.show()


#sampling efficiency
if plot_sample_eff:
    title=f"Sampling Efficiency ({setting})"
    plt.figure(figsize=figsize)
    plt.grid(1)
    for i, df in enumerate(dfs):
        plt.plot(df["Total_Timesteps"],df["Rewards_Eval_Mean"],label=labels[i])
        plt.fill_between(df["Total_Timesteps"], df["Rewards_Eval_Mean"] + df["Rewards_Eval_Std"], df["Rewards_Eval_Mean"] - df["Rewards_Eval_Std"], alpha=0.2)
    plt.axhline(y = thr_r, color = 'r', linestyle = '--',label='Solved')
    plt.xlabel("Training Timesteps")
    plt.ylabel("Evalutaion Rewards")
    plt.title(title)
    plt.legend(loc="upper right")
    if save_results:
        plt.savefig(f'{plots_ts_dir}sample_effeciency_{env_key}.png')
        plt.close()
    else:
        plt.show()


# plot sampled regions
if plot_sampled_regs:
    eps_step=int((tr_eps-T_svpg_init)/4)
    region_step=eps_step*T_svpg*n_particles
    
    for j, df_sr in enumerate(dfs_sr): 
        sampled_regions=[list(df_sr.values[:,i]) for i in range(df_sr.values.shape[-1])]
        for dim, regions in enumerate(sampled_regions):
            
            low=env.unwrapped.dimensions[dim].range_min
            high=env.unwrapped.dimensions[dim].range_max
            
            dim_name=env.unwrapped.dimensions[dim].name
            
            # d = decimal.Decimal(str(low))
            # step_exp=d.as_tuple().exponent-1
            # step=10**step_exp
        
            x=np.arange(low,high+rand_step,rand_step)
            
            title=f"Sampled Regions for Randomization Dim = {dim_name} {env.rand} Over Time for {labels_sr[j]}"
            plt.figure(figsize=figsize)
            plt.grid(1)
            plt.hist((regions[region_step*0:region_step*1],regions[region_step*1:region_step*2],regions[region_step*2:region_step*3], regions[region_step*3:]), np.arange(min(x),max(x)+2*rand_step,rand_step), histtype='barstacked', label=[f'{eps_step*1} eps',f'{eps_step*2} eps', f'{eps_step*3} eps', f'{eps_step*4} eps'],color=["lightskyblue","blueviolet","hotpink","lightsalmon"])
            plt.xlim(min(x), max(x)+rand_step)
            plt.xlabel("Randomization Range")
            plt.ylabel("Number of samples (Sampling Frequency)")
            plt.legend()
            plt.title(title)
            if save_results:
                plt.savefig(f'{plots_ts_dir}sampled_regions{filenames_sr[j]}.png')
                plt.close()
            else:
                plt.show()

#%% Testing

if plot_control_acs or plot_ts_results or visualize:
    rand_range=np.arange(0.0,1.1,test_step,dtype=np.float32) if test_random else range(1)
    
    for i, policy in enumerate(policies):
        if args.verbose: print(f"For {labels[i]}: \n")
        for dim in range(dr): #TODO: do properly for more than one dim
            dim_name= env.unwrapped.dimensions[dim].name
            low=env.unwrapped.dimensions[dim].range_min
            high=env.unwrapped.dimensions[dim].range_max
            scaled_values=low + (high-low) * rand_range
            values=["default"]*dr
            default_value_idx = list(scaled_values).index(min(scaled_values, key=lambda x:abs(x-env.unwrapped.dimensions[dim].default_value)) if test_random else 0)
            if args.verbose: print(f"For Dim: {dim_name}: \n")
            for j, rand_value in enumerate(rand_range):
                rand_value_rewards=[]
                values[dim]=rand_value #randomizing current dim while fixing rest to their default values
                if test_random and args.verbose: print(f"For Rand Value = {scaled_values[j]}: \n")
                for test_ep in range(test_eps):
                    if test_random: env.randomize(values)
                    
                    s=env.reset()
                    
                    if "maml" in filenames[i]:
                        state=torch.from_numpy(s).float().unsqueeze(0).to(device)
                        dist=policy(state,params=None)
                        a=dist.sample().squeeze(0).cpu().numpy()
                        s, r, done, _ = env.step(a)
                        R = r
                        if test_ep == 0 and rand_value==rand_range[default_value_idx] and plot_control_acs:
                            act=a[0] if a.ndim >1 else a
                            act=np.clip(act,-1.0, 1.0)
                            control_actions[i].append(act)
                    else:
                        done=False
                        R=0
                    
                    while not done:
                        
                        state=torch.from_numpy(s).float().unsqueeze(0).to(device)
                        
                        if "maml" in filenames[i]:
                            states=state.unsqueeze(0)
                            actions=torch.from_numpy(a).unsqueeze(0).unsqueeze(0).to(device)
                            rewards=torch.from_numpy(np.array(r)).float().unsqueeze(0).unsqueeze(0).to(device)
                            masks=torch.from_numpy(np.array(1.0)).float().unsqueeze(0).unsqueeze(0).to(device)
                            D=[states, actions, rewards, masks]
                                                
                            theta_dash=adapt(D,value_net,policy,alpha)
                        else:
                            theta_dash=None
                            
                        dist=policy(state,params=theta_dash)
                        a=dist.sample().squeeze(0).cpu().numpy()
                        
                        if test_ep == 0 and rand_value==rand_range[default_value_idx] and plot_control_acs:
                            act=a[0] if a.ndim >1 else a
                            act=np.clip(act,-1.0, 1.0)
                            control_actions[i].append(act)
                            
                        s, r, done, _ = env.step(a)
                        
                        if visualize: env.render()
                        
                        R+=r
                    rand_value_rewards.append(R)
                test_rewards[i].append(np.array(rand_value_rewards).mean())
                test_rewards_var[i].append(np.array(rand_value_rewards).std())
                    
        env.close()


#rewards
if plot_ts_results:
    title=f"Testing Rewards ({setting})"
    plt.figure(figsize=figsize)
    plt.grid(1)
    for i, test_reward in enumerate(test_rewards):
        plt.plot(scaled_values,test_reward,label=labels[i])
        plt.fill_between(scaled_values, np.array(test_reward) + np.array(test_rewards_var[i]), np.array(test_reward) - np.array(test_rewards_var[i]), alpha=0.2)
    plt.axhline(y = thr_r, color = 'r', linestyle = '--',label='Solved')
    plt.xlabel("Randomization Range")
    plt.ylabel("Rewards")
    plt.title(title)
    plt.legend(loc="upper right")
    if save_results:
        plt.savefig(f'{plots_ts_dir}ts_{test_eps}_episodes_{env_key}.png')
        plt.close()
    else:
        plt.show()


#control actions
if plot_control_acs:
    for i, policy in enumerate(policies):
        title=f"Control Actions for {labels[i]} ({setting})"
        plt.figure(figsize=figsize)
        plt.grid(1)
        plt.plot(control_actions[i],label=range(da))
        plt.xlabel("Timesteps for First Test Episode and Default Environment")
        plt.ylabel("Control Action")
        plt.title(title)
        plt.legend(loc="upper right")
        if save_results:
            plt.savefig(f'{plots_ts_dir}control_actions{filenames[i]}.png')
            plt.close()
        else:
            plt.show()
