#%% Imports
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import gym
#------only for spyder IDE
for env in gym.envs.registration.registry.env_specs.copy():
     if 'custom' in env:
         print('Remove {} from registry'.format(env))
         del gym.envs.registration.registry.env_specs[env]
#------
import gym_custom
import torch
import torch.nn as nn
from torch.optim import Adam


#%% General
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device="cpu"
torch.set_default_tensor_type(torch.FloatTensor)
torch.autograd.set_detect_anomaly(True)

#%% Utils

def set_seed(seed,env,det=True):
    import random
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    env.seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
def collect_rollout(env, ds, da, policy, T, b): # a batch of rollouts
    
    rewards = []
    log_probs = []
    vals=[]
    next_vals=[]
    
    #each rollout in the batch is the history of stepping through the environment once till termination
    s =env.reset()
    done=False

    while not done:
    
        dist, value=policy.select_action(s)
    
        action_tensor = dist.sample()
        log_prob=dist.log_prob(action_tensor)
        a=action_tensor.squeeze(0).cpu().detach().numpy()
        
        s_dash, r, done, _ = env.step(a)
        
        _, next_value=policy.select_action(s_dash)
        
        #append to batch
        rewards.append(np.array(r).astype(np.float32))
        log_probs.append(log_prob)
        vals.append(value)
        next_vals.append(next_value)
                
        #reset
        s = s_dash
    
    D=[rewards, log_probs, vals, next_vals]
    
    return D


#%% SVPG 

class SVPGParticleCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SVPGParticleCritic, self).__init__()

        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.critic(x)


class SVPGParticleActor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SVPGParticleActor, self).__init__()

        self.actor_hidden = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
            # nn.Tanh()
        )
        
        self.logstd=nn.Parameter(torch.zeros((1,output_dim)))

    def forward(self, x):
        
        mean=self.actor_hidden(x)
        
        std=torch.exp(self.logstd) + 1e-6 #0.5 
                
        return torch.distributions.Independent(torch.distributions.Normal(mean,std),1) #torch.distributions.Normal(mean, std)


class SVPGParticle(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(SVPGParticle, self).__init__()

        self.critic = SVPGParticleCritic(input_dim, hidden_dim)
        self.actor = SVPGParticleActor(input_dim, hidden_dim, output_dim)

    def forward(self, x):
        dist = self.actor(x)
        value = self.critic(x)

        return dist, value  


class SVPG:
    def __init__(self, ds, da, hidden_dim, lr, gamma):
        self.gamma = gamma
        self.policy = SVPGParticle(input_dim=ds, output_dim=da, hidden_dim=hidden_dim).to(device)
        self.optimizer = Adam(self.policy.parameters(), lr=lr)

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        dist, value = self.policy(state)

        return dist, value

    def compute_returns(self, next_value, rewards, masks):
        return_ = 0. #next_value 
        returns = []
        for step in reversed(range(len(rewards))):
            return_ = self.gamma * masks[step] * return_ + rewards[step]
            returns.insert(0, return_)

        return returns

    def train(self, D):
        
        rewards, log_probs, values, next_values=D

        particle_rewards = torch.from_numpy(np.array(rewards)).float().to(device)
        masks = torch.from_numpy(np.ones(len(rewards))).float().to(device)

        # Calculate advantages
        returns = self.compute_returns(next_values[-1], particle_rewards, masks)
        # returns = torch.cat(returns).detach()
        returns = torch.stack(returns).detach()
        # returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        advantages = returns - torch.cat(values)
        # TODO: normalize advantages
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        #zero grad
        self.optimizer.zero_grad()
        params = self.policy.parameters()
        for p in params:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
                
        # update critic
        critic_loss = 0.5 * advantages.pow(2).mean()
        critic_loss.backward()
        
        #update actor
        actor_loss = - (torch.cat(log_probs)*advantages.detach()).mean()
        actor_loss.backward()
        
        self.optimizer.step()


#%% Implementation (ADR algorithm)
if __name__ == '__main__':
    
    lr_svpg=0.003 #0.0003
    gamma_svpg=0.99
    h_svpg=64 #100
    
    env_names=['halfcheetah_custom_norm-v1','halfcheetah_custom_rand-v1','lunarlander_custom_820_rand-v0','cartpole_custom-v1']
    env_name=env_names[-1]
    env=gym.make(env_name)
    T_env=env._max_episode_steps #task horizon / max env timesteps
    ds=env.observation_space.shape[0] #state dims
    da=env.action_space.shape[0] #action dims
    b=1 #rollout batch size (not implemented)
        
    svpg = SVPG(ds, da, h_svpg, lr_svpg, gamma_svpg)
    
    seed = 1
    set_seed(seed, env)
        
    t_agent=0
    T_agent=int(1e6) #5000 #max agent timesteps
    plot_tr_rewards_mean=[]
    consec=0 #no. of consecutive episodes the reward stays at or above rewards threshold
    
    with tqdm.tqdm(total=T_agent) as pbar:
        while t_agent < T_agent:
            
            #collect rollout batch with svpg particles (storing values)
            D=collect_rollout(env, ds, da, svpg, T_env, b)
            
            #train svpg
            svpg.train(D)
            
            #log rewards and timesteps
            rewards, _, vals, _ =D
            plot_tr_rewards_mean.append(sum(rewards))
            for val in vals:
                t_agent += len(val)
                
            if sum(rewards) >= env.spec.reward_threshold:
                consec +=1
            else:
                consec = 0
            if consec >= 5:
                print(f"Solved at {t_agent} timesteps!")
                break
    
            log_msg="Reward: {:.2f}, Timesteps: {}".format(sum(rewards), t_agent)
            pbar.update(); pbar.set_description(desc=log_msg); pbar.refresh()


    #%% Results & Plots

    title="Training Rewards"
    plt.figure(figsize=(16,8))
    plt.grid(1)
    plt.plot(plot_tr_rewards_mean)
    plt.title(title)
    plt.show()
