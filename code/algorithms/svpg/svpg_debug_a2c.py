#%% Imports
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
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
import queue as Q


#%% General
seed = 101

torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device="cpu"
torch.set_default_tensor_type(torch.FloatTensor)
torch.autograd.set_detect_anomaly(True)


#%% Utils

def set_seed(seed):
    import random
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        

def collect_rollout_batch(envs, ds, da, policy, T, b, n_workers, queue): # a batch of rollouts
    rewards = [[] for _ in range(b)]
    log_probs = [[] for _ in range(b)]
    vals=[[] for _ in range(b)]
    next_vals=[[] for _ in range(b)]
    
    for rollout_idx in range(b):
        queue.put(rollout_idx)
    for _ in range(n_workers):
        queue.put(None)
    
    #each rollout in the batch is the history of stepping through the environment once till termination
    s, rollout_idxs=envs.reset()
    dones=[False]
    
    while (not all(dones)) or (not queue.empty()):
        
        dists, values=zip(*[policy.select_action(rollout_idxs[i],state) if rollout_idxs[i] is not None else (None,None) for i,state in enumerate(s)])
        
        action_tensor = [dist.sample() if dist is not None else None for dist in dists]
        log_prob=[dist.log_prob(action_tensor[i]) if dist is not None else None for i, dist in enumerate(dists)]
        a=np.array([at.squeeze().cpu().detach().numpy() if at is not None else np.zeros(da) if da>1 else 0. for at in action_tensor])
        
        s_dash, r, dones, rollout_idxs_new, _ = envs.step(a)
        
        _, next_values=zip(*[policy.select_action(rollout_idxs[i],state) if rollout_idxs[i] is not None else (None,None) for i,state in enumerate(s_dash)])
        
        #append to batch
        for reward, rollout_idx, lp, value, next_value in zip(r,rollout_idxs, log_prob, values, next_values):
            if rollout_idx is not None:
                rewards[rollout_idx].append(reward.astype(np.float32))
                log_probs[rollout_idx].append(lp)
                vals[rollout_idx].append(value)
                next_vals[rollout_idx].append(next_value)
                
        #reset
        s, rollout_idxs = s_dash, rollout_idxs_new
    
    #reshape
    T_max=max(map(len,rewards))
    rewards_mat=np.zeros((b,T_max),dtype=np.float32)
    masks_mat=np.zeros((b,T_max),dtype=np.float32)
    
    for rollout_idx in range(b):
        T_rollout=len(rewards[rollout_idx])
        rewards_mat[rollout_idx,:T_rollout]= np.stack(rewards[rollout_idx])
        masks_mat[rollout_idx,:T_rollout]=1.0
    
    D=[rewards_mat, masks_mat, log_probs, vals, next_vals]
    
    return D

        
#%% Envs

class SubprocVecEnv(gym.Env):
    def __init__(self,env_funcs,ds,da,queue,lock):
        
        self.parent_conns, self.child_conns = zip(*[mp.Pipe() for _ in env_funcs])
        self.workers = [EnvWorker(child_conn, env_func, queue, lock) for (child_conn, env_func) in zip(self.child_conns, env_funcs)]
        
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


class EnvWorker(mp.Process):
    def __init__(self, child_conn, env_func, queue, lock):
        super().__init__()
        
        self.child_conn = child_conn
        self.env = env_func()
        self.queue = queue
        self.lock = lock
        self.rollout_idx = None
        self.done = False
        self.ds = self.env.observation_space.shape
    
    def try_reset(self):
        with self.lock:
            try:
                self.rollout_idx = self.queue.get()
                self.done = (self.rollout_idx is None)
            except Q.Empty:
                self.done = True
        if self.done:
            state = np.zeros(self.ds, dtype=np.float32)
        else:
            state = self.env.reset()
        return state
    
    def run(self):
        while True:
            func, arg = self.child_conn.recv()
            
            if func == 'step':
                if self.done:
                    state, reward, done, info = np.zeros(self.ds, dtype=np.float32), 0.0, True, {}
                else:
                    state, reward, done, info = self.env.step(arg)
                if done and not self.done:
                    state = self.try_reset()
                self.child_conn.send((state,reward,done,self.rollout_idx,info))
                
            elif func == 'reset':
                state = self.try_reset()
                self.child_conn.send((state,self.rollout_idx))
                
            elif func == 'close':
                self.child_conn.close()
                break

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
        
        std=torch.exp(self.logstd) + 1e-6
                
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
    def __init__(self, n_particles, ds, da, hidden_dim, lr, gamma):
        self.particles = []
        self.optimizers = []
        self.n_particles = n_particles
        self.gamma = gamma

        for i in range(self.n_particles):
            # Initialize each of the individual particles
            policy = SVPGParticle(input_dim=ds,
                                  output_dim=da,
                                  hidden_dim=hidden_dim).to(device)

            optimizer = Adam(policy.parameters(), lr=lr)
            self.particles.append(policy)
            self.optimizers.append(optimizer)

    def select_action(self, policy_idx, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        policy = self.particles[policy_idx]
        dist, value = policy(state)

        return dist, value

    def compute_returns(self, next_value, rewards, masks):
        return_ = 0. #next_value 
        returns = []
        for step in reversed(range(len(rewards))):
            return_ = self.gamma * masks[step] * return_ + rewards[step]
            returns.insert(0, return_)

        return returns

    def train(self, D):
        
        rewards, masks_mat, log_probs, values, next_values=D

        for i in range(self.n_particles):

            particle_rewards = torch.from_numpy(rewards[i]).float().to(device)
            masks = torch.from_numpy(masks_mat[i]).float().to(device)

            # Calculate advantages
            returns = self.compute_returns(next_values[i][-1], particle_rewards, masks)
            returns=returns[:len(values[i])]
            returns = torch.stack(returns).detach()
            advantages = returns - torch.cat(values[i])
            # TODO: normalize advantages
            # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            #zero grad
            self.optimizer.zero_grad()
            params = self.particles[i].parameters()
            for p in params:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                    
            # update critic
            critic_loss = 0.5 * advantages.pow(2).mean()
            critic_loss.backward()
            
            #update actor
            actor_loss = - (torch.cat(log_probs[i])*advantages.detach()).mean()
            actor_loss.backward()
            
            self.optimizers[i].step()


#%% Implementation (ADR algorithm)
if __name__ == '__main__':
    
    queue = mp.Queue()
    lock = mp.Lock()
    
    n_particles=3 #10
    lr_svpg=0.003 #0.0003 
    gamma_svpg=0.99
    h_svpg=64 #100 
    
    env_names=['halfcheetah_custom_norm-v1','halfcheetah_custom_rand-v1','lunarlander_custom_820_rand-v0','cartpole_custom-v1']
    env_name=env_names[-1]
    env=gym.make(env_name)
    T_env=env._max_episode_steps #task horizon / max env timesteps
    ds=env.observation_space.shape[0] #state dims
    da=env.action_space.shape[0] #action dims
    n_workers=n_particles
    b=n_particles #rollout batch size
    
    envs=make_vec_envs(env_name, seed, n_workers, ds, da, queue, lock)
    
    svpg = SVPG(n_particles, ds, da, h_svpg, lr_svpg, gamma_svpg)
    
    set_seed(seed)
    
    t_agent=0
    T_agent=int(1e6) #5000 #max agent timesteps
    plot_tr_rewards_mean=[]
    consec=0 #no. of consecutive episodes the reward stays at or above rewards threshold
    
    with tqdm.tqdm(total=T_agent) as pbar:
        while t_agent < T_agent:
            
            #collect rollout batch with svpg particles (storing values)
            D=collect_rollout_batch(envs, ds, da, svpg, T_env, b, n_workers, queue)
            
            #train svpg
            svpg.train(D)
            
            #log rewards and timesteps
            rewards, _, _, vals, _ =D
            
            mean_rewards=rewards.sum(-1).mean()
            plot_tr_rewards_mean.append(mean_rewards)
            if mean_rewards >= env.spec.reward_threshold:
                    consec +=1
            else:
                consec = 0
            if consec >= 5:
                print(f"Solved at {t_agent} timesteps!")
                break
            
            log_msg="Reward: {:.2f}, Timesteps: {}".format(mean_rewards, t_agent)
            # print(log_msg+f" Timesteps: {t_agent} \n")
            pbar.update(); pbar.set_description(desc=log_msg); pbar.refresh()


    #%% Results & Plots

    title="Training Rewards"
    plt.figure(figsize=(16,8))
    plt.grid(1)
    plt.plot(plot_tr_rewards_mean)
    plt.title(title)
    plt.show()
