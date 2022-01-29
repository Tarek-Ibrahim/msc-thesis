#%% Imports
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from baselines.common.vec_env import VecEnv, CloudpickleWrapper
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
# import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Distribution, Normal
from tensorboardX import SummaryWriter 
# from stable_baselines3 import SAC as sb_SAC
# from stable_baselines3.common.env_util import make_vec_env
# from stable_baselines3.common.vec_env import SubprocVecEnv as vecenv

#TODO: add entropy tuning; initial exploration steps
#???: how is diayn used for policy init?
#TODO: isolate SAC
#TODO: complete and fix implementation

#%% General
seeds=[None,1,2,3,4,5]
seed = seeds[1]

torch.cuda.empty_cache()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device="cpu"
torch.set_default_tensor_type(torch.FloatTensor)
torch.autograd.set_detect_anomaly(True)

epsilon = np.finfo(np.float32).eps.item()

#%% Utils & Common

progress=lambda x: tqdm.trange(x, leave=True) #for visualizing/monitoring training progress

def set_seed(seed,env):
    import random
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    env.seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ReplayBuffer(object):
    def __init__(self, max_size=1e6):
        self.storage = []
        self.max_size = int(max_size)
        self.next_idx = 0

    # Expects tuples of (state, next_state, action, reward, done)
    def add(self, data):
        if self.next_idx >= len(self.storage):
            self.storage.append(data)
        else:
            self.storage[self.next_idx] = data

        self.next_idx = (self.next_idx + 1) % self.max_size

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d = [], [], [], [], []

        for i in ind:
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)


class MLP(nn.Module):
    def __init__(self,in_size,h,out_size):
        super().__init__()
        
        self.mlp=nn.Sequential(
            nn.Linear(in_size, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Linear(h, out_size)
            )
        
    def forward(self,*inputs):
        inputs=torch.cat(inputs, dim=1)
        return self.mlp(inputs)
    
# def rollout(envs,n_workers,T,policy_agent,RB,T_agent_init,b_agent,gamma_agent,t_agent,t_eval,freeze_agent=True,add_noise=False,noise_scale=0.1):

#     states=envs.reset()
#     ep_rewards = np.zeros(n_workers)
#     dones = [False] * n_workers
#     add_to_buffer = [True] * n_workers
#     t=0
#     iters=0
    
#     while not all(dones) and t <= T:
#         actions=policy_agent.select_action(np.array(states))
        
#         if add_noise:
#             actions += np.random.normal(0, noise_scale, size=actions.shape)
#             actions = actions.clip(-1, 1)
        
#         next_states, rewards, dones, _ = envs.step(actions) #this steps through the envs even after they are done, but it doesn't matter here since data from an env is added to buffer only up until the point that the done signal is recieved from that env (and it is an off-policy algorithm)
#         # ep_rewards+=np.sum(rewards)
        
#         for i, st in enumerate(states):
#             if add_to_buffer[i]:
#                 iters += 1
#                 t_eval+=1
#                 t_agent+=1
#                 ep_rewards[i] += rewards[i]
                
#                 if RB is not None:
#                     done_bool = 0 if t + 1 == T else float(dones[i])
#                     RB.add((states[i], next_states[i], actions[i], rewards[i], done_bool))
        
#             if dones[i]:
#                 add_to_buffer[i] = False
                
#         states = next_states
#         t+=1
    
#     if not freeze_agent and len(RB.storage) > T_agent_init: #if it has enough samples
#         eps_agent=iters #t*n_rollouts
#         policy_agent.train(RB=RB, eps=eps_agent,batch_size=b_agent,gamma=gamma_agent)
            
#     return ep_rewards, t_eval, t_agent

    
def collect_rollouts():
    rollouts=[]
    for rollout in range(n_rollouts):
        state=env.reset()
        states, next_states, actions, rewards, dones = [], [], [], [], []
        next_state = None
        for t in range(T):
            skill=sample_skills(state.shape[0]) #???: every step or every rollout / rollout batch?
            action=pi(state,skill)
            next_state, reward, done, _ = env.step(action)
            states.append(state); actions.append(action); rewards.append(reward); dones.append(done); next_states.append(next_state)
            if done:
                break
            state=next_state
        rollout = [states,actions,rewards,next_states,dones]
        rollouts.append(rollout)
    return rollouts

#%% Env

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


def make_vec_envs(env_name, seed, n_workers):
    envs=[make_env(env_name,seed,rank) for rank in range(n_workers)]
    envs=SubprocVecEnv(envs, ds, da)
    return envs


env_names=["cartpole_custom-v1","halfcheetah_custom-v1","HalfCheetah-v2",'halfcheetah_custom_norm-v1','halfcheetah_custom_rand-v1','lunarlander_custom_default_rand-v0']
env_name=env_names[2]
env=gym.make(env_name)
T=env._max_episode_steps #task horizon / max env timesteps
ds=env.observation_space.shape[0] #state dims
da=env.action_space.shape[0] #action dims

#%%

dz=20 #no. of skills

sample_z= lambda: np.random.choice(dz,p=np.full(dz,1/dz)) #z = sample_z() #z is scalar int = idx of skill
#skill=z_one_hot = np.zeros(dz); z_one_hot[z]=1
#aug_state=state=[state,z_one_hot] (a form of obs_preproc)

def sample_skills(n_samples): 
    skills=np.zeros((n_samples,dz))
    for sample in range(n_samples):
        skills[sample,np.random.randint(0,dz)]=1
    return skills

#%% SAC

class TanhNormal(Distribution):
    """
    Represent distribution of X where
        X ~ tanh(Z)
        Z ~ N(mean, std)
    Note: this is not very numerically stable.
    """
    def __init__(self, normal_mean, normal_std, epsilon=1e-6):
        self.normal_mean = normal_mean
        self.normal_std = normal_std
        self.normal = Normal(normal_mean, normal_std)
        self.epsilon = epsilon

    def log_prob(self, value, pre_tanh_value):
        return self.normal.log_prob(pre_tanh_value) - torch.log(1 - value * value + self.epsilon) #enforcing action bound

    def sample(self, reparameterize): #reparametrize = for reparameterization trick (mean + std * N(0,1))
        if reparameterize:
            z = (self.normal_mean + self.normal_std * Normal(torch.zeros(self.normal_mean.size(),device=device), torch.ones(self.normal_std.size(),device=device)).sample()) #self.normal.rsample()
            z.requires_grad_()
        else:
            z = self.normal.sample().detach()
        return torch.tanh(z), z


class Policy(MLP):
    def __init__(self, in_size, h, out_size):
        super().__init__(in_size, h, 2 * out_size)
    
    def forward(self, state, skill=None, reparameterize=True, deterministic=False):
        inputs=state if skill is None else torch.cat([state, skill], -1)
        mean, log_std = self.mlp(inputs)
        log_std = torch.clamp(log_std, min=-20, max=2)
        std = torch.exp(log_std)
        
        log_prob = None
        if deterministic:
            action=torch.tanh(mean)
        else:
            tanh_normal = TanhNormal(mean, std)
            action, pre_tanh_value = tanh_normal.sample(reparameterize)
            log_prob = tanh_normal.log_prob(action,pre_tanh_value).sum(dim=1, keepdim=True)
        return (action, mean, log_std, log_prob) #???: should we take the tanh() of the mean?


h=300

lr_v=3e-4
lr_q=3e-4
lr_pi=3e-4

q1=MLP(in_size=ds+da+dz,h=h,out_size=1).to(device)
q2=MLP(in_size=ds+da+dz,h=h,out_size=1).to(device)
q1_target=MLP(in_size=ds+da+dz,h=h,out_size=1).to(device) #???: where/how are these used?
q2_target=MLP(in_size=ds+da+dz,h=h,out_size=1).to(device)
v=MLP(in_size=ds,h=h,out_size=1).to(device)
v_target=MLP(in_size=ds,h=h,out_size=1).to(device)
pi=Policy(in_size=ds+dz,h=h,out_size=da)

q1_optimizer = Adam(q1.parameters(),lr=lr_q)
q2_optimizer = Adam(q2.parameters(),lr=lr_q)
v_optimizer = Adam(v.parameters(),lr=lr_v)
pi_optimizer = Adam(pi.parameters(),lr=lr_pi)

q_loss_func=nn.MSELoss()
v_loss_func=nn.MSELoss()

#%% Discriminator

lr_disc=3e-4
discriminator=MLP(in_size=ds,h=h,out_size=dz).to(device)
disc_optimizer = Adam(discriminator.parameters(),lr=lr_disc)
disc_loss_func=nn.CrossEntropyLoss()

#%% Algorithm
#DIAYN is basically SAC +  discriminator + augmenting observations with skills

epochs=3000 #500
episodes=1 #1000
iterations=1000
T_init=1000 #minimum no. of steps before training
b=256
target_update_freq=1
n_rollouts=1
alpha=1
alpha_loss=0
policy_mean_reg_weight=1e-3
policy_std_reg_weight=1e-3

gamma=0.99
temp=5.0 #temperature / reward scaling
tau=0.005 #weighting factor for exp. moving average
entropy_target=-np.prod(da).item()

set_seed(seed)
env.seed(seed)
RB=ReplayBuffer()
plot_rewards=[]

# p_z = np.full(dz,1.0/dz)

for epoch in range(epochs):
    for episode in range(episodes):
        #collect rollouts
        rollouts=collect_rollouts()
        #add to buffer
        RB.add(rollouts)
        for iteration in range(iterations):
            #(#TODO: if buffer has enough samples) sample from buffer
            rollouts=RB.sample(b)
            states,actions,rewards,next_states,dones=zip(*rollouts)
            skills=sample_skills(states.shape[0])
            #train
                #update/train SAC agent
                    #update values (critic)
                    q1_value=q1(states,actions)
                    q2_value=q2(states,actions)
                    
                    states_split, skills_split = zip(*states) #split aug_state
                    disc=discriminator(states_split)
                    pz=(skills*skills_split).sum(1)
                    disc_loss = disc_loss_func(skills_split,disc) #FIXME
                    r=-disc_loss - torch.log(pz+epsilon) 
                    
                    q_target_value = temp*r + (1 - dones) * gamma * v_target(next_states) #???: is this aug_next_state? #this is different (uses q_target)
                    q1_loss = q_loss_func(q1_value, q_target_value.detach())
                    q2_loss = q_loss_func(q1_value, q_target_value.detach())
                    
                    new_actions, pi_mean, pi_log_std, log_pi = pi(states)
                    q_new_actions = torch.min(q1(states,new_actions),q2(states,new_actions))
                    v_value=v(states)
                    v_target_value=q_new_actions-log_pi
                    v_loss=v_loss_func(v_value,v_target_value.detach())
                    
                    q1_optimizer.zero_grad()
                    q1_loss.backward()
                    q1_optimizer.step()
            
                    q2_optimizer.zero_grad()
                    q2_loss.backward()
                    q2_optimizer.step()
            
                    v_optimizer.zero_grad()
                    v_loss.backward()
                    v_optimizer.step()
                    
                    #update policy (actor)
                    pi_loss = (log_pi - q_new_actions).mean()
                    #???: pi_loss = (log_pi*(log_pi-q_new_actions+v_value).detach()).mean()
                    
                    #???: how important is this reg loss?
                    mean_reg_loss = pi_mean_reg_weight * (pi_mean ** 2).mean()
                    std_reg_loss = pi_std_reg_weight * (pi_log_std ** 2).mean()
                    pi_reg_loss = mean_reg_loss + std_reg_loss
                    
                    pi_loss = pi_loss + pi_reg_loss
        
                    pi_optimizer.zero_grad()
                    pi_loss.backward()
                    pi_optimizer.step()
                    
                #update/train discriminator
                # disc_loss=disc_loss_func(disc,skills_split) #???: take the mean?
                
                disc_optimizer.zero_grad()
                disc_loss.backward()
                disc_optimizer.step()
                
                #TODO: soft update of targets?