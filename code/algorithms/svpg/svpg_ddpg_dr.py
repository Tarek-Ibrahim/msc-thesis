#%% Imports
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from scipy.spatial.distance import squareform, pdist
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
from torch.nn.utils.convert_parameters import parameters_to_vector as params2vec, _check_param_device, vector_to_parameters as vec2params
from torch.optim import Adam
import torch.nn.functional as F
import random


#%% General

seed = 101

torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device="cpu"
torch.set_default_tensor_type(torch.FloatTensor)
torch.autograd.set_detect_anomaly(True)


#%% Utils

def set_seed(seed):
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        

def parameters_to_vector(parameters, grad=False, both=False):
    # Flag for the device where the parameter is located
    param_device = None

    if not both:
        if not grad:
            return params2vec(parameters)
        else:
            vec = []
            for param in parameters:
                param_device = _check_param_device(param, param_device)
                vec.append(param.grad.data.view(-1))
                return torch.cat(vec)
    else:
        vec_params, vec_grads = [], []
        for param in parameters:
            param_device = _check_param_device(param, param_device)
            vec_params.append(param.data.view(-1))
            vec_grads.append(param.grad.data.view(-1))
        return torch.cat(vec_params), torch.cat(vec_grads)


def vector_to_parameters(vec, parameters, grad=True):
    param_device = None

    # Pointer for slicing the vector for each parameter
    pointer = 0
    if grad:
        for param in parameters:
            # Ensure the parameters are located in the same device
            param_device = _check_param_device(param, param_device)
            # The length of the parameter
            num_param = torch.prod(torch.LongTensor(list(param.size())))
            param.grad.data = vec[pointer:pointer + num_param].view(param.size()) #BUG: is this the problematic inplace operation?
            # Increment the pointer
            pointer = pointer + num_param
    else:
        vec2params(vec, parameters)


class PrioritizedReplayBuffer(object):
    def __init__(self, max_size=1e6):
        self.storage = []
        self.max_size = int(max_size)
        self.next_idx = 0
        self.alpha=0.8
        self.reset = False

    # Expects tuples of (state, next_state, action, reward, done)
    def add(self, data):
        self.reset = True
        if self.next_idx >= len(self.storage):
            self.storage.append((data,1.))
        else:
            self.storage[self.next_idx] = (data,1.)

        self.next_idx = (self.next_idx + 1) % self.max_size

    def sample(self, batch_size):
        
        if self.reset:
            data, weights = zip(*self.storage)
            self.storage=list(zip(data,list(np.array(weights)*self.alpha)))
            self.reset=False
        
        data, weights = zip(*self.storage)
        samples = random.choices(data,weights,k=batch_size)
        # ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d = [], [], [], [], []

        for sample in samples:
            X, Y, U, R, D = sample
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)
        

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

#%% SVPG 


class SVPGParticleActor(nn.Module):
    def __init__(self, in_size, h, out_size, max_action=1.):
        super(SVPGParticleActor, self).__init__()

        self.l1 = nn.Linear(in_size, h)
        self.l2 = nn.Linear(h, h)
        self.l3 = nn.Linear(h, out_size)

        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x) #self.max_action * torch.tanh(self.l3(x))
        return x


class SVPGParticleCritic(nn.Module):
    def __init__(self, in_size, h):
        super(SVPGParticleCritic, self).__init__()

        self.l1 = nn.Linear(in_size, h)
        self.l2 = nn.Linear(h, h)
        self.l3 = nn.Linear(h, 1)


    def forward(self, x, u):
        x = F.relu(self.l1(torch.cat([x, u], 1)))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


# class SVPGParticle(nn.Module):
#     def __init__(self, input_dim, output_dim, hidden_dim):
#         super(SVPGParticle, self).__init__()

#         self.critic = SVPGParticleCritic(input_dim, hidden_dim)
#         self.actor = SVPGParticleActor(input_dim, hidden_dim, output_dim)

#     def forward(self, x):
#         dist = self.actor(x)
#         value = self.critic(x)

#         return dist, value  


class SVPG:
    def __init__(self, n_particles, hidden_dim, lr, temperature, svpg_mode, gamma, T_svpg, dr, delta_max, H_svpg, epochs, batch_size, T_init, xp_type, tau=0.005):

        self.svpg_mode=svpg_mode
        self.temperature = temperature 
        self.n_particles = n_particles
        self.T_svpg = T_svpg
        self.dr = dr
        self.H_svpg = H_svpg
        self.gamma = gamma
        self.epochs = epochs
        self.batch_size = batch_size
        self.tau=tau
        self.xp_type=xp_type
        self.T_init = T_init
        self.a_max = delta_max
        self.RB=ReplayBuffer()
        # self.RB=PrioritizedReplayBuffer()
        
        self.last_states = np.random.uniform(0, 1, (self.n_particles, self.dr))
        self.timesteps = np.zeros(self.n_particles)
        
        self.particles_actor = []
        self.particles_actor_target = []
        self.particles_critic = []
        self.particles_critic_target = []
        self.optimizers_actor = []
        self.optimizers_critic = []
        
        

        for i in range(self.n_particles):
            # Initialize each of the individual particles
            actor = SVPGParticleActor(in_size=dr, h=hidden_dim, out_size=dr, max_action=self.a_max).to(device)
            actor_target = SVPGParticleActor(in_size=dr, h=hidden_dim, out_size=dr, max_action=self.a_max).to(device)
            actor_target.load_state_dict(actor.state_dict())
            actor_optimizer = Adam(actor.parameters(),lr=lr)
    
            critic = SVPGParticleCritic(in_size=2*dr, h=hidden_dim).to(device)
            critic_target = SVPGParticleCritic(in_size=2*dr, h=hidden_dim).to(device)
            critic_target.load_state_dict(critic.state_dict())
            critic_optimizer = Adam(critic.parameters(),lr=lr*10.)

            self.particles_actor.append(actor)
            self.particles_actor_target.append(actor_target)
            self.particles_critic.append(critic)
            self.particles_critic_target.append(critic_target)
            self.optimizers_actor.append(actor_optimizer)
            self.optimizers_critic.append(critic_optimizer)

    def calculate_kernel(self, X):

        X_np = X.cpu().data.numpy()
        pairwise_dists = squareform(pdist(X_np))**2

        # Median trick
        h = np.median(pairwise_dists)  
        h = np.sqrt(0.5 * h / np.log(self.n_particles+1))

        # Compute RBF Kernel
        k = torch.exp(-torch.from_numpy(pairwise_dists).to(device).float() / h**2 / 2)

        # Compute kernel gradient
        grad_k = -(k).matmul(X)
        sum_k = k.sum(1)
        if self.svpg_mode==1:
            grad_k= grad_k + torch.matmul(sum_k.unsqueeze(0),X)
            # for i in range(X.shape[1]):
            #     grad_k[:, i] = grad_k[:, i] + X[:, i].matmul(sum_k)
        elif self.svpg_mode==2:
            grad_k= grad_k + torch.multiply(X,sum_k.unsqueeze(1))
            # for i in range(X.shape[1]):
            #     grad_k[:, i] = grad_k[:, i] + torch.multiply(X[:, i],sum_k)
        elif self.svpg_mode==3:
            grad_k = grad_k
        else:
            RuntimeError("unkown svpg mode")
        grad_k = grad_k / (h ** 2)

        return k, grad_k
    
    def select_action(self, policy_idx, state):
        state = torch.FloatTensor(state).to(device)
        return self.particles_actor[policy_idx](state).cpu().data.numpy()
    
    def step(self):
        self.simulation_instances = np.zeros((self.n_particles, self.T_svpg, self.dr))

        # Store the last states for each particle (calculating rewards)
        # self.masks = np.ones((self.n_particles, self.T_svpg))
        # done=[False]*self.n_particles

        for i in range(self.n_particles):
            current_sim_params = self.last_states[i]

            for t in range(self.T_svpg):
                self.simulation_instances[i][t] = current_sim_params

                action = self.select_action(i, current_sim_params) if len(self.RB.storage) > self.T_init else self.a_max * np.random.uniform(-1, 1, (self.dr,))
                
                next_params = current_sim_params + action
                
                if self.xp_type=="peak":
                    reward = 1. if next_params <= 0.6 and next_params >= 0.4 else -1.
                    # reward = 1./(np.abs(0.5-next_params)+1e-8)
                elif self.xp_type=="valley":
                    reward = -1. if next_params <= 0.6 and next_params >= 0.4 else 1.
                    # reward = -1./(np.abs(0.5-next_params)+1e-8)
                done=True if next_params < 0. or next_params > 1. else False
                # next_params = np.clip(next_params,0,1)
                done_bool = 0 if self.timesteps[i] + 1 == self.H_svpg else float(done)
                
                self.RB.add((current_sim_params, next_params, action, reward, done_bool))
                
                if done_bool:
                    current_sim_params = np.random.uniform(0, 1, (self.dr,))                
                    self.timesteps[i] = 0
                    # add_to_buffer=False
                    # break
                else:
                    current_sim_params = next_params
                    self.timesteps[i] += 1

            self.last_states[i] = current_sim_params

        return np.array(self.simulation_instances), self.last_states

    def train(self):
        
        if len(self.RB.storage) > self.T_init: 
            for _ in range(self.epochs):
        
                policy_grads = []
                parameters = []
                # critic_losses = []
        
                for i in range(self.n_particles):
                    
                    x, y, u, r, d = self.RB.sample(self.batch_size)
                    state = torch.FloatTensor(x).to(device)
                    action = torch.FloatTensor(u).to(device)
                    next_state = torch.FloatTensor(y).to(device)
                    done = torch.FloatTensor(1 - d).to(device)
                    reward = torch.FloatTensor(r).to(device)
                    
                    # Compute the target Q value
                    target_Q = self.particles_critic_target[i](next_state, self.particles_actor_target[i](next_state))
                    target_Q = reward + (done * self.gamma * target_Q).detach()
        
                    # Get current Q estimate
                    current_Q = self.particles_critic[i](state, action)
        
                    # Compute critic loss
                    critic_loss = F.mse_loss(current_Q, target_Q)
                    # critic_losses.append(critic_loss)
                    
                    self.optimizers_critic[i].zero_grad()
                    critic_loss.backward()
                    self.optimizers_critic[i].step()
                    
                    policy_grad = self.particles_critic[i](state, self.particles_actor[i](state)).mean()
                
                    # Optimize the actor 
                    self.optimizers_actor[i].zero_grad()
                    policy_grad.backward()
                    # self.actor_optimizer.step()
                    
                    # Vectorize parameters and PGs
                    vec_param, vec_policy_grad = parameters_to_vector(list(self.particles_actor[i].parameters()), both=True)
        
                    policy_grads.append(vec_policy_grad.unsqueeze(0))
                    parameters.append(vec_param.unsqueeze(0))
        
                # calculating the kernel matrix and its gradients
                parameters = torch.cat(parameters)
                k, grad_k = self.calculate_kernel(parameters)
        
                policy_grads = 1.0 / self.temperature * torch.cat(policy_grads)
                grad_logp = torch.mm(k, policy_grads)
        
                grad_theta = - (grad_logp + grad_k) / self.n_particles
        
                # # update param gradients
                for i in range(self.n_particles):
                    vector_to_parameters(grad_theta[i], list(self.particles_actor[i].parameters()), grad=True)
                    self.optimizers_actor[i].step()
        
                for i in range(self.n_particles):
                    # Update the frozen target models
                    for param, target_param in zip(self.particles_critic[i].parameters(), self.particles_critic_target[i].parameters()):
                        target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
                    for param, target_param in zip(self.particles_actor[i].parameters(), self.particles_actor_target[i].parameters()):
                        target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


#%% Implementation (ADR algorithm)
if __name__ == '__main__':
    
    n_particles=3 #10 
    temp=0.0001 #0.0001 #temperature
    lr_svpg=0.003 #0.0003
    gamma_svpg=0.99
    h_svpg=64 #100
    svpg_modes=[1,2,3] #how to calculate kernel gradient #1: original implementation; 2 & 3: other variants
    svpg_mode=svpg_modes[1]
    xp_types=["peak","valley"] #experiment types
    xp_type=xp_types[0]
    T_svpg=100 #50 #svpg rollout length
    delta_max = 0.5 #0.05 #0.005 #0.05 #maximum allowable change to svpg states (i.e. upper bound on the svpg action)
    H_svpg = 100 # 100 #svpg horizon (how often the particles are reset)
    rewards_scale=1.
    epochs=30
    batch_size=250
    T_init=100
    
    env_names=['halfcheetah_custom_norm-v1','halfcheetah_custom_rand-v1','lunarlander_custom_820_rand-v0','cartpole_custom-v1']
    env_name=env_names[-2]
    env=gym.make(env_name)
    T_env=env._max_episode_steps #task horizon / max env timesteps
    ds=env.observation_space.shape[0] #state dims
    da=env.action_space.shape[0] #action dims
    dr=env.unwrapped.randomization_space.shape[0]
    n_workers=n_particles
        
    svpg = SVPG(n_particles, h_svpg, lr_svpg, temp, svpg_mode, gamma_svpg, T_svpg, dr, delta_max, H_svpg, epochs, batch_size, T_init, xp_type)
    
    set_seed(seed)
    
    T_eps=1000
    plot_tr_rewards_mean=[]
    sampled_regions = [[] for _ in range(dr)]
    rand_step=0.1 #for discretizing the sampled regions plot
    common_name="_svpg_ddpg_dr"
    t_eps=0
    
    with tqdm.tqdm(total=T_eps) as pbar:
        while t_eps < T_eps:
            
            #collect rollout batch with svpg particles (storing values)
            simulation_instances, next_instances = svpg.step()
            
            #calculate deterministic reward
            simulation_instances_mask = np.concatenate([simulation_instances[:,1:,0],next_instances],1)
            rewards = np.ones_like(simulation_instances_mask,dtype=np.float32)
            if xp_type =="peak":
                rewards[((simulation_instances_mask<=0.40).astype(int) + (simulation_instances_mask>=0.60).astype(int)).astype(bool)]=-1.
                # rewards *= 1./(np.abs(0.5-simulation_instances_mask)+1e-8)
            elif xp_type=="valley":
                rewards[((simulation_instances_mask>=0.40).astype(int) * (simulation_instances_mask<=0.60).astype(int)).astype(bool)]=-1.
                # rewards *= - 1./(np.abs(0.5-simulation_instances_mask)+1e-8)
            
            rewards = rewards * rewards_scale
            
            mean_rewards=rewards.sum(-1).mean()
            plot_tr_rewards_mean.append(mean_rewards)
            
            #train svpg 
            svpg.train()
            
            #plot sampled regions
            for dim in range(dr):
                dim_name=env.unwrapped.dimensions[dim].name
                low=env.unwrapped.dimensions[dim].range_min
                high=env.unwrapped.dimensions[dim].range_max
                x=np.arange(low,high+rand_step,rand_step)
                linspace_x=np.arange(min(x),max(x)+2*rand_step,rand_step)
                
                scaled_instances=low + (high-low) * simulation_instances[:, :, dim]
                sampled_regions[dim]=np.concatenate([sampled_regions[dim],scaled_instances.flatten()])
                  
                title=f"Sampled Regions for Randomization Dim = {dim_name} {env.rand} at Episode = {t_eps}"
                plt.figure(figsize=(16,8))
                plt.grid(1)
                plt.hist(sampled_regions[dim], linspace_x, histtype='barstacked')
                plt.xlim(min(x), max(x)+rand_step)
                plt.title(title)
                plt.savefig(f'plots/sampled_regions_dim_{dim_name}_{env.rand}{common_name}.png')
                plt.close()
                
                title=f"Value Function for Randomization Dim = {dim_name} {env.rand} at Episode = {t_eps}"
                plt.figure(figsize=(16,8))
                plt.grid(1)
                ls=np.linspace(0,1,len(linspace_x))
                for i in range(n_particles):
                    a=svpg.particles_actor[i](torch.from_numpy(ls).unsqueeze(1).float().to(device))
                    v=svpg.particles_critic[i](torch.from_numpy(ls).unsqueeze(1).float().to(device),a)
                    plt.plot(linspace_x,v.detach().cpu().numpy())
                plt.xlim(min(x), max(x)+rand_step)
                plt.title(title)
                plt.savefig(f'plots/value_function_dim_{dim_name}_{env.rand}{common_name}.png')
                plt.close()
            
            #log episode results
            log_msg="Reward: {:.2f}, Episode: {}".format(mean_rewards, t_eps)
            pbar.update(); pbar.set_description(desc=log_msg); pbar.refresh()
            t_eps+=1
            
    #%% Results & Plots

    title="Training Rewards"
    plt.figure(figsize=(16,8))
    plt.grid(1)
    plt.plot(plot_tr_rewards_mean)
    plt.title(title)
    plt.show()
