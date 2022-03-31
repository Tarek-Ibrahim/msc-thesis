#%% Imports
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from scipy.spatial.distance import squareform, pdist
import gym
#------only for spyder IDE
"""
for env in gym.envs.registration.registry.env_specs.copy():
     if 'custom' in env:
         print('Remove {} from registry'.format(env))
         del gym.envs.registration.registry.env_specs[env]
"""
#------
import gym_custom
import torch
import torch.nn as nn
from torch.nn.utils.convert_parameters import parameters_to_vector as params2vec, _check_param_device, vector_to_parameters as vec2params
from torch.optim import Adam


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
            nn.Tanh()
        )
        
        self.logstd=nn.Parameter(torch.zeros((1,output_dim))-1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=1e-5)
                nn.init.constant_(m.bias, 0.)


    def forward(self, x):
        mean=self.actor_hidden(x)
        
        std=torch.exp(self.logstd) + 1e-6
                
        return torch.distributions.Independent(torch.distributions.Normal(mean,std),1) #torch.distributions.Normal(mean, std)


class SVPGParticle(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(SVPGParticle, self).__init__()

        self.critic = SVPGParticleCritic(input_dim, hidden_dim)
        self.actor = SVPGParticleActor(input_dim, hidden_dim, output_dim)
        self.saved_log_probs=[]
        self.apply(self.init_weights)

    def init_weights(self, m):
        pass
        #if isinstance(m, nn.Linear):
        #    print(m)
        #    torch.nn.init.normal_(m.weight, 0.0, 2.5)
        #    torch.nn.init.normal_(m.bias, 0.0, 0.01)
        #    #m.bias.data.fill_(0.0)

    def forward(self, x):
        dist = self.actor(x)
        value = self.critic(x)

        return dist, value  


class SVPG:
    def __init__(self, n_particles, hidden_dim, lr, temperature, svpg_mode, gamma, T_svpg, dr, delta_max, H_svpg):
        self.particles = []
        self.optimizers = []
        self.svpg_mode=svpg_mode
        self.temperature = temperature 
        self.n_particles = n_particles
        self.gamma = gamma
        self.T_svpg = T_svpg
        self.dr = dr
        self.delta_max = delta_max
        self.H_svpg = H_svpg
        
        self.last_states = np.random.uniform(0, 1, (self.n_particles, self.dr))
        self.timesteps = np.zeros(self.n_particles)

        for i in range(self.n_particles):
            # Initialize each of the individual particles
            policy = SVPGParticle(input_dim=dr,
                                  output_dim=dr,
                                  hidden_dim=hidden_dim).to(device)

            optimizer = Adam(policy.parameters(), lr=lr)
            self.particles.append(policy)
            self.optimizers.append(optimizer)

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
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        policy = self.particles[policy_idx]
        dist, value = policy(state)
        
        action = dist.sample()               
            
        policy.saved_log_probs.append(dist.log_prob(action))
        
        action = action.item()

        return action, value

    def compute_returns(self, next_value, rewards, masks):
        return_ = next_value  #0.
        returns = []
        for step in reversed(range(len(rewards))):
            return_ = self.gamma * masks[step] * return_ + rewards[step]
            returns.insert(0, return_)

        return returns
    
    def step(self):
        self.simulation_instances = np.zeros((self.n_particles, self.T_svpg, self.dr))

        # Store the values of each state - for advantage estimation
        self.values = [torch.zeros((self.T_svpg, 1)).float().to(device) for _ in range(self.n_particles)]
        # Store the last states for each particle (calculating rewards)
        self.masks = np.ones((self.n_particles, self.T_svpg))
        # done=[False]*self.n_particles

        for i in range(self.n_particles):
            self.particles[i].saved_log_probs = [] #reset
            current_sim_params = self.last_states[i]

            for t in range(self.T_svpg):
                self.simulation_instances[i][t] = current_sim_params

                action, value = self.select_action(i, current_sim_params)  
                self.values[i][t] = value
                
                clipped_action = self.delta_max * np.array(np.clip(action, -1, 1)) 
                next_params = np.clip(current_sim_params + clipped_action, 0, 1)

                #next_params = np.array(np.clip(action, -1, 1)) 
                # next_params = current_sim_params + clipped_action
                # done[i]=True if next_params < 0 or next_params > 1 else False 
                
                if self.timesteps[i] + 1 == self.H_svpg: #or done[i]:
                #if np.array_equal(next_params, current_sim_params) or self.timesteps[i] + 1 == self.H_svpg: #or done[i]:
                    next_params = np.random.uniform(0, 1, (self.dr,))
                    
                    self.masks[i][t] = 0 # done = True
                    self.timesteps[i] = -1

                current_sim_params = next_params
                self.timesteps[i] += 1

            self.last_states[i] = current_sim_params

        return np.array(self.simulation_instances), self.last_states
        

    def train(self, rewards):
        
        policy_grads = []
        parameters = []
        critic_losses = []

        for i in range(self.n_particles):
            
            _, next_value = self.select_action(i, self.last_states[i]) 

            particle_rewards = torch.from_numpy(rewards[i]).float().to(device)
            masks = torch.from_numpy(self.masks[i]).float().to(device)

            # Calculate entropy-augmented returns, advantages
            returns = self.compute_returns(next_value, particle_rewards, masks)
            returns = torch.cat(returns).detach()
            next_values = torch.zeros_like(self.values[i])
            next_values[1:] = self.values[i][:-1]
            next_values[-1] = next_value

            returns = torch.from_numpy(rewards[i]).float() + self.gamma * next_values.squeeze() * masks
            returns = returns.detach()
            # returns = torch.stack(returns).detach()
            # returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            advantages = returns - self.values[i].squeeze()
            # TODO: normalize advantages
            # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            self.optimizers[i].zero_grad()
            params = self.particles[i].parameters()
            for p in params:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                    
            # Compute value loss, update critic
            critic_loss = 0.5 * advantages.pow(2).mean()
            critic_losses.append(critic_loss)
            
            # policy_grad = (torch.cat(log_probs[i])*advantages.detach()).mean()
            policy_grad = (torch.cat(self.particles[i].saved_log_probs[:-1])*advantages.detach()).sum()
            
            policy_grad.backward()
            
            # Vectorize parameters and PGs
            vec_param, vec_policy_grad = parameters_to_vector(list(self.particles[i].actor.parameters()), both=True)

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
            vector_to_parameters(grad_theta[i], list(self.particles[i].actor.parameters()), grad=True)

        for i in range(self.n_particles):
            critic_losses[i].backward()

        for i in range(self.n_particles):
            self.optimizers[i].step()


#%% Implementation (ADR algorithm)
if __name__ == '__main__':
    
    n_particles=3 #10 
    temp=0.0001 #temperature
    lr_svpg=0.003 #0.0003
    gamma_svpg=0.99
    h_svpg=64 #100
    svpg_modes=[1,2,3] #how to calculate kernel gradient #1: original implementation; 2 & 3: other variants
    svpg_mode=svpg_modes[0]
    xp_types=["peak","valley"] #experiment types
    xp_type=xp_types[0]
    T_svpg=5 #50 #svpg rollout length
    delta_max = 0.5 #0.05 #maximum allowable change to svpg states (i.e. upper bound on the svpg action)
    H_svpg = 5 #svpg horizon (how often the particles are reset)
    rewards_scale=1.

    #set_seed(seed)
    
    env_names=['halfcheetah_custom_norm-v1','halfcheetah_custom_rand-v1','lunarlander_custom_820_rand-v0','cartpole_custom-v1']
    env_name=env_names[-2]
    env=gym.make(env_name)
    T_env=env._max_episode_steps #task horizon / max env timesteps
    ds=env.observation_space.shape[0] #state dims
    da=env.action_space.shape[0] #action dims
    dr=env.unwrapped.randomization_space.shape[0]
    n_workers=n_particles
    
        
    svpg = SVPG(n_particles, h_svpg, lr_svpg, temp, svpg_mode, gamma_svpg, T_svpg, dr, delta_max, H_svpg)
    
    T_eps=3000
    plot_tr_rewards_mean=[]
    sampled_regions = [[] for _ in range(dr)]
    rand_step=0.02 #for discretizing the sampled regions plot
    common_name="_svpg_dr_c_"
    t_eps=0
    
    with tqdm.tqdm(total=T_eps) as pbar:
        while t_eps < T_eps:
            
            #collect rollout batch with svpg particles (storing values)
            simulation_instances, next_instances = svpg.step()
            
            #calculate deterministic reward
            simulation_instances_mask = np.concatenate([simulation_instances[:,1:,0],next_instances],1)
            rewards = np.ones_like(simulation_instances_mask,dtype=np.float32) 
            if xp_type =="peak":
                rewards[((simulation_instances_mask<=0.10).astype(int) + (simulation_instances_mask>=0.30).astype(int)).astype(bool)]=-1.
            elif xp_type=="valley":
                rewards[((simulation_instances_mask>=0.10).astype(int) * (simulation_instances_mask<=0.30).astype(int)).astype(bool)]=-1.
            
            rewards = rewards * rewards_scale
            
            mean_rewards=rewards.sum(-1).mean()
            plot_tr_rewards_mean.append(mean_rewards)
            
            #train svpg 
            svpg.train(rewards)
            
            #plot sampled regions
            if t_eps % 50 == 0:
                fig, ax = plt.subplots(1, 2, figsize=(16, 8))
                for dim in range(dr):
                    dim_name=env.unwrapped.dimensions[dim].name
                    low=0
                    high=1
                    x=np.arange(low,high+rand_step,rand_step)
                    
                    scaled_instances=low + (high-low) * simulation_instances[:, :, dim]
                    sampled_regions[dim]=np.concatenate([sampled_regions[dim],scaled_instances.flatten()])
                      
                    title=f"Sampled Regions for Randomization Dim = {dim_name} {env.rand} at Episode = {t_eps}"
                    ax[0].grid(1)
                    ax[0].hist(sampled_regions[dim][-500:], bins=np.arange(min(x),max(x)+2*rand_step,rand_step), histtype='barstacked')
                    ax[0].set_xlim(min(x), max(x)+rand_step)
                    ax[0].set_title(title)
                    
                    with torch.no_grad():
                        x = torch.linspace(0, 1.0, 100).float().unsqueeze(1).to(device)
                        for i, particle in enumerate(svpg.particles):
                            _, value = particle(x)
                            ax[1].plot(x, value.numpy(), label='Particle: {}'.format(i))
                    ax[1].legend()
                    #plt.show()
                    plt.savefig(f'plots/sampled_regions_dim_{dim_name}_{env.rand}{common_name}{T_eps}.png')
                    plt.clf()
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
