# Import packages
import sys
import os

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

import tqdm

import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn

from IPython.display import clear_output
from IPython import display

# check and use GPU if available if not use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CODE from another notebook
class NeuralNet(torch.nn.Module):
    def __init__(self, input_size, output_size, activation, layers=[32,32,16]):
        super().__init__()

        # Define layers with ReLU activation
        self.linear1 = torch.nn.Linear(input_size, layers[0])
        self.activation1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(layers[0], layers[1])
        self.activation2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(layers[1], layers[2])
        self.activation3 = torch.nn.ReLU()

        self.output_layer = torch.nn.Linear(layers[2], output_size)
        self.output_activation = activation

        # Initialization using Xavier normal (a popular technique for initializing weights in NNs)
        torch.nn.init.xavier_normal_(self.linear1.weight)
        torch.nn.init.xavier_normal_(self.linear2.weight)
        torch.nn.init.xavier_normal_(self.linear3.weight)
        torch.nn.init.xavier_normal_(self.output_layer.weight)

    def forward(self, inputs):
        # Forward pass through the layers
        x = self.activation1(self.linear1(inputs))
        x = self.activation2(self.linear2(x))
        x = self.activation3(self.linear3(x))
        x = self.output_activation(self.output_layer(x))
        return x


def generate_single_episode(env, policy_net):
    """
    Generates an episode by executing the current policy in the given env
    """
    states = []
    actions = []
    rewards = []
    log_probs = []
    max_t = 1000 # max horizon within one episode
    state, _ = env.reset()
        
    for t in range(max_t):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = policy_net.forward(Variable(state)) # get each action choice probability with the current policy network
        action = np.random.choice(env.action_space.n, p=np.squeeze(probs.detach().numpy())) # probablistic
        # action = np.argmax(probs.detach().numpy()) # greedy
        
        # compute the log_prob to use this in parameter update
        log_prob = torch.log(probs.squeeze(0)[action])
        
        # append values
        states.append(state)
        actions.append(action)
        log_probs.append(log_prob)
        
        # take a selected action
        state, reward, terminated, truncated, _ = env.step(action)
        rewards.append(reward)

        if terminated | truncated:
            break
            
    return states, actions, rewards, log_probs


def evaluate_policy(env, policy_net):
    """
    Compute accumulative trajectory reward
    """
    states, actions, rewards, log_probs = generate_single_episode(env, policy_net)
    return np.sum(rewards)

def compute_Gs_per_episode(batch_rews, gamma):
    # The rewards-to-go (rtg) per episode per batch to return
    batch_rtgs = []
    
    # Iterate through each episode backwards to maintain same order in batch_rtgs
    for ep_rews in reversed(batch_rews):
        discounted_reward = 0 # Discounted reward so far
        
        for rew in reversed(ep_rews):
            discounted_reward = rew + discounted_reward * gamma
            batch_rtgs.insert(0, discounted_reward)
            
    # Convert the rewards-to-go into a tensor
    batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)

    return batch_rtgs


def generate_multiple_episodes(env, policy_net, max_batch_size=500):
    """
    Generates an episode by executing the current policy in the given env
    """
    states = []
    actions = []
    rewards = []
    log_probs = []
    max_t = 1000 # max horizon within one episode
    i = 0
    
    while i < max_batch_size:
        state, _ = env.reset()
        reward_per_epi = []
        for t in range(max_t):
            state = torch.from_numpy(state).float().unsqueeze(0)
            probs = policy_net.forward(Variable(state)) # get each action choice probability with the current policy network
            action = np.random.choice(env.action_space.n, p=np.squeeze(probs.detach().numpy())) # probablistic
            # action = np.argmax(probs.detach().numpy()) # greedy
            
            # compute the log_prob to use this in parameter update
            log_prob = torch.log(probs.squeeze(0)[action])
            
            # append values
            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            
            # take a selected action
            state, reward, terminated, truncated, _ = env.step(action)
            reward_per_epi.append(reward)
            
            i += 1

            if terminated | truncated:
                break
        rewards.append(reward_per_epi)
        
    return states, actions, rewards, log_probs


def train_PPO_multi_epi(env, policy_net, policy_optimizer, value_net, value_optimizer, num_epochs, clip_val=0.2, gamma=0.99, max_batch_size=100, entropy_coef=0.1, normalize_ad=True, add_entropy=True):
    """
    Trains the policy network on a single episode using REINFORCE with baseline
    """

    # Generate an episode with the current policy network
    states, actions, rewards, log_probs = generate_multiple_episodes(env, policy_net, max_batch_size=max_batch_size)
    T = len(states)
    
    # Create tensors
    states = np.vstack(states).astype(np.float64)
    states = torch.FloatTensor(states).to(device)
    actions = torch.LongTensor(actions).to(device).view(-1,1)
    log_probs = torch.FloatTensor(log_probs).to(device).view(-1,1)

    # Compute total discounted return at each time step in each episode
    Gs = compute_Gs_per_episode(rewards, gamma).view(-1,1)
    
    # Compute the advantage
    state_vals = value_net(states).to(device)
    with torch.no_grad():
        A_k = Gs - state_vals
    if normalize_ad:
        A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10) # Normalize advantages
        
    for _ in range(num_epochs):
        V = value_net(states).to(device)
        
        # Calculate probability of each action under the updated policy
        probs = policy_net.forward(states).to(device)
                
        # compute the log_prob to use it in parameter update
        curr_log_probs = torch.log(torch.gather(probs, 1, actions)) # Use torch.gather(A,1,B) to select columns from A based on indices in B
        
        # Calculate ratios r(theta)
        ratios = torch.exp(curr_log_probs - log_probs)
        
        # Calculate two surrogate loss terms in cliped loss
        surr1 = ratios * A_k
        surr2 = torch.clamp(ratios, 1-clip_val, 1+clip_val) * A_k
        
        # Caluculate entropy
        entropy = 0
        if add_entropy:
            entropy = torch.distributions.Categorical(probs).entropy()
            entropy = torch.tensor([[e] for e in entropy])
        
        # Calculate clipped loss value
        actor_loss = (-torch.min(surr1, surr2) - entropy_coef * entropy).mean() # Need negative sign to run Gradient Ascent
        
        # Update policy network
        policy_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        policy_optimizer.step()
        
        # Update value net
        critic_loss = nn.MSELoss()(V, Gs)
        value_optimizer.zero_grad()
        critic_loss.backward()
        value_optimizer.step()        
        
    return policy_net, value_net


# Define parameter values
env_name = 'CartPole-v1'
num_train_ite = 3500
num_seeds = 5 # fit model with 5 different seeds and plot average performance of 5 seeds
num_epochs = 10 # how many times we iterate the entire training dataset passing through the training
eval_freq = 50 # run evaluation of policy at each eval_freq trials
eval_epi_index = num_train_ite//eval_freq # use to create x label for plot
returns = np.zeros((num_seeds, eval_epi_index))
gamma = 0.99 # discount factor
clip_val = 0.2 # hyperparameter epsilon in clip objective

# Create the environment.
env = gym.make(env_name)
nA = env.action_space.n
nS = 4

policy_lr = 5e-4 # policy network's learning rate 
baseline_lr = 1e-4
# Define parameter values
returns = np.zeros((num_seeds, eval_epi_index))
max_batch_size = 100
entropy_coef = 0.1
normalize_ad = True
add_entropy = True

for i in tqdm.tqdm(range(num_seeds)):
    reward_means = []

    # Define policy and value networks
    policy_net = NeuralNet(nS, nA, torch.nn.Softmax())
    policy_net_optimizer = optim.Adam(policy_net.parameters(), lr=policy_lr)
    value_net = NeuralNet(nS, 1, torch.nn.ReLU())
    value_net_optimizer = optim.Adam(value_net.parameters(), lr=baseline_lr)
    
    for m in range(num_train_ite):
        # Train networks with PPO
        policy_net, value_net = train_PPO_multi_epi(env, policy_net, policy_net_optimizer, value_net, value_net_optimizer, num_epochs, clip_val=clip_val, gamma=gamma, max_batch_size=max_batch_size, entropy_coef=entropy_coef, normalize_ad=normalize_ad, add_entropy=add_entropy)
        if m % eval_freq == 0:
            print("Episode: {}".format(m))
            G = np.zeros(20)
            for k in range(20):
                g = evaluate_policy(env, policy_net)
                G[k] = g

            reward_mean = G.mean()
            reward_sd = G.std()
            print("The avg. test reward for episode {0} is {1} with std of {2}.".format(m, reward_mean, reward_sd))
            reward_means.append(reward_mean)
    returns[i] = np.array(reward_means)

# Create the directory if it doesn't exist
directory = "./data4plot"
if not os.path.exists(directory):
    os.makedirs(directory)
# Save the returns array as a numpy file
np.save(os.path.join(directory, "returns_actor-critic_ppo_plus.npy"), returns)
# Plot the performance over iterations
# x = np.arange(eval_epi_index)*eval_freq
# avg_returns = np.mean(returns, axis=0)
# max_returns = np.max(returns, axis=0)
# min_returns = np.min(returns, axis=0)

# plt.fill_between(x, min_returns, max_returns, alpha=0.1)
# plt.plot(x, avg_returns, '-o', markersize=1)

# plt.xlabel('Episode', fontsize = 15)
# plt.ylabel('Return', fontsize = 15)

# plt.title("PPO Learning Curve", fontsize = 24)