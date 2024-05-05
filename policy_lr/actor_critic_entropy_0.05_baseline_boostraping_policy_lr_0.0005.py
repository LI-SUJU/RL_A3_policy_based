# Import packages
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # INFO and WARNING messages are not printed

import gym
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
import os
# set random seeds
torch.manual_seed(0)
np.random.seed(0)

# check and use GPU if available if not use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def generate_episode(env, policy_net):
    """
    Generates an episode by executing the current policy in the given env
    """
    states = []
    actions = []
    rewards = []
    log_probs = []
    max_t = 1000 # max horizon of one episode
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
    states, actions, rewards, log_probs = generate_episode(env, policy_net)
    return np.sum(rewards)

# Implementation
def train_actor_critic(env, policy_net, policy_optimizer, value_net, value_optimizer, gamma=0.99, n=100, entropy_coef=0.05):
    """
    Trains the policy network on a single episode using Actor Critic
    """

    # Generate an episode with the current policy network
    states, actions, rewards, log_probs = generate_episode(env, policy_net)
    T = len(states)
    
    # Create tensors
    states = np.vstack(states).astype(float)
    states = torch.FloatTensor(states).to(device)
    actions = torch.LongTensor(actions).to(device).view(-1,1)
    rewards = torch.FloatTensor(rewards).to(device).view(-1,1)
    # log_probs = torch.FloatTensor(log_probs).to(device).view(-1,1)

    # Initialize Gs as an empty list
    Gs = []

    # Loop through each time step
    for t in range(T):
        end_time = t + n if t + n < T else T  # Determine the end of the trajectory

        # Calculate the discounted return G for time step t
        G = 0
        for t_inner in range(t, end_time):
            G += gamma**(t_inner - t) * rewards[t_inner]
            
        # If end_time is within the trajectory, add the value function estimate
        if end_time < T:
            V_end = value_net.forward(states[end_time]).item()
            G += gamma**n * V_end

        # Append the calculated G to the list
        Gs.append(G)
        
    Gs = torch.tensor(Gs).view(-1,1)

    # Compute the advantage
    state_vals = value_net(states).to(device)
    with torch.no_grad():
        advantages = Gs - state_vals
        
    # Update policy network weights
    policy_loss = torch.stack([-log_prob * advantage for log_prob, advantage in zip(log_probs, advantages)]).sum() / T
    entropy_loss = torch.stack([torch.distributions.Categorical(probs).entropy() for probs in policy_net(states)]).sum() / T
    total_loss = policy_loss + entropy_coef * entropy_loss
    policy_optimizer.zero_grad()
    total_loss.backward()
    policy_optimizer.step() 
    
    # Update value network weights
    loss_fn = nn.MSELoss()
    value_loss = loss_fn(state_vals, Gs)
    value_optimizer.zero_grad()
    value_loss.backward()
    value_optimizer.step() 
# Define parameter values
env_name = 'CartPole-v1'
num_episodes = 3500
policy_lr = 5e-4
critic_lr = 5e-4
# num_seeds = 5 # fit model with 5 different seeds and plot average performance of 5 seeds
num_seeds = 3 # fit model with 5 different seeds and plot average performance of 5 seeds
l = num_episodes//100 # use to create x label for plot
returns = np.zeros((num_seeds, l)) # dim: (5 ,35)
gamma = 0.99 # discount factor
n = 100 # number of time step to use immediate reward

# Create the environment.
env = gym.make(env_name)
nA = env.action_space.n
nS = 4

for i in tqdm.tqdm(range(num_seeds)):
    reward_means = []

    # Define policy and value networks
    policy_net = NeuralNet(nS, nA, torch.nn.Softmax())
    policy_net_optimizer = optim.Adam(policy_net.parameters(), lr=policy_lr)
    value_net = NeuralNet(nS, 1, torch.nn.ReLU())
    value_net_optimizer = optim.Adam(value_net.parameters(), lr=critic_lr)
    
    for m in range(num_episodes):
        train_actor_critic(env, policy_net, policy_net_optimizer, value_net, value_net_optimizer, gamma=gamma, n=n)
        if m % 100 == 0:
            print("Episode: {}".format(m))
            G = np.zeros(20)
            for k in range(20):
                g = evaluate_policy(env, policy_net)
                G[k] = g

            reward_mean = G.mean()
            reward_sd = G.std()
            print("The avg. test reward for episode {0} is {1} with std. of {2}.".format(m, reward_mean, reward_sd))
            reward_means.append(reward_mean)
    returns[i] = np.array(reward_means)

# Create the directory if it doesn't exist
directory = "./data4plot/"
if not os.path.exists(directory):
    os.makedirs(directory)
# Save the returns array as a numpy file
np.save(os.path.join(directory, "returns_actor-critic_entropy-0.05_baseline_boostrapping_policy-lr-0.0005.npy"), returns)
# Plot the performance over iterations
# ks = np.arange(l)*100
# avs = np.mean(returns, axis=0)
# maxs = np.max(returns, axis=0)
# mins = np.min(returns, axis=0)

# plt.fill_between(ks, mins, maxs, alpha=0.1)
# plt.plot(ks, avs, '-o', markersize=1)

# plt.xlabel('Episode', fontsize = 15)
# plt.ylabel('Return', fontsize = 15)

# plt.title("REINFORCE Learning Curve", fontsize = 24)
# plt.show()
