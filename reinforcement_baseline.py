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

def train_REINFORCE_baseline(env, policy_net, policy_optimizer, value_net, value_optimizer, gamma=0.99):
    """
    Trains the policy network on a single episode using REINFORCE with baseline
    """

    # Generate an episode with the current policy network
    states, actions, rewards, log_probs = generate_episode(env, policy_net)
    T = len(states)
    
    # Create tensors
    states = np.vstack(states).astype(float)
    states = torch.FloatTensor(states).to(device)
    actions = torch.LongTensor(actions).to(device).view(-1,1)
    rewards = torch.FloatTensor(rewards).to(device).view(-1,1)

    # Compute total discounted return at each time step
    Gs = []
    G = 0
    for t in range(T-1,-1,-1): # iterate in backward order to make the computation easier
        G = rewards[t] + gamma*G
        Gs.insert(0,G)
    Gs = torch.tensor(Gs).view(-1,1)
    
    # Compute the advantage
    state_vals = value_net(states).to(device)
    with torch.no_grad():
        advantages = Gs - state_vals
        
    # Update policy network weights
    L_theta = torch.stack([-log_prob * advantage for log_prob, advantage in zip(log_probs, advantages)]).sum() / T
    policy_optimizer.zero_grad()
    L_theta.backward()
    policy_optimizer.step() 
    
    # Update value network weights
    loss_fn = nn.MSELoss()
    value_loss = loss_fn(state_vals, Gs)
    value_optimizer.zero_grad()
    value_loss.backward()
    value_optimizer.step() 


env_name = 'CartPole-v1'
num_episodes = 3500
policy_lr = 5e-4
baseline_lr = 5e-4
# num_seeds = 5 # fit model with 5 different seeds and plot average performance of 5 seeds
num_seeds = 3 # fit model with 5 different seeds and plot average performance of 5 seeds
l = num_episodes//100 # use to create x label for plot
returns = np.zeros((num_seeds, l)) # dim: (5 ,35)
gamma = 0.99 # discount factor

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
    value_net_optimizer = optim.Adam(value_net.parameters(), lr=baseline_lr)
    
    for m in range(num_episodes):
        # Train networks with REINFORCE with baseline
        train_REINFORCE_baseline(env, policy_net, policy_net_optimizer, value_net, value_net_optimizer, gamma=gamma)
        if m % 100 == 0:
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
np.save(os.path.join(directory, "returns_reinforcement_baseline.npy"), returns)

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
