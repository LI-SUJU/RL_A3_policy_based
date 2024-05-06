import os

import matplotlib.pyplot as plt
import numpy as np

from plotHelper import smooth

# Get the list of files in the directory
data_dir = './data4plot'
files = os.listdir(data_dir)

# Create a figure and axis
fig, ax = plt.subplots()

# Increase the figure size
fig.set_size_inches(10, 6)
# Plot each file as a separate line
for file in files:
    if '0.05_baseline_boostrapping' in file:
        file_path = os.path.join(data_dir, file)
        data = np.load(file_path)
        ks = np.arange(35)*100
        avs = np.mean(data, axis=0)
        maxs = np.max(data, axis=0)
        mins = np.min(data, axis=0)
        plt.fill_between(ks, mins, maxs, alpha=0.1)
        label = "actor-critic, entropy-0.05, baseline, bootstrapping"
        plt.plot(ks, avs, '-o', markersize=1, label=label)

        plt.xlabel('Episode', fontsize = 12)
        plt.ylabel('Return', fontsize = 12)
        # ax.plot(data, label=file)
    elif 'ppo' in file:
        if 'ppo_plus' in file:
            continue
        else:
            file_path = os.path.join(data_dir, file)
            data = np.load(file_path)
            ks = np.arange(70)*100
            avs = np.mean(data, axis=0)
            maxs = np.max(data, axis=0)
            mins = np.min(data, axis=0)
            print(len(ks))
            print(len(avs))
            print(len(maxs))
            print(len(mins))
            plt.fill_between(ks, mins, maxs, alpha=0.1)
            label = "actor-critic, PPO"
            # plt.plot(ks, avs, '-o', markersize=1, label=label)
            plt.plot(ks, smooth(avs, 10), '-o', markersize=1, label=label)
            plt.xlabel('Episode', fontsize = 12)
            plt.ylabel('Return', fontsize = 12)
    

# Add legend
ax.legend()
# add title
plt.title("Performance of actor-critic with PPO", fontsize = 15)
# Use .fillbetween method to make the plot nicer
# ...
#save the plot to ./plots
plt.savefig('./plots/ppo.png')
# Show the plot
# plt.show()