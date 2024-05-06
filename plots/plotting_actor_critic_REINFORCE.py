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
    if 'actor-critic' in file or 'REINFORCE.npy' in file:
        if 'ppo' in file:
            continue
        elif '0.01' in file:
            continue
        elif '0.002' in file:
            continue
        else:
            file_path = os.path.join(data_dir, file)
            data = np.load(file_path)
            ks = np.arange(35)*100
            avs = np.mean(data, axis=0)
            maxs = np.max(data, axis=0)
            mins = np.min(data, axis=0)
            plt.fill_between(ks, mins, maxs, alpha=0.1)
        # decide if the file name has two "."
            is_two_dot = len(file.split('.')) == 3
            if is_two_dot:

                split_file = (file.split('.')[0] + '.' + file.split('.')[1]).split('_')
                # remove the first element of the list
                split_file.pop(0)
                # combine split_file with ","
                label = ", ".join(split_file)
                # plt.plot(ks, avs, '-o', markersize=1, label=label)

                plt.plot(ks, smooth(avs, 10), '-o', markersize=1, label=label)

                plt.xlabel('Episode', fontsize = 12)
                plt.ylabel('Return', fontsize = 12)
                # ax.plot(data, label=file)
            else:
                split_file = (file.split('.')[0]).split('_')
                # remove the first element of the list
                split_file.pop(0)
                label = ", ".join(split_file)
                # plt.plot(ks, avs, '-o', markersize=1, label=label)

                plt.plot(ks, smooth(avs, 10), '-o', markersize=1, label=label)

                plt.xlabel('Episode', fontsize = 12)
                plt.ylabel('Return', fontsize = 12)

# Add legend
ax.legend()
# add title
plt.title("Actor-Critic VS REINFORCE", fontsize = 15, y=1.05)
# add subtitle and make it under the title
plt.suptitle("policy net lr=0.0005, critic net lr=0.0005", fontsize = 10, y=0.92)
# Use .fillbetween method to make the plot nicer
# ...
#save the plot to ./plots
plt.savefig('./plots/actor_critic_REINFORCE.png')
# Show the plot
# plt.show()