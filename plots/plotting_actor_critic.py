import os

import matplotlib.pyplot as plt
import numpy as np

# Get the list of files in the directory
data_dir = './data4plot'
files = os.listdir(data_dir)

# Create a figure and axis
fig, ax = plt.subplots()

# Plot each file as a separate line
for file in files:
    if 'actor-critic' in file:
        file_path = os.path.join(data_dir, file)
        data = np.load(file_path)
        ks = np.arange(35)*100
        avs = np.mean(data, axis=0)
        maxs = np.max(data, axis=0)
        mins = np.min(data, axis=0)
        plt.fill_between(ks, mins, maxs, alpha=0.1)
        split_file = (file.split('.')[0] + '.' + file.split('.')[1]).split('_')
        # remove the first element of the list
        split_file.pop(0)
        # combine split_file with ","
        label = ",".join(split_file)
        plt.plot(ks, avs, '-o', markersize=1, label=label)

        plt.xlabel('Episode', fontsize = 12)
        plt.ylabel('Return', fontsize = 12)
        # ax.plot(data, label=file)

# Add legend
ax.legend()
# add title
plt.title("different varients of actor-critic", fontsize = 15)
# Use .fillbetween method to make the plot nicer
# ...
#save the plot to ./plots
plt.savefig('./plots/actor_critic.png')
# Show the plot
# plt.show()