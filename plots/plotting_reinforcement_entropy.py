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
    if 'baseline_entropy' in file:
        file_path = os.path.join(data_dir, file)
        data = np.load(file_path)
        ks = np.arange(35)*100
        avs = np.mean(data, axis=0)
        maxs = np.max(data, axis=0)
        mins = np.min(data, axis=0)
        plt.fill_between(ks, mins, maxs, alpha=0.1)
        split_file = (file.split('.')[0] + '.' + file.split('.')[1]).split('_')
        if len(split_file) == 2:
            label = split_file[1]
        else:
            label = split_file[3] + '_' + split_file[4]
        plt.plot(ks, avs, '-o', markersize=1, label=label)

        plt.xlabel('Episode', fontsize = 12)
        plt.ylabel('Return', fontsize = 12)
        # ax.plot(data, label=file)

# Add legend
ax.legend()
# add title
plt.title("REINFORCE-baseline with different entropy coefficients", fontsize = 15)
# Use .fillbetween method to make the plot nicer
# ...
#save the plot to ./plots
plt.savefig('./plots/reinforcement_entropy.png')
# Show the plot
# plt.show()