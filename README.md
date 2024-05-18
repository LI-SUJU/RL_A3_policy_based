## Words for TA
#### If you're having trouble with the code on bright space, we highly recommend cloning our github repo, and to be fair, we won't commit new code after due.
## Words for everyone
#### how to use this repo:
- Run any .py files in RL_A3_policy_based, ./critic_lr, ./policy_lr. This will generate data which will be saved into ./data4plot. (At the end of every .py file as mentioned, there are some commented lines of code for plotting. Feel free to use them.)
- Go to ./plots, run any .py files in it, and it's expected to generate corresponding plots in this directory. Check them out, have fun!
- [Here](https://github.com/LI-SUJU/RL_A3_policy_based/blob/main/RL_A3_8_pages_with_reference.pdf) is a detailed report about this project.
#### Intro
In this project, we will implement policy-based RL in the environment provided by OpenAI’s Cartpole, which is a classic reinforcement learning problem that serves as a benchmark for various algorithms. In this environment, we aim to implement algorithms such as REINFORCE, Actor-Critic, and its variants. We will also use entropy regurization to encourage exploration by the learning agent, which is particularly useful in preventing the policy from prematurely converging to a suboptimal deterministic strategy. At last, we will conduct experiments to evaluate the performance of algorithms, interpret the results, discuss the limits and possible improvement in the future.
