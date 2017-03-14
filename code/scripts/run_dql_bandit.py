
from algorithms import DQLearning, Q_learning
from environnements.MultiarmedBandit import MultiarmBandit
from utils import get_fixed_value_func,get_scheduled_value_func
import numpy as np

num_episodes = 1
max_iter = 8000
env = MultiarmBandit(1)

epsilon_func = get_fixed_value_func(0.2)

alpha_func = get_fixed_value_func(0.05)

qvalue, all_rewards, _, softmax_values = DQLearning(env, alpha_func=alpha_func,epsilon_func=epsilon_func,num_episodes=num_episodes, max_iter=max_iter, monitor_softmax_values=True)
print(qvalue)
print(all_rewards.shape)
print(np.average(all_rewards[-2000:]))
line_information = [("DQL", all_rewards,softmax_values)]

epsilon_func = get_scheduled_value_func([0, 13000],[0.3,0.00],"it_count")

qvalue, rewards, _, softmax_values = Q_learning(env, alpha_func=alpha_func, epsilon_func=epsilon_func, num_episodes=num_episodes, max_iter=max_iter,monitor_softmax_values=True)

print(qvalue)
line_information.append(("QL", rewards,softmax_values))
print(np.average(rewards[-2000:]))
print(env.plot_rewards(line_information,80))