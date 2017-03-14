
from algorithms import DQLearning, Q_learning
from environnements.GridWorld import WindyCliffWorld
from utils import get_fixed_value_func,get_scheduled_value_func,plot_rewards
import numpy as np

num_episodes = 500
max_iter = 500
env = WindyCliffWorld()
softmax_temp=4

alpha_func = get_fixed_value_func(0.05)
epsilon_func = get_scheduled_value_func([0, 100,200,300],[0.3,0.2,0.1,0.05],"ep_count")


qmatrix, all_rewards, final_iter_count= DQLearning(env, num_episodes=num_episodes, alpha_func=alpha_func,max_iter=max_iter,softmax_temp=softmax_temp)
env.plot_qvalue(qmatrix,"Double Q-learning")
print(np.sum(all_rewards,axis=1))
print(final_iter_count)

epsilon_func = get_scheduled_value_func([0, 100,200,300],[0.3,0.2,0.1,0.05],"ep_count")

qvalue, rewards, iter_taken = Q_learning(env, num_episodes=num_episodes, alpha_func=alpha_func, max_iter=max_iter,softmax_temp=softmax_temp)
env.plot_qvalue(qvalue,"Q-learning")
print(np.sum(rewards,axis=1))
print(iter_taken)

plot_rewards([(np.sum(all_rewards,axis=1),final_iter_count,'DQL'),(np.sum(rewards,axis=1),iter_taken,'QL')])

