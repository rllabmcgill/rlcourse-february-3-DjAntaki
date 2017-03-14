from utils import *

def Q_learning(env, gamma=0.9, alpha_func=get_fixed_value_func(0.05), num_episodes=1000, max_iter=750, epsilon_func=get_fixed_value_func(0.05), softmax_temp=2, verbose=False, monitor_softmax_values=False):
    assert (gamma > 0 and gamma < 1)

    qvalue = np.ones((env.num_states, env.num_actions))
    all_rewards = np.zeros((num_episodes, max_iter))
    final_iter_count = np.zeros((num_episodes,))

    if monitor_softmax_values :
        softmax_values = np.zeros((num_episodes, max_iter, env.num_actions))
    #action_taken = np.zeros((num_episodes, max_iter), dtype=np.int)

    for ep_count in range(num_episodes):
        it_count = -1
        env.restart()
        while True:
            it_count += 1
            current_state = env.current_state
            policy = softmax(qvalue[current_state],softmax_temp)
            action = egreedy_sample(policy,epsilon=epsilon_func(ep_count=ep_count,it_count=it_count))

            next_state, reward = env.step(action)

            all_rewards[ep_count,it_count] = reward
            if monitor_softmax_values :
                softmax_values[ep_count,it_count] = policy
#            action_taken[ep_count,it_count] = action

            qvalue[current_state, action] = qvalue[current_state, action] + \
                                            alpha_func(ep_count=ep_count,it_count=it_count) * (
                                            reward + gamma * np.max(qvalue[next_state]) - qvalue[current_state, action])

            if env.is_terminal_state() or it_count +1 == max_iter:
                if verbose :
                    print('Episode done. Took ' + str(it_count) + " iteration.")
                    print(qvalue)
                break

        final_iter_count[ep_count] = it_count

    if monitor_softmax_values :
        return qvalue, all_rewards, final_iter_count, softmax_values
    return qvalue, all_rewards, final_iter_count


def DQLearning(env, gamma=0.9, alpha_func=get_fixed_value_func(0.05), epsilon_func=get_fixed_value_func(0.05), softmax_temp=2,num_episodes=1, monitor_softmax_values=False, max_iter=1000):
    Qa, Qb = np.ones((env.num_states, env.num_actions)), np.ones((env.num_states, env.num_actions))
    ep_count = -1

    all_rewards = np.zeros((num_episodes, max_iter))
    final_iter_count = np.zeros((num_episodes,))

    if monitor_softmax_values :
        softmax_values = np.zeros((num_episodes, max_iter, env.num_actions))

    while True:
        env.restart()
        current_state = env.current_state
        ep_count += 1
        it_count = -1

        while True:
            it_count += 1

            probs = softmax((Qa + Qb)[current_state],softmax_temp)
#            probs = softmax(((Qa + Qb)/2.0)[current_state],softmax_temp)
            next_action = egreedy_sample(probs, epsilon_func(ep_count=ep_count,it_count=it_count))
            next_state, reward = env.step(next_action)
            all_rewards[ep_count,it_count] = reward
            if monitor_softmax_values :
                softmax_values[ep_count,it_count] = probs

            update_a = np.random.binomial(1,p=0.5)
            if update_a:
                a_star = np.argmax(Qa[next_state])
                Qa[current_state, next_action] = Qa[current_state, next_action] + alpha_func(ep_count=ep_count,it_count=it_count) * \
                                                                                  (reward + gamma * Qb[
                                                                                      next_state, a_star] - Qa[
                                                                                       current_state, next_action])
            else:
                b_star = np.argmax(Qb[next_state])
                Qb[current_state, next_action] = Qb[current_state, next_action] + alpha_func(ep_count=ep_count,it_count=it_count) * \
                                                                                  (reward + gamma * Qa[
                                                                                      next_state, b_star] - Qb[
                                                                                       current_state, next_action])

            if it_count +1 == max_iter or env.is_terminal_state():
                final_iter_count[ep_count] = it_count
                break

            current_state = next_state

        if ep_count +1 == num_episodes:
            break

#    qmatrix = (Qa + Qb)/2.0
    qmatrix = (Qa + Qb)
    if monitor_softmax_values:
        return qmatrix, all_rewards, final_iter_count, softmax_values
    return qmatrix, all_rewards, final_iter_count