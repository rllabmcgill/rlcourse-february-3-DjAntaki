import numpy as np
from matplotlib import pyplot as plt

def plot_rewards(l,moving_avg_window_size=None):
    """
    :param l: a list of 3-tuple where the first element is the array representing the total reward by episode,
    the second is an array representing the number of iteration taken to finish the every episode and the third is the label associated with the algorithm.
    """

    ax1 = plt.subplot(2,1,1)
    ax2 = plt.subplot(2,1,2)
    ax1.set_title("Reward in fct. of episode")
    ax2.set_title("Reward in fct of iteration completed")



    for reward, iter_taken, label in l:
        xpoints = np.array(range(len(reward)))
        if not (moving_avg_window_size is None):
            r = movingaverage(reward, moving_avg_window_size)
            xpoints = xpoints[moving_avg_window_size - 1:]
        else :
            r = reward

        ax1.plot(xpoints, r, label=label)

        sum_value,xpoints2 = 0,[]
        for i in iter_taken:
            sum_value += i
            xpoints2.append(sum_value)

        ax2.plot(xpoints2, reward,label=label)

    legend = ax1.legend(loc='lower right', shadow=True)
    legend = ax2.legend(loc='lower right', shadow=True)

    plt.show()

def movingaverage(values, window):
    #taken from https://gordoncluster.wordpress.com/2014/02/13/python-numpy-how-to-generate-moving-averages-efficiently-part-2/
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma



#
#

def softmax(x,temp=1):
    e_x = np.exp(x/temp)
    return e_x / e_x.sum(axis=0)


def sample(weights):
    return np.random.choice(range(len(weights)),p=weights)

def egreedy_sample(policy, epsilon=1e-2):
    """ returns argmax with prob (1-epsilon), else returns a random index"""
    if np.random.binomial(1,1-epsilon) == 1:
        return np.argmax(policy)
    else :
        return np.random.choice(range(len(policy)))

#
# functions
#

def get_fixed_value_func(v):
    def get_value(*args, **kwargs):
        return v
    return get_value

def get_scheduled_value_func(times,values,key=None):
    """ times : the scheduled time at which the returned value change.
        values : the values to be returned
        key : if none then looks at first argumen

        TODO : less sketchy code"""

    assert sorted(times) == times
    global sched_value_index
    sched_value_index = 0
    times_len = len(times)
    def get_value(*args, **kwargs):
        global sched_value_index
        if key is None:
            t = args[0]
        else :
            t = kwargs[key]
        if sched_value_index < times_len -1 and t >= times[sched_value_index+1]:
            sched_value_index += 1

        return values[sched_value_index]
    return get_value

#def get_linear_decrease_value_func(ini,)