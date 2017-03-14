from numpy.random import beta, poisson, binomial, normal, uniform, chisquare
import numpy as np
from utils import movingaverage
from operator import itemgetter
from matplotlib import pyplot as plt

def get_beta_func(a,b):
    def f():
        return beta(a,b)
    return f, float(a)/(a+b)

def get_binomial_func(p,r=1):
    def f():
        return r*binomial(1,p)
    mean = p*r
    return f, mean
    
def get_gaussian_func(mu,std):
    def f():
        return normal(mu,std)
    return f, mu

def get_poisson_func(lam):
    def f():
        return poisson(lam)
    return f, lam
    
def get_chisquare_func(df):
    def f():
        return chisquare(df)
    return f, df

def get_uniform_func(a,b):
    def f():
        return uniform(a,b)
    return f, (a+b)/2.0

class MultiarmBandit():
    def __init__(self, problem_id=1):
        self.current_state = 0
        self.num_states = 1
        if problem_id==1:
            self.options = [get_uniform_func(-0.5,0.5),get_chisquare_func(0.3), get_binomial_func(0.5,1),get_gaussian_func(0.2,1.5),get_beta_func(0.9,0.6)]
            self.option_label = ["uniform [-0.5,0.5] ", "chisquare df=0.3", "binomial p=0.5", "N(0.2,1.5)", "beta(0.9,0.6)"]
            self.num_actions = len(self.options)
            self.options_func = list(map(itemgetter(0), self.options))
            options_mean_values = list(map(itemgetter(1),self.options))
            print("Options mean values : "+str(options_mean_values))
            best_option_index = np.argmax(options_mean_values)
            best_mean_value = options_mean_values[best_option_index]
            print('best mean value %f, option %i'% (best_mean_value,best_option_index))
        elif problem_id==2:
            self.options = [get_gaussian_func(0, 0.25), get_gaussian_func(0.2, 0.25), get_gaussian_func(0.4, 0.25),
                            get_gaussian_func(0.6, 0.25),get_gaussian_func(0.8, 0.25),get_gaussian_func(1, 0.25),
                            get_gaussian_func(1.2, 0.25)]


            self.option_label = ["N(0,0.25)", "N(0.2,0.25)", "N(0.4,0.25)", "N(0.6,0.25)", "N(0.8,0.25)", "N(1,0.25)", "N(1.2,0.25)"]

#            self.options = [get_gaussian_func(0.1, 0.5), get_gaussian_func(0, 0.8), get_gaussian_func(0.30, 0.3),
#                            get_gaussian_func(0.2, 0.3)]
#            self.option_label = ["N(0.1,0.5)", "N(0,0.8)", "N(0.3,0.3)", "N(0.2,0.3)"]
            self.num_actions = len(self.options)
            self.options_func = list(map(itemgetter(0), self.options))
            options_mean_values = list(map(itemgetter(1), self.options))
            print("Options mean values : " + str(options_mean_values))
            best_option_index = np.argmax(options_mean_values)
            best_mean_value = options_mean_values[best_option_index]
            print('best mean value %f, option %i' % (best_mean_value, best_option_index))

        else :
            raise NotImplemented()

    def restart(self):
        pass

    def step(self, action):
        reward = self.options_func[action]()
        return 0, reward

    def is_terminal_state(self):
        """No terminal state is ever reached"""
        return False

    def plot_rewards(self, line_informations, avg_window_size=None):
        """
        :param line_informations: a list of 3 tuples (label, rewards, softmax_values)
        :param avg_window_size:
        :return:
        """
        nb_row = len(line_informations) + 1

        ax1 = plt.subplot(nb_row,1,nb_row)
        ax1.set_title("reward in fct. of iteration")


        for i,(label, rewards, softmax_values) in enumerate(line_informations):

            ax2 = plt.subplot(nb_row,1,i+1)
            ax2.set_title("%s - policy in fct. of iteration"%label)

            if rewards.shape[0] == 1:
                rewards = rewards.flatten()
                xpoints = np.array(range(rewards.size))
                for j, s in enumerate(softmax_values.T):
                    ax2.plot(xpoints,s.flatten(),label=self.option_label[j])

                if i == 0 :
                    legend = ax2.legend(loc='upper right', shadow=True)

                if not (avg_window_size is None):
                    rewards =movingaverage(rewards,avg_window_size)
                    xpoints = xpoints[avg_window_size-1:]

                ax1.plot(xpoints,rewards,label=label)

                legend = ax1.legend(loc='lower right', shadow=True)
            else :
                raise NotImplementedError()

        plt.show()


