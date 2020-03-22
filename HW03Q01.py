import matplotlib.pyplot as plt
import numpy as np
import argparse
import random

import os
import sys

from time import time
from datetime import datetime

# constants
#ARMS = 10
RUNS = 1
STEPS_PER_RUN = 1000
ALPHA = 0.01
CHOOSE_IMPLEMENTATION = "one_agent"
#TRAINING_STEPS = 10
#TESTING_STEPS = 5
seed_count = 16


NOW = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())

# #############################################################################
#
# Parser
#
# #############################################################################


def get_arguments():
    def _str_to_bool(s):
        '''Convert string to boolean (in argparse context)'''
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a '
                             'boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]

    parser = argparse.ArgumentParser(description='Implementing Baird Counterexample.')
    parser.add_argument('-s', '--steps', type=int, default=STEPS_PER_RUN,
                        help='Number of steps in each run. One run step is '
                        'the ensemble of training steps and testing steps. '
                        'Default: ' + str(STEPS_PER_RUN))
    parser.add_argument('-alpha', '--alpha', type=int, default=ALPHA,
                        help='learning rate'
                        'Default: ' + str(ALPHA))
    parser.add_argument('-choose_implementation', '--choose_implementation', type=str, default=CHOOSE_IMPLEMENTATION,
                        help='choose if you do 1 run, 50 runs, or variance for 50 runs'
                        'Default: ' + CHOOSE_IMPLEMENTATION)

    return parser.parse_args()

# #############################################################################
#
# Plotting
#
# #############################################################################

def plot_all_variances(data):
    '''Creates the two required plots: cumulative_reward and number of timesteps
        per episode.

    data: data of shape(nb_runs, steps, 8)'''

    fig, axs = plt.subplots(nrows=3, ncols=3,
                            sharey=True,
                            figsize=(12,14))

    for id_ax in range(8):
        label = "$w_{}$".format(str(id_ax+1))
        color = "C" + str(id_ax+1)
        data_one_var  = data[:,:,id_ax]
        plot_line_variance(axs, id_ax, data_one_var, label, color, axis=0, delta=1)

    plt.show()

def plot_line_variance(axs, id_ax, data_one_var, label, color, axis=0, delta=1):
    '''Plots the average data for each time step and draws a cloud
    of the standard deviation around the average.
    Input:
    ax      : axis object where the plot will be drawn
    data    : data of shape (nb_runs, steps)
    color   : the color to be used
    delta   : (optional) scaling of the standard deviation around the average
              if ommitted, delta = 1.'''

    avg = np.average(data_one_var, axis)
    std = np.std(data_one_var, axis)
    x_values = list(range(1,data_one_var.shape[1]+1))

    # ax.plot(avg + delta * std, color + '--', linewidth=0.5)
    # ax.plot(avg - delta * std, color + '--', linewidth=0.5)
    #fig, ax = plt.subplots(nrows=1, ncols=1,
                            #constrained_layout=True,
                            #sharey=True,
                            #figsize=(5,5))
    ax = axs[id_ax//len(axs[0]), id_ax % len(axs[0])]
    ax.fill_between(x_values,
                    avg + delta * std,
                    avg - delta * std,
                    facecolor=color,
                    alpha=0.2)
    ax.set_xlabel('Steps')
    #ax.set_ylabel('mean and variance of w' + str(id_ax + 1))
    ax.set_title('mean and variance of $w_{}$'.format(str(id_ax + 1)))
    #ax.set_xlim([0, 1.0])
    #ax.set_ylim([-0.2, 1.0])
    ax.plot(x_values, avg, label=label, color=color, marker='.')
    #plt.show()


def plot_coefficients_w(ws):
    aver_ws = np.mean(ws, axis = 0)
    x_range = list(range(aver_ws.shape[0]))
    for pos_w in range(aver_ws.shape[1]):
        plt.plot(x_range, aver_ws[:,pos_w], label="$w_{}$".format(pos_w+1))
    plt.xlabel('Steps')
    # Set the y axis label of the current axis.
    #plt.ylabel('y - axis')
    # Set a title of the current axes.
    plt.title('Semi-gradient Off-policy TD')
    # show a legend on the plot
    plt.legend()
    # Display a figure.
    plt.show()


# #############################################################################
#
# Helper functions
#
# #############################################################################

def softmax(x):
    '''Softmax implementation for a vector x.'''

    # subtract max for numerical stability
    # (does not change result because of identity softmax(x) = softmax(x + c))
    z = x - max(x)

    return np.exp(z) / np.sum(np.exp(z), axis=0)


def random_argmax(vector):
    '''Select argmax at random... not just first one.'''

    index = np.random.choice(np.where(vector == vector.max())[0])

    return index

# #############################################################################
#
# Agent performing semi-gradient TD(0) for the Baird's counterexample
#
# #############################################################################

class TD_Zero_Agent_Baird_Counterexample():
    def __init__(self,args, nb_runs, gamma = 0.99):
        self.args = args
        self.alpha = self.args.alpha
        self.gamma = gamma
        self.nb_runs = nb_runs
        self.ws = np.zeros((self.nb_runs, self.args.steps+1, 8))
        for run in range(self.ws.shape[0]):
            self.ws[run] = np.array([1,1,1,1,1,1,10,1])
        self.features = np.zeros((7,8))
        self.features[0,0]=2
        self.features[0,7] = 1
        self.features[1, 1] = 2
        self.features[1, 7] = 1
        self.features[2, 2] = 2
        self.features[2, 7] = 1
        self.features[3, 3] = 2
        self.features[3, 7] = 1
        self.features[4, 4] = 2
        self.features[4, 7] = 1
        self.features[5, 5] = 2
        self.features[5, 7] = 1
        self.features[6, 6] = 1
        self.features[6, 7] = 2
        self.current_state = None

    def train_all_runs(self):
        for run_id in range(0, self.nb_runs):
            global seed_count
            np.random.seed(seed_count)
            seed_count += 1
            self.current_state = np.random.choice(7)
            self.semi_gradient_one_run(run_id)

    def semi_gradient_one_run(self, run_id):
        for step in range(1,self.args.steps+1):
            self.semi_gradient_one_step(run_id, step)

    def semi_gradient_one_step(self, run_id, step):
        old_state = self.current_state
        new_state = np.random.choice(7)
        w = self.ws[run_id, step-1]
        delta = self.gamma * np.sum(self.features[new_state] * w) - \
                np.sum(self.features[old_state] * w)
        ratio = (7*(new_state==6))
        self.ws[run_id, step] = self.ws[run_id, step-1] + self.alpha * ratio * delta * self.features[old_state]
        self.current_state = new_state

# #############################################################################
#
# Main
#
# #############################################################################

def train_one_agent(args):
    # parses command line arguments
    #global seed_count
    #print(seed_count)
    agent = TD_Zero_Agent_Baird_Counterexample(args, nb_runs=1)
    agent.train_all_runs()
    #print(np.mean(agent.ws[0:,-1], axis = 0))
    #print(agent.ws[0:, -1])
    plot_coefficients_w(agent.ws)
    """
    In the previous plot, you can observe the curves for all the parameters $w_1$, $w_2$, $w_3$, $w_4$, $w_5$,
    $w_6$, \$w_7$, $w_8$. The parameters grow very similarly to Figure 11.2 of the RL book of Sutton and Barto. 
    The parameter $w_7$ barely decreases. The remaining parameters are indistiguishable in the algorithm, and they grow 
    very similarly between the curves of $w_7$ and $w_8$. It is clear from this plot that 7 of the parameters diverge. 
    As in the book, this shows that the combination of function approximation, bootstrapping and off-policy training 
    (i.e. the deadly trial) can diverge, even in the linear case and when $\\alpha=0.01$ is very small.
    """

def train_agents_50(args):
    agents_50 = TD_Zero_Agent_Baird_Counterexample(args, nb_runs=50)
    agents_50.train_all_runs()
    plot_coefficients_w(agents_50.ws)

    """
    In the previous plot, we did the same experiment as in the first plot but we averaged 50 runs instead of a single run. 
    We thought we didn't need to do that, but we decided to include it anyway. The curves of the parameters
    $w_1$, $w_2$, $w_3$, $w_4$, $w_5$ and $w_6$ are almost indistinguishable, as expected.
    """


def agents_50_variance(args):
    agents_50 = TD_Zero_Agent_Baird_Counterexample(args, nb_runs=50)
    agents_50.train_all_runs()
    plot_all_variances(agents_50.ws)

    """
    Just as the previous comments, we were not sure if we had to run the algorithm for multiple runs. We did it 
    anyway and the variance seems proportional to the value on the y-axis. More specifically, the variance is the 
    highest for the parameter $w_8$ which is also the parameter that grows the fastest. The variance is close to 
    0 for $w_7$ and the variance is intermediate for all the other parameters.
    """

def main():
    args = get_arguments()
    #print(args.choose_implementation)
    assert args.choose_implementation in ["one_agent", "agents_50", "agents_50_variance"]
    if args.choose_implementation =="one_agent":
        train_one_agent(args)
    elif args.choose_implementation == "agents_50":
        train_agents_50(args)
    elif args.choose_implementation == "agents_50_variance":
        agents_50_variance(args)

if __name__ == '__main__':
    main()