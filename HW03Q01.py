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
RUNS = 100
STEPS_PER_RUN = 1000
#TRAINING_STEPS = 10
#TESTING_STEPS = 5
seed_count = 1

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
    parser.add_argument('-n', '--runs', type=int, default=RUNS,
                        help='Number of runs to be executed. Default: '
                        + str(RUNS))
    parser.add_argument('-s', '--steps', type=int, default=STEPS_PER_RUN,
                        help='Number of steps in each run. One run step is '
                        'the ensemble of training steps and testing steps. '
                        'Default: ' + str(STEPS_PER_RUN))

    return parser.parse_args()

# #############################################################################
#
# Plotting
#
# #############################################################################

def plot_line_variance(ax, data, gamma=1):
    '''Plots the average data for each time step and draws a cloud
    of the standard deviation around the average.

    ax:     axis object where the plot will be drawn
    data:   data of shape (num_trials, timesteps)
    gamma:  (optional) scaling of the standard deviation around the average
            if ommitted, gamma = 1.'''

    avg = np.average(data, axis=0)
    std = np.std(data, axis=0)

    # ax.plot(avg + gamma * std, 'r--', linewidth=0.5)
    # ax.plot(avg - gamma * std, 'r--', linewidth=0.5)
    ax.fill_between(range(len(avg)),
                    avg + gamma * std,
                    avg - gamma * std,
                    facecolor='red',
                    alpha=0.2)
    ax.plot(avg)


def plot4(title, training_return, training_regret, testing_reward, testing_regret):
    '''Creates the four required plots: average training return, training regret,
    testing policy reward and testing regret.'''

    fig, axs = plt.subplots(nrows=2, ncols=2,
                            constrained_layout=True,
                            figsize=(10, 6))

    fig.suptitle(title, fontsize=12)

    plot_line_variance(axs[0, 0], training_return)
    axs[0, 0].set_title('Training return')

    plot_line_variance(axs[0, 1], training_regret)
    axs[0, 1].set_title('Total training regret')

    plot_line_variance(axs[1, 0], testing_reward)
    axs[1, 0].set_title('Policy reward')
    axs[1, 0].set_ylim(bottom=0)

    plot_line_variance(axs[1, 1], testing_regret)
    axs[1, 1].set_title('Total testing regret')

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
    def __init__(self,alpha,args, gamma = 0.99):
        self.alpha = alpha
        self.args = args
        self.gamma = gamma
        self.ws = np.zeros((self.args.runs, self.args.steps+1, 8))
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
        for run_id in range(0, self.args.runs):
            global seed_count
            np.random.seed(seed_count)
            seed_count += 1
            self.current_state = random.randint(0, 6)
            self.semi_gradient_one_run(run_id)

    def semi_gradient_one_run(self, run_id):
        for step in range(1,self.args.steps+1):
            self.semi_gradient_one_step(run_id, step)

    def semi_gradient_one_step(self, run_id, step):
        old_state = self.current_state
        new_state = random.randint(0,6)
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

def main():

    # parses command line arguments
    args = get_arguments()
    alpha = 0.01
    agent = TD_Zero_Agent_Baird_Counterexample(alpha, args)
    agent.train_all_runs()
    print(np.mean(agent.ws[0:,-1], axis = 0))
    #print(agent.ws[0:, -1])

if __name__ == '__main__':
    main()
