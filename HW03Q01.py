import matplotlib.pyplot as plt
import numpy as np
import argparse

import os
import sys

from time import time
from datetime import datetime

# constants
ARMS = 10
RUNS = 10
STEPS_PER_RUN = 1000
TRAINING_STEPS = 10
TESTING_STEPS = 5

NOW = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
SEED = None
# SEED = 197710

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

    parser = argparse.ArgumentParser(description='Creating a k-armed bandit.')
    parser.add_argument('--seed', type=int, default=SEED,
                        help='Seed for the random number generator.')
    parser.add_argument('-k', '--arms', type=int, default=ARMS,
                        help='Number of arms on the bandit. Default: '
                        + str(ARMS))
    parser.add_argument('-n', '--runs', type=int, default=RUNS,
                        help='Number of runs to be executed. Default: '
                        + str(RUNS))
    parser.add_argument('-s', '--steps', type=int, default=STEPS_PER_RUN,
                        help='Number of steps in each run. One run step is '
                        'the ensemble of training steps and testing steps. '
                        'Default: ' + str(STEPS_PER_RUN))
    parser.add_argument('--training_steps', type=int, default=TRAINING_STEPS,
                        help='Number of training steps to be executed. '
                        'Default: ' + str(TRAINING_STEPS))
    parser.add_argument('--testing_steps', type=int, default=TESTING_STEPS,
                        help='Number of testing steps to be executed. '
                        'Default: ' + str(TESTING_STEPS))

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
# Main
#
# #############################################################################


def main():

    # parses command line arguments
    args = get_arguments()


if __name__ == '__main__':
    main()
