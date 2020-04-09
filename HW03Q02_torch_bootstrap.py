import os
import sys
import gym
import copy
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.distributions import Categorical

from tqdm import tqdm
from datetime import datetime

SEED = None
GAMMA = 0.99
ALPHAS = [0.01]
HIDDEN_SIZE = 8
RUNS = 5
EPISODES = 2000
MAX_STEPS = 200
UPDATE_EVERY = 10
ENV = 'CartPole-v0'
SAVED_MODELS_FOLDER = './data/'
NOW = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())

# #############################################################################
#
# Parser
#
# #############################################################################


def get_arguments():

    parser = argparse.ArgumentParser(
        description='Controlling a gym environment with Sarsa(lambda).')

    parser.add_argument('--seed', type=int, default=SEED,
                        help='Seed for the random number generator.')
    parser.add_argument('--env', type=str, default=ENV,
                        help='The environment to be used. Default: '
                        + ENV)
    parser.add_argument('--gamma', type=float, default=GAMMA,
                        help='Defines the discount rate. Default: '
                        + str(GAMMA))
    parser.add_argument('--alphas', type=float, default=ALPHAS,
                        nargs='*',
                        help='The learning rates to be used for the '
                        'policy. More than one value can be specified if '
                        'separated by spaces. Default: ' + str(ALPHAS))
    parser.add_argument('-s', '--hidden_size', type=int, default=HIDDEN_SIZE,
                        help='Size of the hidden layer. '
                        'Default: ' + str(HIDDEN_SIZE))
    parser.add_argument('-n', '--runs', type=int, default=RUNS,
                        help='Number of runs to be executed. Default: '
                        + str(RUNS))
    parser.add_argument('-e', '--episodes', type=int,
                        default=EPISODES,
                        help='Number of episodes to be executed in a single '
                        'run. Default: ' + str(EPISODES))
    parser.add_argument('--max_steps', type=int, default=MAX_STEPS,
                        help='Number of maximum steps allowed in a single '
                        'episode. Default: ' + str(MAX_STEPS))
    parser.add_argument('-u', '--update_every', type=int, default=UPDATE_EVERY,
                        help='Number of episodes to run before every weight '
                        'update. Default: ' + str(UPDATE_EVERY))
    parser.add_argument('-v', '--verbose', action="store_true",
                        help='If this flag is set, the algorithm will '
                        'generate more output, useful for debugging.')
    parser.add_argument('-r', '--render', action="store_true",
                        help='If this flag is set, each episode will be '
                        'rendered.')
    parser.add_argument('-l', '--load', type=str, default=None,
                        help='Filename of a .pickle pre-saved data file saved '
                        'in the {} folder. Please include the .pickle '
                        'extension.'.format(SAVED_MODELS_FOLDER))

    return parser.parse_args()


# #############################################################################
#
# Plotting
#
# #############################################################################


def plot3(title, steps_rf, steps_ac):
    '''Creates the plots: average steps per lambda, per alpha, per episodes'''

    fig, axs = plt.subplots(nrows=1, ncols=1,
                            constrained_layout=True,
                            sharey=True,
                            figsize=(10, 10))

    fig.suptitle(title, fontsize=12)

    plot_learning_curves(axs, steps_rf, steps_ac)
    axs.set_xlabel('Episodes')
    axs.set_ylabel('Number of steps')
    axs.set_title('Learning curves')
    axs.legend()

    plt.show()


def plot_learning_curves(ax, steps_rf, steps_ac):
    '''Plots the average number of steps per lambda for each alpha.

    Input:
    ax      : the target axis object
    steps   : array of shape
              (len(alphas), args.runs, args.episodes)
              containing the number of steps for each alpha_w, alpha_t, run,
              episode'''

    x_values = np.arange(1, args.episodes + 1)
    color_idx = 0

    for alpha in range(steps_rf.shape[0]):
        # get number of steps from last episode
        data = steps_rf[alpha, :, :]
        plot_line_variance(
            ax,
            x_values,
            data,
            label='Reinforce',
            color='C' + str(color_idx),
            axis=0
        )
        color_idx += 1

    for alpha in range(steps_ac.shape[0]):
        # get number of steps from last episode
        data = steps_ac[alpha, :, :]
        plot_line_variance(
            ax,
            x_values,
            data,
            label='Actor-Critic',
            color='C' + str(color_idx),
            axis=0
        )
        color_idx += 1

    ax.set_title('Reinforce and Actor-Critic')


def plot_line_variance(ax, x_values, data, label, color, axis=0, delta=1):
    '''Plots the average data for each time step and draws a cloud
    of the standard deviation around the average.

    Input:
    ax      : axis object where the plot will be drawn
    data    : data of shape (num_trials, timesteps)
    color   : the color to be used
    delta   : (optional) scaling of the standard deviation around the average
              if ommitted, delta = 1.'''

    avg = np.average(data, axis)
    std = np.std(data, axis)

    # min_values = np.min(data, axis)
    # max_values = np.max(data, axis)

    # ax.plot(min_values, color + '--', linewidth=0.5)
    # ax.plot(max_values, color + '--', linewidth=0.5)

    ax.fill_between(x_values,
                    avg + delta * std,
                    avg - delta * std,
                    facecolor=color,
                    alpha=0.5)
    # ax.plot(x_values, avg, label=label, color=color, marker='.')
    ax.plot(x_values, avg, label=label, color=color)

# #############################################################################
#
# Rewards
#
# #############################################################################


def discount_rewards(rewards):
    returns = []
    R = 0

    # calculate discounted rewards from inversed array
    for r in rewards[::-1]:
        # calculate the discounted value
        R = r + args.gamma * R
        # insert in the beginning of the list
        # (the list is once again in the correct order)
        returns.insert(0, R)

    return returns

# #############################################################################
#
# Classes
#
# #############################################################################


class Critic(nn.Module):
    def __init__(self, input_dim, output_dim, alpha):
        super(Critic, self).__init__()
        hidden_size = args.hidden_size

        self.hidden1 = nn.Linear(input_dim, 400)
        self.hidden2 = nn.Linear(400, 400)

        self.value_head = nn.Linear(400, 1)

        self.actions = []
        self.rewards = []

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        # critic: evaluates being in the state s_t
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))

        value = self.value_head(x)

        return value.item()

    def backprop(self, log_prob, target, v0, i):

        gamma = args.gamma

        value_losses = []

        advantage = target - v0


        # convert to tensor
        advantage = torch.tensor(advantage)
        log_prob = torch.tensor(log_prob)
        v0 = torch.tensor([v0])
        target = torch.tensor([target])

        target.requires_grad = True
        v0.requires_grad = True
        advantage.requires_grad = True

        # critic loss using L1 smooth loss
        # value_losses.append(F.smooth_l1_loss(v0, target))
        value_losses.append(-v0)

        # sum up all the values of policy_losses and value_losses
        loss = torch.stack(value_losses).sum()

        # backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # reset buffers
        del self.rewards[:]
        del self.actions[:]


class Actor(nn.Module):
    def __init__(self, input_dim, output_dim, alpha):
        super(Actor, self).__init__()
        hidden_size = args.hidden_size

        self.hidden1 = nn.Linear(input_dim, 40)
        self.hidden2 = nn.Linear(40, 40)

        self.action_head = nn.Linear(40, output_dim)

        self.actions = []
        self.rewards = []

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.00002)

    def forward(self, x):

        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))

        # actor: choses action to take from state s_t 
        # by returning probability of each action
        action_prob = F.softmax(self.action_head(x), dim=-1)

        return action_prob

    def choose_action(self, state):
        # state = torch.from_numpy(state).float()
        probs= self(state)

        # get categorical distribution from probabilities
        # and sample an action
        distr = Categorical(probs)
        action = distr.sample()

        return action.item(), distr.log_prob(action)

    def backprop(self, log_prob, target, v0, i):

        gamma = args.gamma

        policy_losses = []
        value_losses = []

        advantage = target - v0

        # # convert to tensor
        # advantage = torch.tensor(advantage)
        # log_prob = torch.tensor(log_prob)
        # v0 = torch.tensor(v0)
        # target = torch.tensor(target)

        # actor loss (negative log-likelihood)
        policy_losses.append(-log_prob * advantage * i)

        # sum up all the values of policy_losses
        loss = torch.stack(policy_losses).sum()

        # backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# #############################################################################
#
# Methods
#
# #############################################################################


def actor_critic(alpha, seed=None):

    n_episodes = args.episodes
    update_every = args.update_every
    gamma = args.gamma
    scores = []
    batch_size = 10

    assert 0 <= gamma <= 1
    assert alpha > 0

    env = gym.make(args.env)
    env.seed(args.seed)

    # env._max_episode_steps = args.max_steps

    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    ac = Actor(input_dim, output_dim, alpha)
    cr = Critic(input_dim, output_dim, alpha)

    for episode in range(args.episodes):

        # reset environment and episode reward
        state0 = env.reset()
        state0 = torch.from_numpy(state0).float()
        steps = 0
        done = False
        i = 1
        while not done:

            # select action from policy
            action, log_prob = ac.choose_action(state0)
            v0 = cr(state0)

            # take the action
            state1, reward, done, _ = env.step(action)

            if args.render:
                env.render()

            if done:
                v1 = torch.tensor([0])
            else:
                state1 = torch.from_numpy(state1).float()
                v1 = cr(state1)

            target = reward + gamma * v1

            # model.rewards.append(reward)
            steps += 1
            ac.backprop(log_prob, target, v0, i)
            cr.backprop(log_prob, target, v0, i)
            i *= gamma
            state0 = state1


        scores.append(steps)

        # log results
        if episode % args.update_every == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  episode, steps, np.mean(scores[-args.update_every:])))


# #############################################################################
#
# Main
#
# #############################################################################


# global variables
args = get_arguments()
eps = np.finfo(np.float32).eps.item()


def save(objects, filename):
    f = open(os.path.join(SAVED_MODELS_FOLDER,
                          NOW + '_' + filename + '.pickle'), 'wb')
    pickle.dump(objects, f)
    f.close()


def load(filename):
    # try to open in current folder or full path
    try:
        f = open(filename, 'rb')
        steps_rf, steps_ac, args = pickle.load(f)
        f.close()
    except FileNotFoundError:
        # try to open file in the data folder
        filename = os.path.join(SAVED_MODELS_FOLDER, filename)
        load(filename)
        print('Could not open file {}'.format(filename))
        sys.exit()

    return steps_rf, steps_ac, args


def runs(agent, alphas):
    '''Performs multiple runs (as defined by parameter --runs)
    for a list of parameters alpha and a list of parameter alphas_w.

    Input:
    agent     : the agent to be used
    alphas    : list of alpha_t (learning rates)

    Output:
    array of shape (len(alphas), args.runs, args.episodes)
    containing the number of steps for each alpha, run, episode
    '''

    steps = np.zeros((len(alphas), args.runs, args.episodes))
    for alpha_idx, alpha in enumerate(alphas):
        for run in tqdm(range(args.runs)):
            # sets a new seed for each run
            seed = np.random.randint(0, 2**32 - 1)

            steps[alpha_idx, run, :] = agent(
                alpha, seed)

    return steps


def main():
    global args

    # sets the seed for random experiments
    np.random.seed(args.seed)
    if args.seed is not None:
        torch.manual_seed(args.seed)

    # env._max_episode_steps = args.max_steps

    alphas = args.alphas

    if args.load is not None:
        # load pre-saved data
        filename = args.load
        steps_rf, steps_ac, args = load(filename)
        print('Using saved data from: {}'.format(filename))
    else:
        # steps_rf = runs(reinforce, alphas_t, [0])
        # steps_ac = runs(actor_critic, alphas_t, alphas_w)
        # reinforce(alpha=0.01)
        actor_critic(0.001)
        # actor_critic_original(0.01, 0.01)
        # save([steps_rf, steps_ac, args], 'steps')

    # plot3('', steps_rf, steps_ac)


if __name__ == '__main__':
    main()
