import numpy as np
import tensorflow as tf
import gym
import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pickle
import os
import sys
import copy
from tqdm import tqdm
from tiling import tiles, IHT
from datetime import datetime

SEED = None
GAMMA = 1
EPSILON = 0.1
TILINGS = 8
SIZE = 4096
ALPHAS = 2.0**np.array([-6, -4, -2, 0]) / TILINGS
LAMBDAS = [1, 0.99, 0.95, 0.5, 0]
RUNS = 20
EPISODES = 1000
MAX_STEPS = 2000
ENV = 'MountainCar-v0'
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
    parser.add_argument('--epsilon', type=float, default=EPSILON,
                        help='Defines the parameter epsilon for the '
                        'epsilon-greedy policy. The algorithm will '
                        'perform an exploratory action with probability '
                        'epsilon. Default: ' + str(EPSILON))
    parser.add_argument('--alphas', type=float, default=ALPHAS,
                        nargs='*',
                        help='The learning rates to be '
                        'used. More than one value can be specified if '
                        'separated by spaces. Default: ' + str(ALPHAS))
    parser.add_argument('--lambdas', type=float, default=LAMBDAS,
                        nargs='*',
                        help='The lambda parameters for '
                        'Sarsa(lambda). More than one value can be specified '
                        'if separated by spaces. Default: ' + str(LAMBDAS))
    parser.add_argument('-t', '--tilings', type=int, default=TILINGS,
                        help='Number of tiles to use. Default: '
                        + str(TILINGS))
    parser.add_argument('-s', '--size', type=int, default=SIZE,
                        help='Size of each tile, generally a square number. '
                        'This corresponds to the size index hash table of the '
                        'tiling algorithm. Default: ' + str(SIZE))
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


def plot3(title, steps, alphas, lambdas):
    '''Creates the plots: average steps per lambda, per alpha, per episodes'''

    fig, axs = plt.subplots(nrows=1, ncols=3,
                            constrained_layout=True,
                            sharey=True,
                            figsize=(10, 3))

    fig.suptitle(title, fontsize=12)

    plot_lambdas(axs[0], steps, x_values=alphas, series=lambdas)
    axs[0].set_xlabel('alpha (log scale)')
    axs[0].set_ylabel('Average steps')
    axs[0].set_title('Sarsa($\\lambda$)')
    axs[0].set_xscale('log', basex=2)
    axs[0].legend()

    plot_alphas(axs[1], steps, x_values=lambdas, series=alphas)
    axs[1].set_xlabel('lambda')
    axs[1].set_ylabel('Average steps')
    axs[1].set_title('Learning rate')
    axs[1].legend()

    plot_alphas2(axs[2], steps, alphas, lambdas)
    axs[2].set_xlabel('episode')
    axs[2].set_ylabel('Average steps')
    axs[2].set_xlim(None, 200)
    axs[2].legend()

    fig, axs_zoom = plt.subplots(nrows=1, ncols=3,
                            constrained_layout=True,
                            sharey=True,
                            figsize=(10, 3))

    plot_lambdas(axs_zoom[0], steps, x_values=alphas, series=lambdas)
    axs_zoom[0].set_xlabel('alpha (log scale)')
    axs_zoom[0].set_ylabel('Average steps')
    axs_zoom[0].set_title('Sarsa($\\lambda$)')
    axs_zoom[0].set_xscale('log', basex=2)
    axs_zoom[0].legend()

    plot_alphas(axs_zoom[1], steps, x_values=lambdas, series=alphas)
    axs_zoom[1].set_xlabel('lambda')
    axs_zoom[1].set_ylabel('Average steps')
    axs_zoom[1].set_title('Learning rate')
    axs_zoom[1].legend()

    plot_alphas2(axs_zoom[2], steps, alphas, lambdas)
    axs_zoom[2].set_xlabel('episode')
    axs_zoom[2].set_ylabel('Average steps')
    axs_zoom[2].set_xlim(None, 200)
    axs_zoom[2].legend()

    axs_zoom[2].set_ylim(None, 500)
    fig.suptitle(title + ' (zoomed)', fontsize=12)

    plt.show()


def plot_lambdas(ax, steps, x_values, series):
    '''Plots the average number of steps per alpha for each lambda.

    Input:
    ax      : the target axis object
    steps   : array of shape
              (len(lambdas), len(alphas), args.runs, args.episodes)
              containing the number of steps for each lambda, alpha, run,
              episode
    x_values: array of shape len(aplhas) with labels for the x axis
    series  : array of shape len(lambdas) with the series names for the
              legend'''

    for lmbda in range(steps.shape[0]):
        # get number of steps from last episode
        data = steps[lmbda, :, :, args.episodes - 1]
        plot_line_variance(ax,
                           x_values,
                           data,
                           label='$\\lambda=$' + str(series[lmbda]),
                           color='C' + str(lmbda),
                           axis=1)


def plot_alphas(ax, steps, x_values, series):
    '''Plots the average number of steps per lambda for each alpha.

    Input:
    ax      : the target axis object
    steps   : array of shape
              (len(lambdas), len(alphas), args.runs, args.episodes)
              containing the number of steps for each lambda, alpha, run,
              episode
    x_values: array of shape len(lambdas) with labels for the x axis
    series  : array of shape len(alphas) with the series names for the
              legend'''

    for alpha in range(steps.shape[1]):
        # get number of steps from last episode
        data = steps[:, alpha, :, args.episodes - 1]
        plot_line_variance(ax,
                           x_values,
                           data,
                           label='$\\alpha=$' + str(series[alpha]),
                           color='C' + str(alpha),
                           axis=1)


def plot_alphas2(ax, steps, series, lambdas):
    '''Plots the learning curves (the average number of steps per
    episode) for each alpha.

    Input:
    ax      : the target axis object
    steps   : array of shape
              (len(lambdas), len(alphas), args.runs, args.episodes)
              containing the number of steps for each lambda, alpha, run,
              episode
    series  : array of shape len(alphas) with the series names for the
              legend
    lambdas : arrays of lambdas
    '''

    # use the middle element of the lambdas vector
    idx = np.int(len(lambdas) / 2)

    x_values = np.arange(args.episodes)

    for alpha in range(steps.shape[1]):
        data = steps[idx, alpha, :, :]
        plot_line_variance(ax,
                        x_values,
                        data,
                        label='$\\alpha=$' + str(series[alpha]),
                        color='C' + str(alpha),
                        axis=0)

    ax.set_title('Sarsa({})'.format(lambdas[idx]))


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

    # ax.plot(avg + delta * std, color + '--', linewidth=0.5)
    # ax.plot(avg - delta * std, color + '--', linewidth=0.5)
    ax.fill_between(x_values,
                    avg + delta * std,
                    avg - delta * std,
                    facecolor=color,
                    alpha=0.2)
    ax.plot(x_values, avg, label=label, color=color, marker='.')

# #############################################################################
#
# Policy
#
# #############################################################################


class policy():
    def __init__(self, env, seed=None):
        self.env = env
        self.hidden_size = args.hidden_size

        # initialize environement and weights
        self.env.seed(seed)

        # initialize network
        self.n_inputs = self.env.observation_space.shape[0]
        self.n_outputs = self.env.action_space.n

        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(self.hidden_size, activation='relu')
            # tf.keras.layers.Dropout(0.2)
            tf.keras.layers.Dense(self.hidden_size, activation='relu')
            # tf.keras.layers.Dropout(0.2)
            tf.keras.layers.Dense(self.n_outputs)
        ])

    def choose_action(self, state):
        predictions = self.model(state).numpy()
        action_probs = tf.nn.softmax(predictions).numpy()
        action = np.random.choice(self.env.action_space, p=action_probs)
        return action

    def get_episode(self):
        states = []
        actions = []
        rewards = []

        state = self.env.reset()

        done = False
        while not done:
            states.append(state)
            action = self.choose_action(state)
            actions.append(action)

            state, reward, done, _ = self.env.step(action)
            rewards.append(reward)

        episode = {
            'states': states,
            'actions': actions,
            'rewards': rewards
        }

        return episode

# #############################################################################
#
# Algorithms
#
# #############################################################################


def reinforce(env, alpha, seed=None):
    gamma = args.gamma

    assert 0 <= gamma <= 1
    assert alpha > 0

    delta = np.Infinity
    pi = policy(env, seed)

    while delta > tol:
        episode = pi.get_episode()
        T = len(episode['states'])
        for t in range(T):
            G = sum
            delta = G - v_hat(state, w)
            w += alpha_w * delta * grad_v(state, w)
            theta += alpha_t * gamma ** t * delta * grad_pol(state, action, theta)


def actor_critic(env, alpha_t, alpha_w, lambda_t, lambda_w, seed=None):
    gamma = args.gamma

    assert 0 <= gamma <= 1    
    assert alpha_t > 0
    assert alpha_w > 0
    assert 0 <= lambda_t <= 1
    assert 0 <= lambda_w <= 1

    theta_size = 10
    w_size = 10

    theta = np.zeros(theta_size)
    w = np.zeros(w_size)

    delta = np.Infinity

    while delta > tol:
        state0
        z_t = np.zeros(theta_size)
        z_w = np.zeros(w_size)
        I = 1
        done = False

        while not done:
            action0 = pol_state(state0, theta)
            state1, R, done, info = env.step(action0)
            delta = R + gamma * v_hat(state1, w) - v_hat(state0, w)
            z_w = gamma * lambda_w * z_w + grad_v(state0, w)
            z_t = gamma * lambda_t * z_t + I * grad_pol(state0, action0, theta)
            w += alpha_w * delta * z_w
            theta += alpha_t * delta * z_t
            I *= gamma
            state0 = state1

# #############################################################################
#
# Main
#
# #############################################################################


# global variables
args = get_arguments()
iht = IHT(args.size)


def save(objects, filename):
    f = open(os.path.join(SAVED_MODELS_FOLDER,
                          NOW + '_' + filename + '.pickle'), 'wb')
    pickle.dump(objects, f)
    f.close()


def load(filename):
    filename = os.path.join(SAVED_MODELS_FOLDER, filename)
    try:
        f = open(filename, 'rb')
        steps, args = pickle.load(f)
        f.close()
    except FileNotFoundError:
        print('Could not open file {}'.format(filename))
        sys.exit()

    return steps, args


def runs(env, alphas, lambdas):
    '''Performs multiple runs (as defined by parameter --runs)
    for a list of parameters alpha and a list of parameter lambdas.

    Input:
    env     : the environment to be used
    alphas  : list of alphas (learning rates)
    lambdas : list of lambdas

    Output:
    array of shape (len(lambdas), len(alphas), args.runs, args.episodes)
    containing the number of steps for each lambda, alpha, run, episode
    '''

    steps = np.zeros((len(lambdas), len(alphas), args.runs, args.episodes))
    for lambda_idx, lmbda in enumerate(lambdas):
        for alpha_idx, alpha in enumerate(alphas):
            print('Running Sarsa({}) with learning rate {}'.format(lmbda, alpha))
            for run in tqdm(range(args.runs)):
                # sets a new seed for each run
                seed = np.random.randint(0, 2**32 - 1)

                steps[lambda_idx, alpha_idx, run, :] = sarsa(
                    env, lmbda, alpha, seed)

    return steps


def main():
    global args

    # sets the seed for random experiments
    np.random.seed(args.seed)

    env = gym.make(args.env)
    env._max_episode_steps = args.max_steps

    alphas = args.alphas
    lambdas = args.lambdas

    if args.load is not None:
        # load pre-saved data
        filename = args.load
        steps, args = load(filename)
        print('Using saved data from: {}'.format(filename))
    else:
        steps = runs(env, alphas, lambdas)
        save([steps, args], 'steps')

    plot3('Average steps', steps, alphas, lambdas)


if __name__ == '__main__':
    main()
