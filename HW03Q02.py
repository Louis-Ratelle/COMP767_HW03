import numpy as np
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
# Helper functions
#
# #############################################################################

def random_argmax(vector):
    '''Select argmax at random in case of tie for the max.'''

    index = np.random.choice(np.where(vector == vector.max())[0])

    return index


# #############################################################################
#
# Sarsa
#
# #############################################################################

def greedy_action(env, state, weights):
    '''Chooses a greedy action.

    Input:
    env     : the environment to be used
    state   : the current state
    weights : the current weights to evaluate the state-action
              value function for the greedy action

    Output:
    action index'''

    q = np.zeros(env.action_space.n)

    for action in range(env.action_space.n):
        # features[f(state, action)] = 1
        # q[action] = np.dot(weights, features)
        q[action] = np.sum(weights[f(state, action)])

    try:
        action = random_argmax(q)
    except (RuntimeError, ValueError):
        action = env.action_space.sample()
        'Warning: using random actions for greedy policy. Try modifying alpha.'

    if args.verbose:
        print('greedy action: {}'.format(action))

    return action


def random_action(env):
    '''Chooses a random action from the
    action space in the environment env.

    Input:
    env     : the environment to be used

    Output:
    action index'''

    action = env.action_space.sample()
    if args.verbose:
        print('Random action: {}'.format(action))
    return action


def choose_action(env, state, weights):
    '''Chooses a random action with probability args.epsilon
    and a greedy action with probability 1 - args.epsilon

    Input:
    env     : the environment to be used
    state   : the current state
    weights : the current weights to evaluate the state-action
              value function for the greedy action'''

    choices = ['random', 'greedy']
    c = np.random.choice(choices, p=[args.epsilon, 1 - args.epsilon])
    if c == 'random':
        return random_action(env)
    else:
        return greedy_action(env, state, weights)


def f(s, a):
    '''Returns list of indices where the features are active.

    Input:
    s   : state
    a   : action

    Output:
    list of indices where features are active. The list has
    length args.tilings (the number of active features) and
    each element on the list can go from
    0 to args.size - 1'''

    # get position and velocity from state s
    x, xdot = s

    # For a state s = [x, xdot] and action a
    # obtain indices for each tiling as defined in Sutton, R. Reinforcement
    # Learning, 2nd ed. (2018), Section 10.1, page 246'''
    indices = tiles(
        iht, 8, [8 * x / (0.5 + 1.2), 8 * xdot / (0.07 + 0.07)], [a]
        )
    active_idx = indices
    if args.verbose:
        print('State: {}, Action: {}'.format(s, a))
        print(indices)
        print(active_idx)
    return active_idx


def sarsa(env, lmbda, alpha, seed=None):
    '''Performs the sarsa(lambda) algorithm as defined in
    Sutton, Richard. Reinforcement Learning, 2nd. ed.
    page 305

    Input:
    env     : the environment to be used
    lmbda   : the parameter lambda of Sarsa(lambda) algorithm
    alpha   : the learning rate
    seed    : (optional) if present, the environment will be
              initialised with this seed, otherwise the system
              will generate its own seed.

    Output:
    array of shape (args.episodes) containing the number of
    steps per episode.'''

    assert alpha > 0
    assert 0 <= lmbda <= 1

    # initialize environement and weights
    env.seed(seed)
    w = np.zeros(args.size)
    steps_per_episode = np.zeros(args.episodes)

    # create index hash table
    iht = IHT(args.size)

    for episode in tqdm(range(args.episodes)):

        steps = 0
        env.reset()
        state0 = env.state
        action0 = choose_action(env, state0, w)
        z = np.zeros(w.shape)

        while steps < args.max_steps:
            steps += 1
            state1, reward, done, info = env.step(action0)
            if args.render: env.render()
            delta = reward
            for i in f(state0, action0):
                delta = delta - w[i]
                z[i] = 1
            if done:
                w = w + alpha * delta * z
                # go to next episode
                break
            action1 = choose_action(env, state1, w)
            for i in f(state1, action1):
                delta = delta + args.gamma * w[i]
            w = w + alpha * delta * z
            z = args.gamma * lmbda * z
            state0 = state1
            action0 = action1

        steps_per_episode[episode] = steps
        if args.verbose:
            print('Episode {} finished after {} steps.'.format(episode + 1, steps))
    env.close()

    return steps_per_episode

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
