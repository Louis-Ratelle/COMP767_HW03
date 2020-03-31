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
from datetime import datetime

SEED = None
GAMMA = 0.9
ALPHAS = 2.0**np.array([-6, -4, -2, 0])
HIDDEN_SIZE = 100
RUNS = 20
EPISODES = 1000
MAX_STEPS = 200
UPDATE_EVERY = 10
ENV = 'CartPole-v1'
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
                        help='The learning rates to be '
                        'used. More than one value can be specified if '
                        'separated by spaces. Default: ' + str(ALPHAS))
    parser.add_argument('-s', '--hidden_size', type=int, default=HIDDEN_SIZE,
                        help='Size of each of the hidden layeres. '
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
                        help='Number of episodes to run before every update.'
                        'Default: ' + str(UPDATE_EVERY))
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


class policy(tf.keras.Model):
    def __init__(self, env, alpha, seed=None):
        super(policy, self).__init__()
        self.env = env.unwrapped
        self.hidden_size = args.hidden_size

        # set environment seed
        self.env.seed(seed)

        # initialize network
        self.n_inputs = self.env.observation_space.shape[0]
        self.n_outputs = self.env.action_space.n
        self.actions = np.arange(self.env.action_space.n)

        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(self.hidden_size,
                                  input_dim=self.n_inputs,
                                  activation=tf.nn.relu),
            tf.keras.layers.Dense(self.hidden_size,
                                  activation=tf.nn.relu),
            tf.keras.layers.Dense(self.n_outputs)
        ])

        self.model.build()

        # define loss function and optimizer
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=alpha)

        # initialize model parameters
        self.theta = self.model.trainable_variables
        for i, theta_i in enumerate(self.theta):
            self.theta[i] = theta_i * 0

    def choose_action(self, state):
        # state = tf.reshape(state, shape=(1, -1))
        state = tf.convert_to_tensor(state[None], dtype=tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(self.theta)                
            logits = self.model(state)
            action_probs = tf.squeeze(tf.nn.softmax(logits))
            # action = np.random.choice(
            #     np.arange(self.env.action_space.n),
            #     p=action_probs)
            action = np.argmax(action_probs)
            # loss = self.loss_fn([action], logits)
            loss = tf.nn.softmax_cross_entropy_with_logits(
                    labels=self.actions,
                    logits=logits)

        gradient = tape.gradient(loss, self.theta)

        return action, gradient

    def get_episode(self):
        states = []
        actions = []
        rewards = []
        gradients = []

        state = self.env.reset()

        done = False
        while not done:
            states.append(state)
            action, gradient = self.choose_action(state)
            actions.append(action)
            gradients.append(gradient)

            state, reward, done, _ = self.env.step(action)
            if done:
                reward = -10     # makes training faster
            rewards.append(reward)

        episode = {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'gradients': gradients
        }

        return episode

    def grad_ln(self, state, action, alpha):
        state = tf.convert_to_tensor(state[None], dtype=tf.float32)
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=alpha)

        with tf.GradientTape() as tape:
            loss = tf.nn.softmax_cross_entropy_with_logits(
                labels=np.arange(self.env.action_space.n),
                logits=self.model(state).numpy())

        gradients = tape.gradient(loss, self.model.trainable_variables)
        # update weights
        optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        # self.model.compile(
        #     optimizer=optimizer,
        #     loss=loss,
        #     metrics=['accuracy']
        # )

        return gradients

    def update(self, G, grad_ln):
        self.theta = self.theta + G * grad_ln
        self.optimizer.apply_gradients(zip(self.theta, self.model.trainable_variables))



# #############################################################################
#
# Algorithms
#
# #############################################################################


def sum_rewards(t, rewards, gamma):

    # crop rewards
    rewards = rewards[t:]
    discounted_rewards = [gamma ** i * rewards[i] for i in range(len(rewards))]

    return np.sum(discounted_rewards)


def reinforce(env, alpha, seed=None):
    gamma = args.gamma
    update_every = args.update_every

    assert 0 <= gamma <= 1
    assert alpha > 0

    pi = policy(env, alpha, seed)

    for i in range(args.episodes):
        episode = pi.get_episode()
        if i % update_every == 0:
            T = len(episode['states'])
            for t in range(T):
                G = sum_rewards(t, episode['rewards'], gamma)
                pi.update(G, episode['gradients'][t])



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

    if args.load is not None:
        # load pre-saved data
        filename = args.load
        steps, args = load(filename)
        print('Using saved data from: {}'.format(filename))
    else:
        # steps = runs(env, alphas, lambdas)
        reinforce(env, 0.1)
        save([steps, args], 'steps')

    plot3('Average steps', steps, alphas, lambdas)


if __name__ == '__main__':
    main()
