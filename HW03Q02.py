import tensorflow as tf
import numpy as np
import gym
import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pickle
from datetime import datetime

SEED = None
GAMMA = 0.9
ALPHAS = 2.0**np.array([-6, -4, -2, 0])
THETA_SIZE = 32
W_SIZE = 32
RUNS = 5
EPISODES = 2000
MAX_STEPS = 200
UPDATE_EVERY = 10
ENV = 'CartPole-v1'
SAVED_MODELS_FOLDER = './data/'
NOW = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())

n_episodes = 2000
env = gym.make('CartPole-v1').unwrapped
scores = []
update_every = 10

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
    parser.add_argument('--theta_size', type=int, default=THETA_SIZE,
                        help='Size of the hidden layer. '
                        'Default: ' + str(THETA_SIZE))
    parser.add_argument('--w_size', type=int, default=W_SIZE,
                        help='Size of weight vector. '
                        'Default: ' + str(W_SIZE))
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
# Rewards
#
# #############################################################################

def discount_rewards(r, gamma = 0.9):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


# #############################################################################
#
# Classes
#
# #############################################################################

class policy(tf.keras.Model):
    def __init__(self, env, seed=None):
        super(policy, self).__init__()

        # sets the environment
        self.env = env.unwrapped
        self.env.seed(seed)

        # build 2-layer MLP model
        self.model = tf.keras.Sequential()
        self.model.add(
            tf.keras.layers.Dense(args.theta_size,
                                  input_dim=env.observation_space.shape[0],
                                  activation='relu'))
        self.model.add(
            tf.keras.layers.Dense(env.action_space.n,
                                  activation='softmax'))
        self.model.build()

        # initialize model parameters
        self.theta = self.model.trainable_variables
        for ix, grad in enumerate(self.theta):
            self.theta[ix] = grad * 0

    def call(self, state):
        return self.model(state)

    def choose_action(self, state):
        'Returns action and gradients.'

        compute_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        state = state[None]
        with tf.GradientTape() as tape:
            # forward pass
            logits = self(state)
            action_probs = logits.numpy()
            # Choose random action with p = action dist
            action = np.random.choice(action_probs[0], p=action_probs[0])
            action = np.argmax(action_probs == action)
            loss = compute_loss([action], logits)
        grads = tape.gradient(loss, self.model.trainable_variables)

        return action, grads


class v_hat(tf.keras.Model):
    def __init__(self, alpha_w):
        super(policy, self).__init__()

        # build 2-layer MLP model
        self.model = tf.keras.Sequential()
        self.model.add(
            tf.keras.layers.Dense(
                args.w_size,
                input_dim=env.observation_space.shape[0],
                activation='relu'
                )
            )
        self.model.build()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=alpha_w)


        # initialize model parameters
        self.w = self.model.trainable_variables
        for ix, grad in enumerate(self.w):
            self.w[ix] = grad * 0

    def call(self, state):
        state = state[None]
        return self.model(state)

    def update(self, state, reward):
        compute_loss = tf.keras.losses.MeanSquaredError()

        state = state[None]
        with tf.GradientTape() as tape:
            # forward pass
            value = self(state)
            loss = compute_loss(reward, value)
        grads = tape.gradient(loss, self.model.trainable_variables)

        return grads

    def apply_gradients(w):
        self.optimizer.apply_gradients(zip(w, self.model.trainable_variables))
        for idx, grad in enumerate(w):
            w[idx] = grad * 0



# #############################################################################
#
# Methods
#
# #############################################################################


def reinforce(alpha, seed=None):

    n_episodes = args.episodes
    update_every = args.update_every
    gamma = args.gamma

    assert alpha > 0
    assert 0 <= gamma <= 1

    env = gym.make(args.env)

    # sets optimizer and learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=alpha)
    pi = policy(env, seed)

    for e in range(n_episodes):
        state = env.reset()
        episode = []
        steps = 0
        done = False
        while not done:
            action, grads = pi.choose_action(state)
            state, r, done, _ = env.step(action)
            steps += 1
            if done: r -= 10    # makes training faster
            episode.append([grads, r])

        scores.append(steps)

        # Discound rewards
        episode = np.array(episode)
        episode[:, 1] = discount_rewards(episode[:, 1], gamma)

        for grads, r in episode:
            for idx, grad in enumerate(grads):
                pi.theta[idx] += grad * r

        if e % update_every == 0:
            optimizer.apply_gradients(zip(pi.theta, pi.model.trainable_variables))
            for idx, grad in enumerate(pi.theta):
                pi.theta[idx] = grad * 0

        if e % 100 == 0:
            print("Episode  {}  Score  {}".format(e, np.mean(scores[-100:])))


def actor_critic(alpha_t, alpha_w, seed=None):

    n_episodes = args.episodes
    update_every = args.update_every
    gamma = args.gamma

    assert 0 <= gamma <= 1
    assert alpha_t > 0
    assert alpha_w > 0

    env = gym.make(args.env)

    # sets optimizer and learning rate
    actor_opt = tf.keras.optimizers.Adam(learning_rate=alpha_t)
    actor = policy(env, seed)

    critic = v_hat(alpha_w)

    for e in range(n_episodes):
        state0 = env.reset()
        episode = []
        steps = 0
        done = False
        I = 1

        while not done:
            action, grads = actor.choose_action(state0)
            state1, R, done, _ = env.step(action)
            steps += 1
            delta = R + gamma * critic(state1, w) - critic(state0, w)
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

    # env._max_episode_steps = args.max_steps

    alphas = args.alphas

    if args.load is not None:
        # load pre-saved data
        filename = args.load
        steps, args = load(filename)
        print('Using saved data from: {}'.format(filename))
    else:
        # steps = runs(env, alphas, lambdas)
        reinforce(alpha=0.01)
        # save([steps, args], 'steps')

    # plot3('Average steps', steps, alphas, lambdas)


if __name__ == '__main__':
    main()
