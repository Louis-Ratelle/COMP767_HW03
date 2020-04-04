import gym
import tensorflow.compat.v1 as tf

env = gym.make('CartPole-v1')
gamma = 0.99
sess = tf.Session()

obs_size = env.observation_space.shape[0]

# Inputs
states = tf.placeholder(tf.float32, shape=(None, obs_size), name='state')
actions = tf.placeholder(tf.int32, shape=(None,), name='action')
returns = tf.placeholder(tf.float32, shape=(None,), name='return')

# Policy network
pi = dense_nn(states, [32, 32, env.action_space.n], name='pi_network')
sampled_actions = tf.squeeze(tf.multinomial(pi, 1))  # For sampling actions according to probabilities.

with tf.variable_scope('pi_optimize'):
    loss_pi = tf.reduce_mean(
        returns * tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=pi, labels=actions), name='loss_pi')
    optim_pi = tf.train.AdamOptimizer(0.001).minimize(loss_pi, name='adam_optim_pi')





def act(ob):
    return sess.run(sampled_actions, {states: [ob]})


def train(lr=0.01):
    step = 0
    episode_reward = 0.
    reward_history = []
    reward_averaged = []

    for _ in range(n_episodes):
        ob = env.reset()
        done = False

        obs = []
        actions = []
        rewards = []
        returns = []

        while not done:
            a = act(ob)
            new_ob, r, done, info = env.step(a)

            obs.append(ob)
            actions.append(a)
            rewards.append(r)
            ob = new_ob

        # One trajectory is complete!
        reward_history.append(episode_reward)
        reward_averaged.append(np.mean(reward_history[-10:]))
        episode_reward = 0.
        lr *= config.lr_decay

        # Estimate returns backwards.
        return_so_far = 0.0
        for r in rewards[::-1]:
            return_so_far = gamma * return_so_far + r
            returns.append(return_so_far)

        returns = returns[::-1]

        # Report the performance every `every_step` steps
        print("[episodes:{}/step:{}], best:{}, avg:{:.2f}:{}, lr:{:.4f}".format(
            n_episode, step, np.max(reward_history), np.mean(reward_history[-10:]),
            reward_history[-5:], lr,
        ))

        # Update the policy network with the data from one episode.
        sess.run([optim_pi], feed_dict={
            states: np.array(obs),
            actions: np.array(actions),
            returns: np.array(returns),
        })


def main():
    train()

if __name__ == '__main__':
    main()
