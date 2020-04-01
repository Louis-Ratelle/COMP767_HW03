import tensorflow as tf
import numpy as np
import gym

n_episodes = 2000
env = gym.make('CartPole-v1').unwrapped
scores = []
update_every = 10

# gamma: discount rate 
def discount_rewards(r, gamma = 0.9):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(32, input_dim = 4, activation='relu'))
model.add(tf.keras.layers.Dense(2, activation = "softmax"))
model.build()
optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01)
compute_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

theta = model.trainable_variables
for ix, grad in enumerate(theta):
    theta[ix] = grad * 0

for e in range(n_episodes):
    # reset the enviroment
    state = env.reset()
    episode = []
    score = 0
    done = False
    while not done:
        state = state.reshape([1,4])
        with tf.GradientTape() as tape:
            #forward pass
            logits = model(state)
            action_probs = logits.numpy()
            # Choose random action with p = action dist
            action = np.random.choice(action_probs[0], p=action_probs[0])
            action = np.argmax(action_probs == action)
            loss = compute_loss([action], logits)
        grads = tape.gradient(loss, model.trainable_variables)
        # make the choosen action
        state, r, done, _ = env.step(action)
        score += 1
        if done: r -= 10 # small trick to make training faster

        episode.append([grads, r])

    scores.append(score)
    # Discound the rewards
    episode = np.array(episode)
    episode[:,1] = discount_rewards(episode[:,1])

    for grads, r in episode:
        for ix,grad in enumerate(grads):
            theta[ix] += grad * r

    if e % update_every == 0:
        optimizer.apply_gradients(zip(theta, model.trainable_variables))
        for ix, grad in enumerate(theta):
            theta[ix] = grad * 0

    if e % 100 == 0:
        print("Episode  {}  Score  {}".format(e, np.mean(scores[-100:])))
