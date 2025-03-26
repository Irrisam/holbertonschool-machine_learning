#!/usr/bin/env python3
"""
    module to implement q-learning
"""
import numpy as np
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1,
          gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
        performs q-learning

    :param env: FrozenLake instance
    :param Q: ndarray, Q-table
    :param episodes: total number of episodes to train over
    :param max_steps: max steps per episode
    :param alpha: learning rate
    :param gamma: discount factor
    :param epsilon: initial epsilon for epsilon-greedy
    :param min_epsilon: minimum epsilon value after decay
    :param epsilon_decay: rate at which epsilon decays over episodes

    :return: Q, total_rewards
        Q: updated Q-table
        total_rewards: list of rewards per episode
    """

    total_rewards = []

    for episode in range(episodes):
        episode_rewards = 0

        state, info = env.reset()
        for step in range(max_steps):
            # choose action using epsilon-greedy
            action = epsilon_greedy(Q, state, epsilon)
            # get next state and reward from the environment
            next_state, reward, done, info, truncated = env.step(action)

            # update reward if falling into a hole
            if done and reward == 0:
                reward = -1

            episode_rewards += reward

            # update Q-table using the Q-learning formula
            next_value = np.max(Q[next_state])
            Q[state, action] *= 1 - alpha
            Q[state, action] += alpha * (reward + gamma * next_value)

            # move to the next state
            state = next_state

            if done:
                break

        # decay epsilon for next episode
        epsilon = (min_epsilon + (1 - min_epsilon)
                   * np.exp(-epsilon_decay * episode))

        total_rewards.append(episode_rewards)

    return Q, total_rewards
