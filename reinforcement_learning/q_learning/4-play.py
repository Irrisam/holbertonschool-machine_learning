#!/usr/bin/env python3
"""
    module to play frozen-lake
"""
import numpy as np


def play(env, Q, max_steps=100):
    """
        lets the trained agent play an episode

    :param env: FrozenLakeEnv instance
    :param Q: ndarray, the Q-table
    :param max_steps: max number of steps allowed per episode

    :return: total rewards earned in the episode
    """

    # initial state
    state = env.reset()

    total_rewards = 0

    for step in range(max_steps):
        # render current state
        env.render()

        # pick the best action from Q-table
        action = np.argmax(Q[state, :])

        # take the action and get new state and reward
        new_state, reward, done, info = env.step(action)

        # accumulate reward
        total_rewards += reward

        # update state
        state = new_state

        if done:
            break

    return total_rewards
