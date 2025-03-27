#!/usr/bin/env python3
"""
    module to play q-learning
"""
import numpy as np


def play(env, Q, max_steps=100):
    """
        lets the trained agent play an episode

    :param env: FrozenLakeEnv instance
    :param Q: ndarray, the Q-table
    :param max_steps: max number of steps allowed per episode

    :return: total rewards earned in episode and list of rendered outputs
    """
    # initial state
    state, _ = env.reset()
    full_encouragment = 0
    viewable_feedbacks = []

    for step in range(max_steps):
        # render current state
        rendered_output = env.render()
        viewable_feedbacks.append(rendered_output)

        # pick the best action from Q-table
        action = np.argmax(Q[state, :])

        # take the action and get new state and reward
        new_state, reward, done, info, truncated = env.step(action)

        # accumulate reward
        full_encouragment += reward

        # update state
        state = new_state

        if done:
            rendered_output = env.render()
            viewable_feedbacks.append(rendered_output)
            break

    return full_encouragment, viewable_feedbacks
