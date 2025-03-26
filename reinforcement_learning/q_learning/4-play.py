#!/usr/bin/env python3
"""
    module to play q-learning algo
"""
import numpy as np

def play(env, Q, max_steps=100):
    """
        lets the trained agent play an episode

    :param env: FrozenLakeEnv instance
    :param Q: ndarray, the Q-table
    :param max_steps: max number of steps allowed per episode

    :return: total rewards earned in the episode, list of rendered outputs
    """

    # initial state
    state, _ = env.reset()  

    total_rewards = 0
    rendered_outputs = [] 

    for step in range(max_steps):
        # render current state and capture output
        rendered_output = env.render() 
        rendered_outputs.append(rendered_output)

        # pick the best action from Q-table
        action = np.argmax(Q[state, :])

        # take the action and get new state and reward
        new_state, reward, done, info, truncated = env.step(action)

        # accumulate reward
        total_rewards += reward

        # update state
        state = new_state

        if done or truncated:  # check for both done and truncated
            break

    return total_rewards, rendered_outputs