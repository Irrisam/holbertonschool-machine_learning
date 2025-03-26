#!/usr/bin/env python3
"""
    module to initialize q-table
"""
import gymnasium as gym
import numpy as np


def q_init(env):
    """
        initializes the Q-table

        :param env: FrozenLakeEnv instance

        :return: Q-table as a numpy.ndarray filled with zeros
    """
    # possible observations
    views = env.observation_space.n
    # possible actions
    moves = env.action_space.n

    # create Q-table
    q_table = np.zeros((views, moves))

    return q_table
