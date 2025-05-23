#!/usr/bin/env python3
"""
    module to initialize q-table
"""
import numpy as np


def q_init(env):
    """
        initializes the Q-table

        :param env: FrozenLakeEnv instance

        :return: Q-table as a numpy.ndarray filled with zeros
    """
    views = env.observation_space.n
    moves = env.action_space.n

    q_table = np.zeros((views, moves))

    return q_table
