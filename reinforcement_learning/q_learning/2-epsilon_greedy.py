#!/usr/bin/env python3
"""
    module to implement epsilon-greedy
"""
import gymnasium as gym
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
        uses epsilon-greedy to pick the next action

        :param Q: ndarray, Q-table
        :param state: current state
        :param epsilon: epsilon value for exploration vs exploitation
        random.randint is used to choose an action during exploration

        :return: the index of the next action
    """
    # decide whether to explore or exploit
    p = np.random.uniform(0, 1)

    # exploration: pick a random action
    if p < epsilon:
        action = np.random.randint(0, Q.shape[1])
    # exploitation (p >= epsilon): choose the best action based on Q-table
    else:
        action = np.argmax(Q[state, :])

    return action
