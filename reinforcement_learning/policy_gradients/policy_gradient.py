#!/usr/bin/env python3
"""
    Policy Gradient
"""

import numpy as np


def policy(matrix, weight):
    """
        policy building with weit and matrix

    :param matrix: matrix, state
    :param weight: ndarray, weight to apply in policy

    :return: matrix of proba for each possible action
    """
    matrix = np.atleast_2d(matrix)
    z = matrix @ weight
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    softmax = exp_z / np.sum(exp_z, axis=1, keepdims=True)

    return softmax


def policy_gradient(state, weight):
    """
        Monte-carlo policy gradient

    :param state: matrix, current observation of the environment
    :param weight: matrix of random weights

    :return: chosen action and its corresponding gradient
    """
    state = np.atleast_2d(state)
    probs = policy(state, weight)

    action = np.random.choice(probs.shape[1], p=probs[0])

    grad = np.zeros_like(weight)

    for vert in range(state.shape[1]):
        for hor in range(weight.shape[1]):
            grad[vert, hor] = state[0, vert] * ((1 - probs[0, hor])
                                                if hor == action
                                                else -probs[0, hor])

    return action, grad
