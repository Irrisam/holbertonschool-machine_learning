#!/usr/bin/env python3
"""
    Policy Gradient
"""

import numpy as np


def policy(matrix, weight):
    """
        function that computes to policy with a weight of a matrix

    :param matrix: matrix, state
    :param weight: ndarray, weight to apply in policy

    :return: matrix of proba for each possible action
    """
    # matrix product: score for each possible action
    z = matrix @ weight

    # Softmax: normalize exp scores = distribution proba of action
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    softmax = exp_z / np.sum(exp_z, axis=1, keepdims=True)

    return softmax

def policy_gradient(state, weight):
    """
        Computes the Monte-Carlo policy gradient.

    :param state: matrix, current observation of the environment
    :param weight: matrix of random weights

    :return: chosen action and its corresponding gradient
    """
    # Get policy probabilities
    probs = policy(state, weight)

    # Pick an action based on probabilities
    action = np.random.choice(probs.shape[1], p=probs[0])

    # Initialize gradient matrix
    grad = np.zeros_like(weight)

    # Compute gradients for each weight
    for i in range(state.shape[1]):
        for j in range(weight.shape[1]):
            grad[i, j] = state[0, i] * ((1 - probs[0, j]) if j == action else -probs[0, j])

    return action, grad
