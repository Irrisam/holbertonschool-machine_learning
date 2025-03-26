#!/usr/bin/env python3
"""
    Policy Gradient Training Script
"""
import numpy as np
from policy_gradient import policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False):
    """
        Runs the policy gradient training loop.

    :param env: the environment instance
    :param nb_episodes: number of episodes for training
    :param alpha: learning rate
    :param gamma: discount factor
    :param show_result: whether to render the environment every 1000 episodes

    :return: list of episode scores
    """
    # Initialize random weights based on env state/action space
    weight = np.random.rand(*env.observation_space.shape, env.action_space.n)

    # Track scores over episodes
    scores = []

    for episode in range(1, nb_episodes + 1):
        state = env.reset()[None, :]  # Reset env and format state
        grad = np.zeros_like(weight)  # Initialize gradient
        score = 0  # Reset score counter
        done = False

        while not done:
            # Get action and gradient from policy
            action, delta_grad = policy_gradient(state, weight)

            # Take action in env
            new_state, reward, done, _ = env.step(action)
            new_state = new_state[None, :]

            # Update episode score
            score += reward

            # Accumulate gradients
            grad += delta_grad

            # Update weights using policy gradient update rule
            weight += alpha * grad * (
                (reward + gamma * np.max(new_state.dot(weight)) * (not done))
                - state.dot(weight)[0, action]
            )

            # Move to the next state
            state = new_state

        # Store score for tracking progress
        scores.append(score)

        # Display episode progress
        print(f"Episode: {episode}, Score: {score}", end="\r", flush=True)

        # Render env every 1000 episodes if enabled
        if show_result and episode % 1000 == 0:
            env.render()

    return scores
