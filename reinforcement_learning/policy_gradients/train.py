#!/usr/bin/env python3
"""
    Policy Gradient Training Script
"""
import numpy as np
policy_gradient = __import__('policy_gradient').policy_gradient


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
    weight = np.random.rand(*env.observation_space.shape, env.action_space.n)

    scores = []

    for episode in range(1, nb_episodes + 1):
        obs, _ = env.reset()
        state = obs[None, :]
        grad = np.zeros_like(weight)
        score = 0
        done = False

        while not done:
            action, delta_grad = policy_gradient(state, weight)

            new_state, reward, done, info, truncated = env.step(action)
            new_state = new_state[None, :]

            score += reward

            grad += delta_grad

            weight += alpha * grad * (
                (reward + gamma * np.max(new_state.dot(weight)) * (not done))
                - state.dot(weight)[0, action]
            )

            state = new_state

        scores.append(score)

        print(f"Episode: {episode}, Score: {score}", end="\r", flush=True)

        if show_result and episode % 1000 == 0:
            env.render()

    return scores
