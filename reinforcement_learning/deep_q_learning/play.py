#!/usr/bin/env python3
"""
     game used by trained model on atari's breakout
"""
from __future__ import division

import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Permute
import time
import pygame
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import GreedyQPolicy
from rl.util import *
from rl.core import Processor
from rl.callbacks import Callback


#####################################
#            setup env              #
#####################################
class CompatibilityWrapper(gym.Wrapper):
    """
        wrapper to ensure compatibility with older gym versions
    """

    def step(self, action):
        """
            take a step in the env using the provided action

        :param action: action to perform in the env

        :return: tuple (observation, reward, done, info)
        """
        observation, reward, terminated, truncated, info = (
            self.env.step(action))
        done = terminated or truncated
        return observation, reward, done, info

    def reset(self, **kwargs):
        """
            reset the env and return the initial observation

        :param kwargs: additional args

        :return: initial observation
        """
        observation, info = self.env.reset(**kwargs)
        return observation


def atari_env_builder(env_name):
    """
        create arari env

    :param env_name: name of the atari environment

    :return: configured gym environment
    """
    env = gym.make(env_name, render_mode='rgb_array')
    env = AtariPreprocessing(env,
                             screen_size=84,
                             grayscale_obs=True,
                             frame_skip=1,
                             noop_max=30)
    env = CompatibilityWrapper(env)
    return env


def build_model(window_length, shape, actions):
    """
        cnn model for rl

    :param window_length: int, number of frames to stack
    :param shape: tuple, shape of the input image
    :param actions: int, number of possible actions

    :return: compiled keras model
    """
    model = Sequential()
    model.add(Permute((2, 3, 1), input_shape=(window_length,) + shape))
    model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu'))
    model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model


class AtariProcessor(Processor):
    """
        functionnement processing for console games
    """

    def process_observation(self, observation):
        """
            change feedback into np array

        :param observation: observation from env

        :return: processed observation
        """
        if isinstance(observation, tuple):
            observation = observation[0]

        pic = np.array(observation)
        pic = pic.astype('uint8')
        return pic

    def process_state_batch(self, batch):
        """
            normalize states

        :param batch: ndarray, batch of states

        :return: normalized batch
        """
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_reward(self, reward):
        """
           give rewards clipped

        :param reward: float, reward from env

        :return: clipped reward
        """
        return np.clip(reward, -1., 1.)


class PygameCallback(Callback):
    """
        show game by model
    """

    def __init__(self, env, delay=0.02):
        """
            init env and render delay

        :param env: gym env instance
        :param delay: time between frames in seconds
        """
        self.env = env
        self.delay = delay
        pygame.init()
        self.screen = pygame.display.set_mode((420, 320))
        pygame.display.set_caption("Atari Breakout - DQN Agent")

    def on_action_end(self, action, logs={}):
        """
            updates display after each action

        :param action: action taken
        :param logs: training logs
        """
        frame = self.env.render()
        surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        surf = pygame.transform.scale(surf, (420, 320))
        self.screen.blit(surf, (0, 0))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.env.close()
                pygame.quit()

        time.sleep(self.delay)

    def on_episode_end(self, episode, logs={}):
        """
            wait a bit between episodes

        :param episode: current episode number
        :param logs: training logs
        """
        pygame.time.wait(1000)


if __name__ == "__main__":
    env = atari_env_builder('ALE/Breakout-v5')
    nb_actions = env.action_space.n

    window_length = 4
    input_shape = (84, 84)
    model = build_model(window_length, input_shape, nb_actions
    )