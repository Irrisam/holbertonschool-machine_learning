#!/usr/bin/env python3
"""
    training function for breakout game by atari
"""
from __future__ import division
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing
from keras.optimizers.legacy import Adam
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Permute
import matplotlib.pyplot as plt
import pickle

from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.util import *
from rl.core import Processor

seed = 42
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.1
epsilon_max = 1.0
epsilon_interval = (epsilon_max - epsilon_min)
batch_size = 32
max_steps_per_episode = 10000
max_episodes = 10

class CompatibilityWrapper(gym.Wrapper):
    """
        Compatibility wrapper for
        older gym env version
    """

    def step(self, action):
        """
            Perform one step

        :param action: action taken in env

        :return: tuple with observation, reward, done, additional info and truncated factor
        """
        observation, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return observation, reward, done, info

    def reset(self, **kwargs):
        """
            Reset the environment and return

        :param kwargs: any extra arguments

        :return: initial observation
        """
        observation, info = self.env.reset(**kwargs)
        return observation


def atari_env_builder(env_name):
    """
    
        env setup
    
    :param env_name: Atari environment name

    :return: gym.Env: processed Atari environment
    """
    env = gym.make(env_name, render_mode='rgb_array')
    env = AtariPreprocessing(env, screen_size=84, grayscale_obs=True, frame_skip=1, noop_max=30)
    env = CompatibilityWrapper(env)
    return env


def build_model(window_length, shape, actions):
    """
        Build a CNN model for RL

    :param window_length: frames stacked as input
    :param shape: input image shape (height, width, channels)
    :param actions: number of possible actions in the environment

    :return: keras.models.Sequential: compiled CNN model
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
        Custom processor for Atari environment
    """

    def process_observation(self, observation):
        """
            Convert observation into a numpy array

        :param observation: input observation from environment

        :return: processed observation
        """
        if isinstance(observation, tuple):
            observation = observation[0]

        pic = np.array(observation).astype('uint8')
        return pic

    def process_state_batch(self, batch):
        """
            normalize batches

        :param batch: batch of states

        :return: normalized batch
        """
        return batch.astype('float32') / 255.

    def process_reward(self, reward):
        """
            reward to be between -1 and 1

        :param reward: incoming reward

        :return: clipped reward
        """
        return np.clip(reward, -1., 1.)


if __name__ == "__main__":
    env = atari_env_builder('ALE/Breakout-v5')
    observation = env.reset()

    plt.imshow(observation, cmap='gray')
    plt.title("Initial Observation")
    plt.axis('off')
    plt.show()

    nb_actions = env.action_space.n

    window_length = 4
    model = build_model(window_length, observation.shape, nb_actions)

    memory = SequentialMemory(limit=1000000, window_length=window_length)
    processor = AtariProcessor()

    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps',
                                  value_max=1., value_min=.1,
                                  value_test=.05, nb_steps=1000000)

    dqn = DQNAgent(model=model,
                   nb_actions=nb_actions,
                   policy=policy,
                   memory=memory,
                   processor=processor,
                   nb_steps_warmup=50000,
                   gamma=.99,
                   target_model_update=10000,
                   train_interval=4,
                   delta_clip=1.)

    dqn.compile(Adam(learning_rate=0.00025), metrics=['mae'])


    history = dqn.fit(env, nb_steps=1000000, visualize=False, verbose=2)


    dqn.save_weights('policy.h5', overwrite=True)

    with open('training_history.pkl', 'wb') as f:
        pickle.dump(history.history, f)

    plt.figure(figsize=(14, 7))
    plt.subplot(2, 1, 1)
    plt.plot(history.history['episode_reward'])
    plt.title('Episode Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')

    plt.tight_layout()
    plt.savefig('training_performance.png')
    plt.show()


    test_env = gym.make('ALE/Breakout-v5', render_mode='human')
    test_env = AtariPreprocessing(test_env, screen_size=84, grayscale_obs=True,
                                  frame_skip=4, noop_max=30)

    scores = dqn.test(test_env, nb_episodes=10, visualize=True)
    print(f'Average score over 10 test episodes: {np.mean(scores.history["episode_reward"])}')

    env.close()
