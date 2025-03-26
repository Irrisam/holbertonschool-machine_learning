#!/usr/bin/env python3
"""
    module to load the frozen-lake environment from gym
"""
import gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """
        loads the pre-built FrozenLakeEnv environment from OpenAI's gym

        :param desc: None or list, custom description of the map (optional)
        :param map_name: None or string, choose a pre-made map (optional)
        :param is_slippery: bool, defines if the ice is slippery or not

        :return: the environment object
    """
    env = gym.make('FrozenLake-v0',
                   desc=desc,
                   map_name=map_name,
                   is_slippery=is_slippery)

    return env
