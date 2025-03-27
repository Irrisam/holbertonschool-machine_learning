#!/usr/bin/env python3
"""
    load environment from gym
"""
import gymnasium as gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """
         inits pre-builtenv

        :param desc: None or list, custom description of the map (optional)
        :param map_name: None or string, choose a pre-made map (optional)
        :param is_slippery: bool, defines if the ice is slippery or not

        :return: the environment object
    """
    env = gym.make('FrozenLake-v1',
                   desc=desc,
                   map_name=map_name,
                   is_slippery=is_slippery,
                   render_mode="ansi")

    return env
