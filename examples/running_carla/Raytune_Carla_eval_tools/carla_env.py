#!/usr/bin/env python

# Copyright (c) 2021 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.


from __future__ import print_function
import gym
import argparse
import gym
from gym.spaces import Discrete, Box, MultiBinary, Dict, MultiDiscrete
import numpy as np

from .carla_core import CarlaCore
import carla

class CarlaEnv(gym.Env):
    """
    This is a carla environment, responsible of handling all the CARLA related steps of the training.
    """
    #TODO: MUST HANDLE ADDING ARGS "config": {"args": args,},
    def __init__(self, config):
        """Initializes the environment"""
        self.config = config
        self.action_space = Box(np.array([0,-1]),np.array([1,1]))

        observation_space_dict = {
            'img': Box(low=0, high=255,shape=(84, 84, 3), dtype=np.float64), #TODO: 1 is number of samples when it should be batch size!
            'velocity_mag': Box(low=-1, high=30,shape=(1,), dtype=np.float32), #assumes max velocity is 30 m/s
            'd2target': Box(low=0, high=100000,shape=(1,), dtype=np.float32), #assumes max d2target is 100000m
            'pitch': Box(low=-360, high=360,shape=(1,), dtype=np.float32),
            'yaw': Box(low=-360, high=360,shape=(1,), dtype=np.float32),
            'roll': Box(low=-360, high=360,shape=(1,), dtype=np.float32),
            'command':MultiDiscrete([2,2,2,2,2,2])
        }
        #observation_space_dict = {'img': Box(low=0, high=255,shape=(1,80, 80, 3), dtype=np.uint8)}

        self.observation_space = Dict(observation_space_dict)

        args = self.config["args"]
        args.client = self.launch_client(args)
        self.core = CarlaCore(args,save_video=False,i=1)
        #CarlaCore(self.config['carla'])
        # self.core.setup_experiment(self.experiment.config)

        self.reset()

    def launch_client(self,args):
        client = carla.Client(args.host, args.world_port)
        client.set_timeout(args.client_timeout)
        return client

    def state2obs(self, s):
        observation = {'img': s[0][0],
                        'velocity_mag': [s[1]],
                        'd2target': [s[2]],
                        'pitch': [s[3]],
                        'yaw': [s[4]],
                        'roll': [s[5]],
                        'command': s[6:]}
        return observation

    def reset(self):
        # print ("TRYING TO RESET!")
        s, _, _, _ = self.core.reset(False, 0)
        # print ("SUCCESSFULLY RESET!")
        return self.state2obs(s)

    def step(self, action):
        """Computes one tick of the environment in order to return the new observation,
        as well as the rewards"""
        s_prime, reward, done, info = self.core.step(action=action, timeout=2)
        return self.state2obs(s_prime), reward, done, info