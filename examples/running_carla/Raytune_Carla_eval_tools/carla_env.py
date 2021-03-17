#!/usr/bin/env python

"""
import
"""
from __future__ import print_function
import gym
import argparse
import gym
from gym.spaces import Discrete, Box, MultiBinary, Dict, MultiDiscrete
import numpy as np

from .carla_core import CarlaCore
import carla

MIN_VELOCITY = -1 # m/s
MAX_VELOCITY = 30 # m/s

MIN_TARGET_DISTANCE = 0
MAX_TARGET_DISTANCE = 100000

MIN_PITCH = -360
MAX_PITCH = 360

MIN_YAW = -360
MAX_YAW = 360

MIN_ROLL = -360
MAX_ROLL = 360

IMG_DIM = 84
RGB_CHANNELS = 3

MIN_INTENSITY 0
MAX_INTENSITY 255

COMMAND = [2,2,2,2,2,2]


"""
Class
"""
class CarlaEnv(gym.Env):
  # TODO: MUST HANDLE ADDING ARGS "config": {"args": args,},
  def __init__(self, config):
    self.config = config

    self.action_space = Box(np.array([0,-1]),np.array([1,1]))
    observation_space_dict = {
      # TODO: 1 is number of samples when it should be batch size!
      'img': Box(low=MIN_INTENSITY, 
                 high=MAX_INTENSITY,
                 shape=(IMG_DIM, IMG_DIM, RGB_CHANNELS), 
                 dtype=np.float64), 
      'velocity_mag': Box(low=MIN_VELOCITY, 
                          high=MAX_VELOCITY,
                          shape=(1,), 
                          dtype=np.float32),
      'd2target': Box(low=MIN_TARGET_DISTANCE, 
                      high=MAX_TARGET_DISTANCE,
                      shape=(1,), 
                      dtype=np.float32),
      'pitch': Box(low=-360, high=360,shape=(1,), dtype=np.float32),
      'yaw': Box(low=-360, high=360,shape=(1,), dtype=np.float32),
      'roll': Box(low=-360, high=360,shape=(1,), dtype=np.float32),

      'command':MultiDiscrete(COMMAND)
    }
    self.observation_space = Dict(observation_space_dict)

    args = self.config["args"]
    args.client = self.launch_client(args)
    self.core = CarlaCore(args,save_video=False,i=1)

    """
    intiliaze
    """
    self.reset()

  """
  launch
  """
  def launch_client(self,args):
    client = carla.Client(args.host, args.world_port)
    client.set_timeout(args.client_timeout)
    return client

  """
  pull obs directly from state
  """
  def state2obs(self, s):
    observation = {'img': s[0][0],
                   'velocity_mag': [s[1]],
                   'd2target': [s[2]],
                   'pitch': [s[3]],
                   'yaw': [s[4]],
                   'roll': [s[5]],
                   'command': s[6:]}
    return observation


  """
  standard
  """

  def reset(self):
    s, _, _, _ = self.core.reset(False, 0)
    return self.state2obs(s)

  def step(self, action):
    s_prime, reward, done, info = self.core.step(action=action, timeout=2)
    return self.state2obs(s_prime), reward, done, info
