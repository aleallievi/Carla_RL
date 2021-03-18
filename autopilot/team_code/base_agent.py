"""
lib
"""
import time

import cv2
import team_code.base
from leaderboard.autoagents import autonomous_agent

# autopilot
from team_code.planner import routeplanner 
from base import *

"""
class
"""

class base_agent(autonomous_agent.AutonomousAgent):
  def setup(self, path_to_conf_file):
    print("SETUP BASEAGENT")
    self.track = autonomous_agent.Track.SENSORS
    self.config_path = path_to_conf_file
    self.step = -1
    self.wall_start = time.time()
    self.initialized = False
    self.manager = None

  def _init(self):
    print("INIT BASEAGENT")
    self._command_planner = routeplanner(7.5, 25.0, 257)
    self._command_planner.set_route(self._global_plan, True)
    self.initialized = True

  def _get_position(self, tick_data):
    gps = tick_data[GPS]
    gps = (gps - self._command_planner.mean) * self._command_planner.scale
    return gps

  def sensors(self):
    return [
                {
                    'type': 'sensor.camera.rgb',
                    'x': 1.3, 'y': 0.0, 'z': 1.3,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'width': 256, 'height': 144, 'fov': 90,
                    'id': 'rgb'
                    },
                {
                    'type': 'sensor.camera.rgb',
                    'x': 1.2, 'y': -0.25, 'z': 1.3,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': -45.0,
                    'width': 256, 'height': 144, 'fov': 90,
                    'id': 'rgb_left'
                    },
                {
                    'type': 'sensor.camera.rgb',
                    'x': 1.2, 'y': 0.25, 'z': 1.3,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 45.0,
                    'width': 256, 'height': 144, 'fov': 90,
                    'id': 'rgb_right'
                    },
                {
                    'type': 'sensor.other.imu',
                    'x': 0.0, 'y': 0.0, 'z': 0.0,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'sensor_tick': 0.05,
                    'id': 'imu'
                    },
                {
                    'type': 'sensor.other.gnss',
                    'x': 0.0, 'y': 0.0, 'z': 0.0,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'sensor_tick': 0.01,
                    'id': 'gps'
                    },
                {
                    'type': 'sensor.speedometer',
                    'reading_frequency': 20,
                    'id': 'speed'
                    }
                ]

  """
  bug: causes map_agent to "sensor()" twice
  """
  def sensors_broken(self):
    return SENSORS

  """
  tick
  """
  def tick(self, input_data):
    self.step += 1

    rgb = cv2.cvtColor(input_data[CENTER_CAMERA][1][:, :, :3], 
                       cv2.COLOR_BGR2RGB)
    rgb_left = cv2.cvtColor(input_data[LEFT_CAMERA][1][:, :, :3], 
                            cv2.COLOR_BGR2RGB)
    rgb_right = cv2.cvtColor(input_data[RIGHT_CAMERA][1][:, :, :3], 
                             cv2.COLOR_BGR2RGB)
    gps = input_data[GPS][1][:2]
    speed = input_data[SPEEDOMETER][1][SPEEDOMETER]
    imu = input_data[IMU][1][-1]

    return {
            CENTER_CAMERA: rgb,
            LEFT_CAMERA: rgb_left,
            RIGHT_CAMERA: rgb_right,
            GPS: gps,
            SPEEDOMETER: speed,
            'compass':imu 
           }
