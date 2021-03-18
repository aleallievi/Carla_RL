"""
lib
"""
import os
from collections import deque
import numpy as np

# autopilot
from plotter import *
from base import *

"""
define
"""
# gps
GPS_ENABLED_DEFAULT  = False

# debug
DEFAULT_DEBUG_SIZE = 256

# visualization 
MIN_INTENSITY = 0
MAX_INTENSITY = 255
RGB_RED = (MAX_INTENSITY, MIN_INTENSITY, MIN_INTENSITY)
RGB_GREEN = (MIN_INTENSITY, MAX_INTENSITY, MIN_INTENSITY)
RGB_BLUE = (MIN_INTENSITY, MAX_INTENSITY, MAX_INTENSITY)

# magic
MAGIC_SCALE = [111324.60662786, 73032.1570362]
MAGIC_MEAN = [49.0, 8.0]

"""
class
"""
class routeplanner(object):
  def __init__(self, 
               min_distance, 
               max_distance, 
               debug_size=DEFAULT_DEBUG_SIZE):
    print("INIT ROUTEPLANNER")
    self.route = deque()
    self.min_distance, self.max_distance  = min_distance, max_distance
    self.display_planner = DISPLAY_PLANNER

    """
    ?????
    """
    self.mean = np.array(MAGIC_MEAN)
    self.scale = np.array(MAGIC_SCALE)
    self.debug = plotter(debug_size)

  """
  route 
  """
  def set_route(self, global_plan, gps=GPS_ENABLED_DEFAULT):
    self.route.clear()
    for pos, cmd in global_plan:
       if gps:
         pos = np.array([pos['lat'], pos['lon']])
         pos -= self.mean
         pos *= self.scale
       else:
         pos = np.array([pos.location.x, pos.location.y])
         pos -= self.mean
       self.route.append((pos, cmd))

  def run_step(self, gps):
    self.debug.clear()

    if len(self.route) == 1:
      return self.route[0]

    to_pop = 0
    farthest_in_range = -np.inf
    cumulative_distance = 0.0

    """
    loop
    """
    for i in range(1, len(self.route)):
      if cumulative_distance > self.max_distance:
        break
      #if

      cumulative_distance += np.linalg.norm(self.route[i][0] - 
                                            self.route[i-1][0])
      distance = np.linalg.norm(self.route[i][0] - gps)

      if distance <= self.min_distance and distance > farthest_in_range:
        farthest_in_range = distance
        to_pop = i
      #if

      r = MAX_INTENSITY * int(distance > self.min_distance)
      g = MAX_INTENSITY * int(self.route[i][1].value == 4)
      b = MAX_INTENSITY 

      if self.display_planner: 
        self.debug.dot(gps, self.route[i][0], (r, g, b))
      #if 
    #for

    for _ in range(to_pop):
      if len(self.route) > 2:
        self.route.popleft()
    #for

    if self.display_planner: 
      self.visualize_dots(gps)

    return self.route[1]
  # end def

  def visualize_dots(self, gps):
    self.debug.dot(gps, self.route[0][0], RGB_GREEN)
    self.debug.dot(gps, self.route[1][0], RGB_RED)
    self.debug.dot(gps, gps, RGB_BLUE)
    self.debug.show()
