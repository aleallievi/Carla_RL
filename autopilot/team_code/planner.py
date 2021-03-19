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
               min_distance,  # horizon start
               max_distance,  # horizon end
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
  global_plan from autonomous_agent.set_global_plan()
  """
  def set_route(self, global_plan, gps=GPS_ENABLED_DEFAULT):

    if(GPS_ENABLED_DEFAULT):
      print("GPS enabled, [lat lon] in global plan")
    else:
      print("GPS disabled, location.x, location.y in global_plan")

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
    print("planner: set_route(global_plan), gps: ", gps, " len:", len(self.route))

  def run_step(self, gps):
    self.debug.clear()

    if len(self.route) == 1:
      return self.route[0]

    to_pop = 0
    farthest_in_range = -np.inf
    cumulative_distance = 0.0

    """
    loop through route waypoints
    """
    for i in range(1, len(self.route)):

      # return if passed visualization max distance
      if cumulative_distance > self.max_distance:
        break

      # calculate distance between current waypoint and previous
      cumulative_distance += np.linalg.norm(self.route[i][0] - 
                                            self.route[i-1][0])

      # distance between go and current route waypoint 
      distance = np.linalg.norm(self.route[i][0] - gps)

      # cache furthest within min_distance (change var name) as waypoint node target
      if distance <= self.min_distance and distance > farthest_in_range:
        farthest_in_range = distance 
        to_pop = i # change variable names

      r = MAX_INTENSITY * int(distance > self.min_distance)
      g = MAX_INTENSITY * int(self.route[i][1].value == 4)
      b = MAX_INTENSITY 

      if self.display_planner: 
        self.debug.dot(gps, self.route[i][0], (r, g, b))

    """
    remove passed waypoints
    """
    for _ in range(to_pop):
      if len(self.route) > 2:
        self.route.popleft()

    if self.display_planner: 
      self.visualize_dots(gps)

    # current furthest within distance range waypoint
    return self.route[1]

  """
  visualize
  """
  def visualize_dots(self, gps):
    self.debug.dot(gps, self.route[0][0], RGB_GREEN)
    self.debug.dot(gps, self.route[1][0], RGB_RED)
    self.debug.dot(gps, gps, RGB_BLUE)
    self.debug.show()
