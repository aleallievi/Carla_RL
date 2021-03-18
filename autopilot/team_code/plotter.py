"""
lib
"""
import os
from collections import deque
import numpy as np

from PIL import Image, ImageDraw

# autopilot
from base import *

"""
define
"""
DISPLACEMENT = 5.5

"""
class
"""

"""
plotter:  visualization only
"""
class plotter(object):
  def __init__(self, size):
    print("INIT PLOTTER")
    self.display_plotter = DISPLAY_PLOTTER

    if(DISPLAY_PLANNER):
      self.display_plotter = DISPLAY_PLANNER

    self.size = size
    self.clear()
    self.title = str(self.size)

  def clear(self):
    if not self.display_plotter:
      return
    """
    lib
    """
    self.img = Image.fromarray(np.zeros((self.size, 
                                         self.size, 
                                         RGB_CHANNEL_NUM), 
                               dtype=np.uint8))
    self.draw = ImageDraw.Draw(self.img)

  def dot(self, pos, node, 
          color=(MAX_INTENSITY, 
                 MAX_INTENSITY, 
                 MAX_INTENSITY), 
          r=2):
    x, y = DISPLACEMENT * (pos - node)
    x += self.size / 2
    y += self.size / 2
    self.draw.ellipse((x-r, y-r, x+r, y+r), color)

  def show(self):
    if self.display_plotter:
      return
    """
    lib
    """
    import cv2

    cv2.imshow(self.title, 
               cv2.cvtColor(np.array(self.img), 
               cv2.COLOR_BGR2RGB))
    cv2.waitKey(1)
