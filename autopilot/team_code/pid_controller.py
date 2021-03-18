"""
#pid https://www.ni.com/en-us/innovations/white-papers/06/pid-theory-explained.html
"""

"""
lib
"""
from collections import deque
import numpy as np

# autopilot
from base import *

"""
define
"""
INIT_VALUE = 0.0
KEY_WAIT = 1
IMG_DIM_W  = 100
IMG_DIM_H = IMG_DIM_W - 1
DEFAULT_WINDOW_N = 20
MIN_WINDOW_LEN = 2
LAST_ELEM = -1
NEXT_TO_LAST_ELEM =  LAST_ELEM - 1

DEFAULT_K_P = 1.0
DEFAULT_K_I = 0.0
DEFAULT_K_D = 0.0

"""
class
"""
class pid_controller(object):
  def __init__(self, 
               K_P=DEFAULT_K_P, 
               K_I=DEFAULT_K_I, 
               K_D=DEFAULT_K_D, 
               n=DEFAULT_WINDOW_N):
    self._K_P, self._K_I, self._K_D, self.n = K_P, K_I, K_D, n
    self.init_error_window()

  def init_error_window(self):
    self._window = deque([0 for _ in range(self.n)], 
                         maxlen=self.n)
    self.reset_extrema()

  """
  update control every time step
  """
  def step(self, error):
    self.track_error_in_window(error)
    return self.pid()
 
  def pid(self):
    self.track_extrema()
    return self.calculate_update()

  """
  proportional: last error
  """
  def calculate_proportional(self):
    return self.last_error()

  """
  integral: error window mean
  """
  def calculate_integral(self):
    return np.mean(self._window)

  """
  derivative : diff last two errors
  """
  def calculate_derivative(self):
    return (self.last_error() - 
            self.next_to_last_error())

  def calculate_update(self):
    return (self._K_P * self.calculate_proportional() +
            self._K_I * self.calculate_integral() +
            self._K_D * self.calculate_derivative())

  """
  utils track error
  """
  def track_error_in_window(self, error):
    self._window.append(error)

  def last_error(self):
    return self._window[LAST_ELEM]

  def next_to_last_error(self):
    return self._window[NEXT_TO_LAST_ELEM]

  def track_extrema(self):
    if len(self._window) >= MIN_WINDOW_LEN:
      self.update_extrema()
    else:
      self.reset_extrema()

  def update_extrema(self):
    self._max = max(self._max, abs(self.last_error()))
    self._min = -abs(self._max)

  def reset_extrema(self):
    self._max = self._min = INIT_VALUE

  """
  UNUSED original code: broken?
  """
  def generate_visualization_data(self):
    """
    lib
    """
    import cv2

    canvas = np.ones((IMG_DIM_W, IMG_DIM_W, 3), dtype=np.uint8)
    w = int(canvas.shape[1] / len(self._window))
    h = IMG_DIM_H

    for i in range(1, len(self._window)):
      y1 = (self._max - self._window[i-1]) / (self._max - self._min + 1e-8)
      y2 = (self._max - self._window[i]) / (self._max - self._min + 1e-8)
      cv2.line(
        canvas,
        ((i-1) * w, int(y1 * h)),
        ((i) * w, int(y2 * h)),
        (255, 255, 255), 2)
    canvas = np.pad(canvas, ((5, 5), (5, 5), (0, 0)))
    cv2.imshow('%.3f %.3f %.3f' % (self._K_P, self._K_I, self._K_D), canvas)
    cv2.waitKey(KEY_WAIT)
