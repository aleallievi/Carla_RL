"""
lib
"""
import carla
from sensors import *

"""
define
"""
# make dict
HAS_DISPLAY     = False # pass
DISPLAY_PLANNER = False # pass 
DISPLAY_PLOTTER = False # pass

# environments

TOWNS = {
  "Town01": "A basic town layout with all 'T junctions'.",
  "Town02": "Similar to Town01, but smaller.",
  "Town03": "The most complex town, with a 5-lane junction, a roundabout, unevenness, a tunnel, and much more. Essentially a medley.",
  "Town04": "An infinite loop with a highway and a small town.",
  "Town05": "Squared-grid town with cross junctions and a bridge. It has multiple lanes per direction. Useful to perform lane changes.",
  "Town06": "Long highways with many highway entrances and exits. It also has a Michigan left.",
  "Town07": "A rural environment with narrow roads, barely non traffic lights and barns.",
  "Town10": "A city environment with with different environments such as an avenue or a promenade, and more realistic textures."
}

WEATHERS = {
  "ClearNoon": carla.WeatherParameters.ClearNoon,
  "ClearSunset": carla.WeatherParameters.ClearSunset,

  "CloudyNoon": carla.WeatherParameters.CloudyNoon,
  "CloudSunset": carla.WeatherParameters.CloudySunset,

  "WetNoon": carla.WeatherParameters.WetNoon,
  "WetSunset": carla.WeatherParameters.WetSunset,

  "MidRainyNoon": carla.WeatherParameters.MidRainyNoon,
  "MidRainSunet": carla.WeatherParameters.MidRainSunset,

  "WetCloudyNoon": carla.WeatherParameters.WetCloudyNoon,
  "WetClousySunset": carla.WeatherParameters.WetCloudySunset,

  "HardRainNoon": carla.WeatherParameters.HardRainNoon,
  "HardRainSunset": carla.WeatherParameters.HardRainSunset,

  "SoftRainNoon": carla.WeatherParameters.SoftRainNoon,
  "SoftRainSunset": carla.WeatherParameters.SoftRainSunset,
}

# Visualization
MIN_INTENSITY = 0
MAX_INTENSITY = 255
RGB_RED = (MAX_INTENSITY, MIN_INTENSITY, MIN_INTENSITY)
RGB_GREEN = (MIN_INTENSITY, MAX_INTENSITY, MIN_INTENSITY)
RGB_BLUE = (MIN_INTENSITY, MAX_INTENSITY, MAX_INTENSITY)
RGB_CHANNEL_NUM = 3
