"""
lib
"""
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

# autopilot
from team_code.base_agent import base_agent 
from team_code.planner import routeplanner
from carla_project.src.carla_env import draw_traffic_lights, get_nearby_lights

# autopilot
from base import *

"""
carla entry
"""
def get_entry_point():
  return 'MapAgent'

"""
class
"""
class MapAgent(base_agent):
  def sensors(self):
    result = super().sensors()
    result.append({'type': 'sensor.camera.semantic_segmentation',
                    'x': 0.0, 
                    'y': 0.0, 
                    'z': 100.0,
                    'roll': 0.0, 
                    'pitch': -90.0, 
                    'yaw': 0.0,
                    'width': 512, 
                    'height': 512, 
                    'fov': 5 * 10.0,
                    'id': 'map'
                 })
    return result

  """
  global plan
  """
  def set_global_plan(self, global_plan_gps, global_plan_world_coord):
    super().set_global_plan(global_plan_gps, global_plan_world_coord)

    self._plan_HACK = global_plan_world_coord
    self._plan_gps_HACK = global_plan_gps

  def _init(self):
    print("INIT MAPAGENT")
    super()._init()
    self._vehicle = CarlaDataProvider.get_hero_actor()

    self._waypoint_planner = routeplanner(4.0, 50)
    self._waypoint_planner.set_route(self._plan_gps_HACK, True)
    self._world = self._vehicle.get_world()

    self._traffic_lights = list()
    self._stop_signs = list()

  """
  tick
  """
  def tick(self, input_data):
    self._actors = self._world.get_actors()
    self._traffic_lights = get_nearby_lights(self.
                                            _vehicle, 
                                             self._actors.filter('*traffic_light*'))
    result = super().tick(input_data)
    if HAS_DISPLAY:
      topdown = input_data['map'][1][:, :, 2]
      topdown = draw_traffic_lights(topdown, 
                                    self._vehicle, 
                                    self._traffic_lights)
      result['topdown'] = topdown
    return result
