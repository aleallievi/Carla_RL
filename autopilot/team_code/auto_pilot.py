"""
lib
"""
# standard
import os
import time
import datetime
import pathlib

# ml/vis
import cv2
import numpy as np

# autopilot
from team_code.map_agent import MapAgent
from team_code.pid_controller import pid_controller 
from team_code.base import *

"""
defines
"""
SAVE_LOG = False
SAVE_FPS = 100
RENDER_FPS = 20
LOG_FPS = 20
WEATHER_UPDATE = 100
DEBUG = False
MAP_SIZE = 256
MAP_SIZE = int(MAP_SIZE / 4)
RANDOM_SEED = 1337

# PID 
DEFAULT_TURN_PID_K_P = 1.25
DEFAULT_TURN_PID_K_I = 0.75
DEFAULT_TURN_PID_K_D = 0.3
DEFAULT_TURN_WINDOW_N = 40

DEFAULT_SPEED_PID_K_P = 5.0
DEFAULT_SPEED_PID_K_I = 0.5
DEFAULT_SPEED_PID_K_D = 1.0
DEFAULT_SPEED_WINDOW_N = 40

# throttle
NO_THROTTLE = 0.0
MIN_THROTTLE = 0.0
MAX_THROTTLE = 0.75

# speed
NO_SPEED = 0.0
MID_SPEED = 4.0
MAX_SPEED = 7.0

# steer
MIN_STEER = -1.0
MAX_STEER = 1.0
LARGE_FARBY_STEER = 45.0
LARGE_NEARBY_STEER = 5.0

# obstacles
OBSTACLE_FREE = None
NORMAL_ANGLE_DEG = 90

# EPSILON
EPSILON_ONE = 1e-1
EPSILON_TWO = 1e-2
EPSILON_THREE = 1e-3
EPSILON_FOUR = 1e-4

"""
carla entry
"""
def get_entry_point():
  return 'AutoPilot'

"""
class
"""
class AutoPilot(MapAgent):
  def setup(self, path_to_conf_file):
    super().setup(path_to_conf_file)
    self.save_path = None

    """
    file i/o
    """
    if path_to_conf_file:
      now = datetime.datetime.now()
      string = pathlib.Path(os.environ['ROUTES']).stem + '_'
      string += '_'.join(map(lambda x: '%02d' % x, 
                             (now.month, 
                              now.day, 
                              now.hour, 
                              now.minute, 
                              now.second)))
      print(string)
      self.save_path = pathlib.Path(path_to_conf_file) / string
      self.save_path.mkdir(exist_ok=False)
      (self.save_path / CENTER_CAMERA).mkdir()
      (self.save_path / LEFT_CAMERA).mkdir()
      (self.save_path / RIGHT_CAMERA).mkdir()
      (self.save_path / 'topdown').mkdir()
      (self.save_path / 'measurements').mkdir()

  """
  init
  """
  def _init(self):
    print("INIT AUTOPILOT")
    super()._init()

    np.random.seed(RANDOM_SEED) # seed not working

    self.obstacles = 0
    self.walker_obstacles = 0
    self.vehicle_obstacles = 0
    self.light_obstacles = 0
    self.speeds = 0
    self.brakes = 0
    self.reduce_speed = 0
    self.red_lights = 0
    self.stop_signs = 0
    self.weather_key = None

    self._turn_controller = pid_controller(K_P=DEFAULT_TURN_PID_K_P, 
                                           K_I=DEFAULT_TURN_PID_K_I,
                                           K_D=DEFAULT_TURN_PID_K_D,
                                           n=DEFAULT_TURN_WINDOW_N,)

    self._speed_controller = pid_controller(K_P=DEFAULT_SPEED_PID_K_P, 
                                            K_I=DEFAULT_SPEED_PID_K_I, 
                                            K_D=DEFAULT_SPEED_PID_K_D, 
                                            n=DEFAULT_SPEED_WINDOW_N)
  """
  UNUSED
  """
  def random_npcs(self):
    env.reset(n_vehicles=np.random.choice([50, 100, 200]),
              n_pedestrians=np.random.choice([50, 100, 200]),
              seed=np.random.randint(0, 256))
  """
  episode step
  """
  def run_step(self, sensors, timestamp):
    if not self.initialized:
      self._init()

    # update environment
    self.update_environment()

    # get sensor observations
    observations = self.get_observations(sensors)

    # apply policy
    return self.take_action(self.policy(observations))

  """
  states
  """
  def update_environment(self):
    if self.step % WEATHER_UPDATE == 0:
      weathers = list(WEATHERS.keys())
      index = (self.step // WEATHER_UPDATE) % len(weathers)
      k = weathers[index]
      self.weather_key = k
      self._world.set_weather(WEATHERS[k])

  """
  estimate current state by reading observations from sensors
  """
  def get_observations(self, sensors):
    return self.process_sensor_readings(sensors)

  """
  return: processed sensor data - from parent, parent base_agent
  """
  def process_sensor_readings(self, sensors):
    # calls base_agent to populate observation data
    return self.tick(sensors)

  """
  compute furthest nearby waypoint, and furthest farby waypoint
  (with command RoadOption.labels)
  """
  def plan_trajectory(self, sensor_data):
    gps = self._get_position(sensor_data)
    near_node, near_command = self.furthest_nearby_waypoint(gps)
    far_node, far_command = self.furthest_farby_waypoint(gps)
    return near_node, near_command, far_node, far_command 

  def furthest_nearby_waypoint(self, gps):
    # interpolated route 
    return self._waypoint_planner.run_step(gps)

  def furthest_farby_waypoint(self, gps):
    # non interpolated route 
    return self._command_planner.run_step(gps)

  """
  action to control
  """
  def take_action(self, actions):
    throttle, brake, steer = actions
    control = carla.VehicleControl()
    control.steer = steer + EPSILON_TWO * np.random.randn() # seed in init
    control.throttle = throttle
    control.brake = float(brake)
    return control 

  """
  policy
  """
  def policy(self, observations):
    return self.rule_based_policy(observations)

  def rule_based_policy(self, observations):
    """
    update trajectory, command RoadOption.labels
    """
    near_target, near_command, far_target, far_command = self.plan_trajectory(observations)

    """ 
    estimate state
    """
    # pose, velocity
    pos, theta, speed = self.pose_estimation(observations)

    """
    compute angles to near and far waypoint targets. unnorms 
    steering limits determine throttle
    """
    angle, angle_unnorm, angle_far_unnorm = self.update_angles(pos, theta, near_target, far_target)

    """
    update actions 
    """
    # brake
    brake = self.update_brake()

    # steering
    steer = self.update_steering(angle, brake)

    # throttle
    throttle, target_speed  = self.update_throttle(speed, brake, angle_unnorm, angle_far_unnorm) 

    """
    debug: performance tracking
    """
    fps = self.update_framerate()
    """
    visualization, file logging
    """
    self.output(fps, observations, far_target, near_command, 
                steer, throttle, brake, target_speed)
    """
    console logging
    """
    if self.step % LOG_FPS == 0:
      print("Steps:", self.step)
      print("FPS:", fps)
      print("Weather:", self.weather_key)
      print("Near:", near_command)
      print("Near Target:", near_target)
      print("Far:", far_command)
      print("Far Target:", far_target)
      print("Position:", pos)
      print("Heading:", theta)
      print("Speed:", speed)
      print("Mean Speed:", self.speeds/self.step)
      print("Throttle:", throttle)
      print("Brake:", brake)
      print("Steer:", steer)
      print("Brakes:", self.brakes)
      print("Reduce Speed:", self.reduce_speed)
      print("Vehicles:", self.vehicle_obstacles)
      print("Lights:", self.light_obstacles)
      print("Red Lights:", self.red_lights)
      print("Stop Signs:", self.stop_signs)
      print("Pedestrian:", self.walker_obstacles)
      print("Total Obstacles:", self.obstacles)

      route_record = self.manager.query_statistics()
      print("Scenario Stats:")
      print("Pedestrian Collisions:", len(route_record.infractions['collisions_pedestrian']))
      print("Vehicle Collisions:", len(route_record.infractions['collisions_vehicle']))
      print("Layout Collisions:", len(route_record.infractions['collisions_layout']))
      print("Outside route lanes:", len(route_record.infractions['outside_route_lanes']))
      print("Route deviation:", len(route_record.infractions['route_dev']))
      print("Route timeout:", len(route_record.infractions['route_timeout']))
      print("Vehicle Blocked:", len(route_record.infractions['vehicle_blocked']))
      print("Run red light:", len(route_record.infractions['red_light']))
      print("Run stop sign:", len(route_record.infractions['stop_infraction']))
      print(route_record.scores)
      print()

    """
    return actions
    """
    return (throttle, brake, steer)

  """
  estimate state
  """
  def pose_estimation(self, observations):
    # Position
    pos = self._get_position(observations)

    # Heading
    theta = observations['compass']

    # Speed
    speed = observations['speed']
    self.speeds += speed
    return pos, theta, speed

  def update_angles(self, pos, theta, near_target, far_target):
    angle_unnorm = self.rotate_by_theta(pos, theta, near_target)
    angle = angle_unnorm / NORMAL_ANGLE_DEG
    angle_far_unnorm = self.rotate_by_theta(pos, theta, far_target)
    return angle, angle_unnorm, angle_far_unnorm

  """
  update actions
  """
  def update_brake(self):
    actors = self._world.get_actors()

    vehicle = self._is_vehicle_obstacle(actors.filter('*vehicle*'))
    light = self._is_light_red(actors.filter('*traffic_light*'))
    walker = self._is_walker_obstacle(actors.filter('*walker*'))

    if vehicle is not OBSTACLE_FREE:
      self.vehicle_obstacles += 1
      self.obstacles += 1
    if light is not OBSTACLE_FREE:
      self.light_obstacles += 1
      self.obstacles += 1
    if walker is not OBSTACLE_FREE:
      self.walker_obstacles += 1
      self.obstacles += 1
    brake = any(x is not OBSTACLE_FREE for x in [vehicle, light, walker])
    if brake:
      self.brakes +=1
    return brake

  def update_steering(self, angle, brake):
    steer = self._turn_controller.step(angle)
    steer = np.clip(steer, MIN_STEER, MAX_STEER)
    steer = round(steer, 3)
    if brake:
      steer *= 0.5
    return steer

  def update_throttle(self, speed, brake, angle_unnorm, angle_far_unnorm):
    # if large farby steering or large nearby steering, then slow down
    reduce_speed =  ((abs(angle_far_unnorm) > LARGE_FARBY_STEER) or
                        (abs(angle_unnorm) > LARGE_NEARBY_STEER))
    if(reduce_speed):
      print("Reduce speed due to large farby steering or nearby steering risk")
      self.reduce_speed +=1

    # Slow Down or Max Speed
    target_speed = MID_SPEED if reduce_speed else MAX_SPEED
    target_speed = target_speed if not brake else NO_SPEED

    if not brake:
      delta = np.clip(target_speed - speed, 0.0, 0.25)
      throttle = self._speed_controller.step(delta)
      throttle = np.clip(throttle, MIN_THROTTLE, MAX_THROTTLE)
    else:
      throttle = NO_THROTTLE
    
    return throttle, target_speed

  """
  output logging
  """
  def update_framerate(self):
    return (self.step / (time.time() - self.wall_start))

  def output(self, fps, observations, far_target, near_command, 
             steer, throttle, brake, target_speed):
    _draw = None
    if HAS_DISPLAY:
      """
      lib
      """
      from carla_project.src.common import CONVERTER, COLOR
      from PIL import Image, ImageDraw

      """
      _draw.text((5, 90), 'Speed: %.3f' % speed)
      _draw.text((5, 110), 'Target: %.3f' % target_speed)
      _draw.text((5, 130), 'Angle: %.3f' % angle_unnorm)
      _draw.text((5, 150), 'Angle Far: %.3f' % angle_far_unnorm)
      """

      topdown = observations['topdown']
      rgb = np.hstack((observations[LEFT_CAMERA],
                       observations[CENTER_CAMERA],
                       observations[RIGHT_CAMERA]))

      _topdown = Image.fromarray(COLOR[CONVERTER[topdown]])
      _rgb = Image.fromarray(rgb)
      _draw = ImageDraw.Draw(_topdown)

      _topdown.thumbnail((MAP_SIZE, MAP_SIZE))
      _rgb = _rgb.resize((int(MAP_SIZE/ _rgb.size[1] * _rgb.size[0]), MAP_SIZE))

      _combined = Image.fromarray(np.hstack((_rgb, _topdown)))
      _draw = ImageDraw.Draw(_combined)
      _draw.text((5, 10), 'FPS: %.3f' % fps)

      """
      _draw.text((5, 30), 'Steer: %.3f' % steer)
      _draw.text((5, 50), 'Throttle: %.3f' % throttle)
      _draw.text((5, 70), 'Brake: %s' % brake)
      """
      if self.step % RENDER_FPS == 0:
        if HAS_DISPLAY:
          cv2.imshow('map', cv2.cvtColor(np.array(_combined),
                                         cv2.COLOR_BGR2RGB))
          cv2.waitKey(1)
    """
    logging
    """
    if SAVE_LOG and self.step % SAVE_FPS == 0:
      self.save(far_target,
                near_command,
                steer,
                throttle,
                brake,
                target_speed,
                observations)

  """
  collision detection
  """
  def get_collision(self, p1, v1, p2, v2):
    A = np.stack([v1, -v2], 1)
    b = p2 - p1

    if abs(np.linalg.det(A)) < EPSILON_THREE:
      return False, None

    x = np.linalg.solve(A, b)
    collides = all(x >= 0) and all(x <= 1)
    return collides, p1 + x[0] * v1

  """
  obstacle 
  """
  def _is_light_red(self, lights_list):
    # red or yellow
    if self._vehicle.get_traffic_light_state() != carla.libcarla.TrafficLightState.Green:
      affecting = self._vehicle.get_traffic_light()

      # street light
      for light in self._traffic_lights:
        if light.id == affecting.id:
          self.red_lights += 1 
          return affecting

      # stop sign todo: fix bug
      if(self.is_stop() is not OBSTACLE_FREE):
        self.stop_signs += 1
        affecting = 1
        return affecting

    return OBSTACLE_FREE

  """
  stop sign
  """
  def is_stop(self):
    ego_wp = self._world.get_map().get_waypoint(self._vehicle.get_location())
    light_id = self._vehicle.get_traffic_light().id if self._vehicle.get_traffic_light() is not None else -1
    if ego_wp.is_junction and light_id != -1:
      return light_id
    return OBSTACLE_FREE

  def _is_walker_obstacle(self, walkers_list):
    z = self._vehicle.get_location().z
    p1 = self.to_numpy(self._vehicle.get_location())
    v1 = 10.0 * self._orientation(self._vehicle.get_transform().rotation.yaw)

    if HAS_DISPLAY:
      self._draw_line(p1, v1, z+2.5, RGB_BLUE)

    for walker in walkers_list:
      v2_hat = self._orientation(walker.get_transform().rotation.yaw)
      s2 = np.linalg.norm(self.to_numpy(walker.get_velocity()))

      if s2 < 0.05:
        v2_hat *= s2

      p2 = -3.0 * v2_hat + self.to_numpy(walker.get_location())
      v2 = 8.0 * v2_hat

      if HAS_DISPLAY:
        self._draw_line(p2, v2, z+2.5)

      collides, collision_point = self.get_collision(p1, v1, p2, v2)
      if collides:
        return walker

    return OBSTACLE_FREE 

  def _is_vehicle_obstacle(self, vehicle_list):
    z = self._vehicle.get_location().z
    o1 = self._orientation(self._vehicle.get_transform().rotation.yaw)
    p1 = self.to_numpy(self._vehicle.get_location())
    s1 = max(7.5, 2.0 * np.linalg.norm(self.to_numpy(self._vehicle.get_velocity())))
    v1_hat = o1
    v1 = s1 * v1_hat

    if HAS_DISPLAY: 
      self._draw_line(p1, v1, z+2.5, RGB_RED)

    for target_vehicle in vehicle_list:
      if target_vehicle.id == self._vehicle.id:
        continue

      o2 = self._orientation(target_vehicle.get_transform().rotation.yaw)
      p2 = self.to_numpy(target_vehicle.get_location())
      s2 = max(5.0, 2.0 * np.linalg.norm(self.to_numpy(target_vehicle.get_velocity())))
      v2_hat = o2
      v2 = s2 * v2_hat

      p2_p1 = p2 - p1
      distance = np.linalg.norm(p2_p1)
      p2_p1_hat = p2_p1 / (distance + EPSILON_FOUR)

      if HAS_DISPLAY:
        self._draw_line(p2, v2, z+2.5, RGB_RED)

      angle_to_car = np.degrees(np.arccos(v1_hat.dot(p2_p1_hat)))
      angle_between_heading = np.degrees(np.arccos(o1.dot(o2)))

      if angle_between_heading > 60.0 and not (angle_to_car < 15 and distance < s1):
        continue
      elif angle_to_car > 30.0:
        continue
      elif distance > s1:
        continue
      return target_vehicle
    return OBSTACLE_FREE 

  """
  localization
  """
  def _location(self, x, y, z):
    return carla.Location(x=float(x),
      y=float(y), z=float(z))

  """
  pose
  """
  def _orientation(self, yaw):
    return np.float32([np.cos(np.radians(yaw)),
      np.sin(np.radians(yaw))])

  """
  conversion
  """
  def rotate_by_theta(self, pos, theta, target):
    R = np.array([
                 [np.cos(theta), -np.sin(theta)],
                 [np.sin(theta),  np.cos(theta)],
                 ])

    aim = R.T.dot(target - pos)
    angle = -np.degrees(np.arctan2(-aim[1], aim[0]))
    angle = 0.0 if np.isnan(angle) else angle
    return angle

  """
  util
  """
  def to_numpy(self, carla_vector, normalize=False):
    result = np.float32([carla_vector.x, carla_vector.y])
    if normalize:
      return result / (np.linalg.norm(result) + EPSILON_FOUR)
    return result

  """
  vis
  """
  def _draw_line(self, p, v, z, color=RGB_RED):
    if not DEBUG:
      return
    p1 = self._location(p[0], p[1], z)
    p2 = self._location(p[0]+v[0], p[1]+v[1], z)
    color = carla.Color(*color)
    self._world.debug.draw_line(p1, p2, 0.25, color, 0.01)

  """
  logging
  """
  def save(self, far_node, near_command, steer, throttle, brake, target_speed, observations):
    frame = self.step // 10
    pos = self._get_position(observations)
    theta = observations['compass']
    speed = observations['speed']

    data = {
            'x': pos[0],
            'y': pos[1],
            'theta': theta,
            'speed': speed,
            'target_speed': target_speed,
            'x_command': far_node[0],
            'y_command': far_node[1],
            'command': near_command.value,
            'steer': steer,
            'throttle': throttle,
            'brake': brake,
            }
    (self.save_path / 'measurements' / ('%04d.json' % frame)).write_text(str(data))

    if HAS_DISPLAY:
      from PIL import Image
      Image.fromarray(tick_data[CENTER_CAMERA]).save(self.save_path / CENTER_CAMERA / ('%04d.png' % frame))
      Image.fromarray(tick_data[LEFT_CAMERA]).save(self.save_path / LEFT_CAMERA / ('%04d.png' % frame))
      Image.fromarray(tick_data[RIGHT_CAMERA]).save(self.save_path / RIGHT_CAMERA / ('%04d.png' % frame))
      Image.fromarray(tick_data['topdown']).save(self.save_path / 'topdown' / ('%04d.png' % frame))
