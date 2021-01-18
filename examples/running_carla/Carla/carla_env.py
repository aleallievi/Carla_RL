import collections
import queue
import numpy as np
import cv2
import carla
import argparse
import logging
import time
import math
import random
from sklearn.neighbors import KDTree
from shapely.geometry import LineString
from PIL import Image, ImageDraw
import sys

from traffic_events import TrafficEventType
from statistics_manager import StatisticManager
#IK this is bad, fix file path stuff later :(
sys.path.append("/scratch/cluster/stephane/Carla_0.9.10/PythonAPI/carla/agents/navigation")
from global_route_planner import GlobalRoutePlanner
from global_route_planner_dao import GlobalRoutePlannerDAO
#from scripts.launch_carla import launch_carla_server
#from scripts.kill_carla import kill_carla
from score_tests import RouteCompletionTest, InfractionsTests


class CarlaEnv(object):
    def __init__(self, args, town='Town01', save_video=False):
        # Tunable parameters
        self.FRAME_RATE = 5.0  # in Hz
        self.MAX_EP_LENGTH = 60  # in seconds
        self.MAX_EP_LENGTH = self.MAX_EP_LENGTH / (1.0 / self.FRAME_RATE)  # convert to ticks

        self._client = args.client

        self._town_name = town
        self._world = self._client.load_world(town)
        self._map = self._world.get_map()
        self._blueprints = self._world.get_blueprint_library()
        self._spectator = self._world.get_spectator()
        self._car_agent_model = self._blueprints.filter("model3")[0]
        self.save_video = save_video

        self.command2onehot =\
            {"RoadOption.LEFT":             [0, 0, 0, 0, 0, 1],
             "RoadOption.RIGHT":            [0, 0, 0, 0, 1, 0],
             "RoadOption.STRAIGHT":         [0, 0, 0, 1, 0, 0],
             "RoadOption.LANEFOLLOW":       [0, 0, 1, 0, 0, 0],
             "RoadOption.CHANGELANELEFT":   [0, 1, 0, 0, 0, 0],
             "RoadOption.CHANGELANERIGHT":  [1, 0, 0, 0, 0, 0]
             }

        self.DISTANCE_LIGHT = 15
        self.PROXIMITY_THRESHOLD = 50.0  # meters
        self.SPEED_THRESHOLD = 0.1
        self.WAYPOINT_STEP = 1.0  # meters
        self.ALLOWED_OUT_DISTANCE = 1.3          # At least 0.5, due to the mini-shoulder between lanes and sidewalks
        self.MAX_ALLOWED_VEHICLE_ANGLE = 120.0   # Maximum angle between the yaw and waypoint lane
        self.MAX_ALLOWED_WAYPOINT_ANGLE = 150.0  # Maximum change between the yaw-lane angle between frames
        self.WINDOWS_SIZE = 3   # Amount of additional waypoints checked (in case the first on fails)

        self.init()

    def __enter__(self):
        self.frame = self.set_sync_mode(True)
        return self

    def __exit__(self, *args):
        """
        Make sure to set the world back to async,
        otherwise future clients might have trouble connecting.
        """
        self._cleanup()
        self.set_sync_mode(False)

    def init(self, randomize=False, i=0):
        self._settings = self._world.get_settings()    
        self.reward = None

        # vehicle, sensor
        self._actor_dict = collections.defaultdict(list)
        # self.rgb_img = np.reshape(np.zeros(80*80*3), [1, 80, 80, 3]) # DEBUG

        self._tick = 0
        self._car_agent = None

        if(randomize):
            self._settings.set(SendNonPlayerAgentsInfo=True, NumberOfVehicles=random.randrange(30),
                              NumberOfPedestrians=random.randrange(30), WeatherId=random.randrange(14))
            self._settings.randomize_seeds()
            self._world.apply_settings(self._settings)

        self.blocked_start = 0
        self.blocked = False
        self.last_col_time = 0
        self.last_col_id = 0
        self.n_step_cols = 0

        self.collisions = []
        self.followed_waypoints = []

        # spawn car at random location
        np.random.seed(49)
        self._start_pose = np.random.choice(self._map.get_spawn_points())

        self._current_velocity = None

        self._spawn_car_agent()
        print('car agent spawned')
        self._setup_sensors()
        print('sensors created')

        # create random target to reach
        np.random.seed(6)
        self._target_pose = np.random.choice(self._map.get_spawn_points())
        while(self._target_pose is self._start_pose):
            self.target = np.random.choice(self._map.get_spawn_points())
        self.get_route()
        # create statistics manager
        self.statistics_manager = StatisticManager(self.route_waypoints)
        # get all initial waypoints
        self._pre_ego_waypoint = self._map.get_waypoint(self._car_agent.get_location())
        self._time_start = time.time()

        # create sensor queues
        self._queues = []
        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)
        # make_queue(self._world.on_tick)
        for sensor in self._actor_dict['camera']:
            make_queue(sensor.listen)

        #TODO: Ciuld this be causing problems?
        for i in range(10):
            self._world.tick()

        self.target_waypoint_idx = 0
        self.at_waypoint = []
        self.followed_target_waypoints = []
        self.dist_to_target_wp_tr = None

        self.completion_test = RouteCompletionTest(self._car_agent, self.route_waypoints_unformatted, self._map)
        self.infractions_test = InfractionsTests(self._car_agent,self._map,self._world)

    def reset(self, randomize, i):
        # self._cleanup()
        # self.init()
        return self.step(timeout=2)

    def _spawn_car_agent(self):
        self._car_agent = self._world.try_spawn_actor(self._car_agent_model, self._start_pose)
        # handle invalid spawn point
        while self._car_agent is None:
            self._start_pose = random.choice(self._map.get_spawn_points())
            self._car_agent = self._world.try_spawn_actor(self._car_agent_model, self._start_pose)
        self._actor_dict['car_agent'].append(self._car_agent)

    def draw_waypoints(self,world, waypoints):
        for w in waypoints:
            t = w.transform
            begin = t.location + carla.Location(z=0.5)
            angle = math.radians(t.rotation.yaw)
            end = begin + carla.Location(x=math.cos(angle), y=math.sin(angle))
            world.debug.draw_arrow(begin, end, arrow_size=0.3, life_time=1.0)

    def _setup_sensors(self, height=480, width=640, fov=10):
        sensor_relative_transform = carla.Transform(carla.Location(x=2.5, z=0.7))

        # get camera sensor
        self.rgb_cam = self._blueprints.find("sensor.camera.rgb")
        self.rgb_cam.set_attribute("image_size_x", f"{width}")
        self.rgb_cam.set_attribute("image_size_y", f"{height}")
        self.rgb_cam.set_attribute("fov", f"{fov}")
        self._rgb_cam_sensor = self._world.spawn_actor(self.rgb_cam, sensor_relative_transform, attach_to=self._car_agent)
        self._actor_dict['camera'].append(self._rgb_cam_sensor)

        # get collision sensor
        col_sensor_bp = self._blueprints.find("sensor.other.collision")
        self._col_sensor = self._world.spawn_actor(col_sensor_bp, sensor_relative_transform, attach_to=self._car_agent)
        self._actor_dict['col_sensor'].append(self._col_sensor)
        self._col_sensor.listen(lambda event: self.handle_collision(event))

        # get obstacle sensor
        obs_sensor_bp = self._blueprints.find("sensor.other.obstacle")
        self._obs_sensor = self._world.spawn_actor(obs_sensor_bp, sensor_relative_transform, attach_to=self._car_agent)
        self._actor_dict['obs_sensor'].append(self._obs_sensor)
        self._obs_sensor.listen(lambda event: self.handle_obstacle(event))

        self.out = None
        if self.save_video:
            print ("saving video turned on")
            self.draw_waypoints(self._world,self.route_waypoints_unformatted)
            #self.cap = cv2.VideoCapture(0)
            fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
            self.out = cv2.VideoWriter("episode_footage/output_"+str(iter)+".avi", fourcc,FPS, (height+60,width))
            self.n_img = 0

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(block=True, timeout=timeout)
            if data.frame == self.frame:
                sensor_queue.task_done()
                # print(self.frame)
                return data

    def step(self, timeout, action=None):
        self.started_sim = True
        # spectator camera with overhead view of ego vehicle
        spectator_rot = self._car_agent.get_transform().rotation
        spectator_rot.pitch -= 10
        self._spectator.set_transform(carla.Transform
                                      (self._car_agent.get_transform().location + carla.Location(z=2), spectator_rot)
                                      )

        if action is not None:
            self._car_agent.apply_control(carla.VehicleControl(throttle=action[0][0], steer=action[0][1]))
        else:
            self._car_agent.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0))

        self.frame = self._world.tick()
        # time.sleep(0.5)
        self._tick += 1

        transform = self._car_agent.get_transform()
        velocity = self._car_agent.get_velocity()
        rotation = transform.rotation

        dist_completed, wp_index, is_route_completed = self.completion_test.update()
        command_encoded = self.command2onehot.get(str(self.route_commands[wp_index]))
        d2target = self.statistics_manager.route_record['route_length'] - dist_completed

        velocity_kmh = int(3.6*np.sqrt(np.power(velocity.x, 2) + np.power(velocity.y, 2) + np.power(velocity.z, 2)))
        velocity_mag = np.sqrt(np.power(velocity.x, 2) + np.power(velocity.y, 2) + np.power(velocity.z, 2))
        self.cur_velocity = velocity_mag

        # print ("QUEUE")
        # print (self._queues[0].queue)

        state = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in state)
        # state = [self.rgb_cam]
        state = [self.process_img(img, 80, 80) for img in state]
        state = [state, velocity_mag, d2target, rotation.pitch, rotation.yaw, rotation.roll]
        state.extend(command_encoded)

        # check for traffic light infraction/stoplight infraction
        InfractionsTests.check_traffic_light_infraction()
        InfractionsTests.check_stop_sign_infraction()

        # check if the vehicle is either on a sidewalk or at a wrong lane.
        InfractionsTests.check_outside_route_lane()

        # get done information
        if self._tick > self.MAX_EP_LENGTH or self.infractions_test.colllided_w_static:
            done = True
            self.infractions_test.events.append([TrafficEventType.ROUTE_COMPLETION, dist_completed])
        elif is_route_completed:
            done = True
            self.infractions_test.events.append([TrafficEventType.ROUTE_COMPLETED])
        else:
            done = False
            self.infractions_test.events.append([TrafficEventType.ROUTE_COMPLETION, dist_completed])

        # print(self.events)
        # get reward information
        self.statistics_manager.compute_route_statistics(time.time(), self.infractions_test.events)

        #------------------------------------------------------------------------------------------------------------------
        reward = self.statistics_manager.route_record["score_composed"] - self.statistics_manager.prev_score
        self.reward = reward
        # DEBUG
        # if reward != 0:
        #     print(f'score_route:{self.statistics_manager.route_record["score_route"]}')
        #     print(f'score_penalty:{self.statistics_manager.route_record["score_penalty"]}')
        #     print(f'score_composed:{self.statistics_manager.route_record["score_composed"]}')
        #     print(f'prev_score:{self.statistics_manager.prev_score}')
        self.statistics_manager.prev_score = self.statistics_manager.route_record["score_composed"]
        #reward = self.statistics_manager.route_record["score_composed"]
        #self.events.clear()
        #------------------------------------------------------------------------------------------------------------------
        # reward = 1000*(self.d_completed - self.statistics_manager.prev_d_completed) + 0.05*(velocity_kmh-self.statistics_manager.prev_velocity_kmh) - 10*self.statistics_manager.route_record["score_penalty"]
        # self.statistics_manager.prev_d_completed = self.d_completed
        # self.statistics_manager.prev_velocity_kmh = velocity_kmh
        #------------------------------------------------------------------------------------------------------------------
        #reset is blocked if car is moving
        if self.cur_velocity > 0 and self.blocked:
            self.blocked = False
            self.blocked_start = 0
        self.n_step_cols = 0

        return state, reward, done, [self.statistics_manager.route_record["score_composed"],
                                     self.statistics_manager.route_record['route_percentage'], self.infractions_test.n_collisions,
                                     self.infractions_test.n_tafficlight_violations, self.infractions_test.n_stopsign_violations, self.infractions_test.n_route_violations,
                                     self.infractions_test.n_vehicle_blocked]

    def _cleanup(self):
        """
        Remove and destroy all actors
        """
        print('cleaning up..')

        # TODO why doesn't this work?
        # self._client.apply_batch_sync([carla.command.DestroyActor(x[0]) for x in self._actor_dict.values()])

        [x[0].destroy() for x in self._actor_dict.values()]

        for q in self._queues:
            with q.mutex:
                q.queue.clear()

        self._actor_dict.clear()
        self._queues.clear()
        self._tick = 0
        self._time_start = time.time()
        self._car_agent = None
        self._spectator = None
        # self._world.tick()
        if self.out != None:
            self.out.release()

    def set_sync_mode(self, sync):
        settings = self._world.get_settings()
        settings.synchronous_mode = sync
        settings.fixed_delta_seconds = 1.0 / self.FRAME_RATE
        frame = self._world.apply_settings(settings)
        return frame

    def get_route(self):
        dao = GlobalRoutePlannerDAO(self._map, 0.5)
        grp = GlobalRoutePlanner(dao)
        grp.setup()
        route = dict(grp.trace_route(self._start_pose.location, self._target_pose.location))

        self.route_waypoints = []
        self.route_commands = []
        self.route_waypoints_unformatted = []
        for waypoint in route.keys():
            self.route_waypoints.append((waypoint.transform.location.x, waypoint.transform.location.y,
                                         waypoint.transform.location.z))
            self.route_commands.append(route.get(waypoint))
            self.route_waypoints_unformatted.append(waypoint)
        self.route_kdtree = KDTree(np.array(self.route_waypoints))

    def process_img(self, img, height, width):
        img_reshaped = np.frombuffer(img.raw_data, dtype='uint8').reshape(480, 640, 4)
        rgb_reshaped = img_reshaped[:, :, :3]
        rgb_reshaped = cv2.resize(rgb_reshaped,(height,width))
        rgb_f = rgb_reshaped[:, :, ::-1]
        if self.save_video and self.reward != None and self.started_sim and 'route_percentage' in self.statistics_manager.route_record:
            #img = np.frombuffer(img.raw_data, dtype='uint8').reshape(height, width, 4)
            rgb = np.frombuffer(img.raw_data, dtype='uint8').reshape(480, 640, 4)
            rgb = rgb[:, :, :3]
            #percent complete
            rgb_mat = cv2.UMat(rgb)
            rgb_mat = cv2.copyMakeBorder(rgb_mat, 60,0,0,0, cv2.BORDER_CONSTANT, None, 0)
            cv2.putText(rgb_mat, "Route % complete: " + str(self.statistics_manager.route_record['route_percentage']), (2,10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(rgb_mat, "Step reward: " + str(self.reward), (2,25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(rgb_mat, "Total reward: " + str(self.total_reward), (2,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(rgb_mat, "WP index: " + str(self.statistics_manager._current_index), (2,55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            # rgb = rgb.reshape(480+60,640,3)
            rgb_mat = cv2.resize(rgb_mat,(480+60,640))
            self.out.write(rgb_mat)
            #cv2.imwrite("/scratch/cluster/stephane/cluster_quickstart/examples/running_carla/episode_footage/frame_"+str(iter)+str(self.n_img)+".png",rgb)
            self.n_img+=1
        return rgb_f

    '''Evaluation tools'''

    def handle_collision(self, event):
        distance_vector = self._car_agent.get_location() - event.other_actor.get_location()
        distance = math.sqrt(math.pow(distance_vector.x, 2) + math.pow(distance_vector.y, 2))

        if not (self.last_col_id == event.other_actor.id and time.time() - self.last_col_time < 1) and self.n_step_cols < 2:
            self.n_step_cols += 1
            self.collisions.append(event)
            self.infractions_test.n_collisions += 1
            self.last_col_id = event.other_actor.id
            self.last_col_time = time.time()

            if ("pedestrian" in event.other_actor.type_id):
                self.infractions_test.events.append([TrafficEventType.COLLISION_PEDESTRIAN])
            if ("vehicle" in event.other_actor.type_id):
                self.infractions_test.events.append([TrafficEventType.COLLISION_VEHICLE])
            if ("static" in event.other_actor.type_id):
                self.infractions_test.events.append([TrafficEventType.COLLISION_STATIC])
                self.infractions_test.colllided_w_static = True

    def handle_obstacle(self, event):
        if event.distance < 0.5 and self.cur_velocity == 0:
            if self.blocked == False:
                self.blocked = True
                self.blocked_start = time.time()
            else:
                #if the car has been blocked for more that 180 seconds
                if time.time() - self.blocked_start > 180:
                    self.infractions_test.events.append([TrafficEventType.VEHICLE_BLOCKED])
                    #reset
                    self.blocked = False
                    self.blocked_start = 0
                    self.infractions_test.n_vehicle_blocked += 1
