# !/usr/bin/env python

# Copyright (c) 2018-2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides all atomic evaluation criteria required to analyze if a
scenario was completed successfully or failed.
Criteria should run continuously to monitor the state of a single actor, multiple
actors or environmental parameters. Hence, a termination is not required.
The atomic criteria are implemented with py_trees.
"""

import math
import numpy as np
import shapely

import carla

# from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
# from srunner.scenariomanager.timer import GameTime
from traffic_events import TrafficEvent, TrafficEventType


class Criterion(object):

    """
    Base class for all criteria used to evaluate a scenario for success/failure
    Important parameters (PUBLIC):
    - name: Name of the criterion
    - expected_value_success:    Result in case of success
                                 (e.g. max_speed, zero collisions, ...)
    - expected_value_acceptable: Result that does not mean a failure,
                                 but is not good enough for a success
    - actual_value: Actual result after running the scenario
    - test_status: Used to access the result of the criterion
    - optional: Indicates if a criterion is optional (not used for overall analysis)
    """

    def __init__(self,
                 name,
                 actor,
                 _map
                 ):
        super(Criterion, self).__init__()

        self.name = name
        self.actor = actor
        self.map = _map
        self.actual_value = 0
        self.list_traffic_events = []


class RouteCompletionTest(Criterion):

    """
    Check at which stage of the route is the actor at each tick
    Important parameters:
    - actor: CARLA actor to be used for this test
    - route: Route to be checked
    - terminate_on_failure [optional]: If True, the complete scenario will terminate upon failure of this test
    """
    DISTANCE_THRESHOLD = 10.0  # meters
    WINDOWS_SIZE = 2

    def __init__(self, actor, route, _map, name="RouteCompletionTest", terminate_on_failure=False):
        """
        """
        super(RouteCompletionTest, self).__init__(name, actor, _map)
        self._actor = actor
        self._route = route
        self._map = _map

        self._wsize = self.WINDOWS_SIZE
        self._current_index = 0
        self._route_length = len(self._route)
        # self._waypoints, _ = zip(*self._route)
        self._waypoints = self._route
        self.target = self._waypoints[-1]

        self._accum_meters = []
        prev_wp = self._waypoints[0]
        for i, wp in enumerate(self._waypoints):
            d = wp.transform.location.distance(prev_wp.transform.location)
            if i > 0:
                accum = self._accum_meters[i - 1]
            else:
                accum = 0

            self._accum_meters.append(d + accum)
            prev_wp = wp

        self._traffic_event = TrafficEvent(event_type=TrafficEventType.ROUTE_COMPLETION)
        self.list_traffic_events.append(self._traffic_event)
        self._percentage_route_completed = 0.0
        self.is_route_completed = False

    def update(self):
        """
        Check if the actor location is within trigger region
        """

        # location = CarlaDataProvider.get_location(self._actor)
        location = self._actor.get_location()

        for index in range(self._current_index, min(self._current_index + self._wsize + 1, self._route_length)):

            # Get the dot product to know if it has passed this location
            ref_waypoint = self._waypoints[index].transform.location
            wp = self._map.get_waypoint(ref_waypoint)
            # wp = self._waypoints[index]
            wp_dir = wp.transform.get_forward_vector()          # Waypoint's forward vector
            wp_veh = location - ref_waypoint                    # vector waypoint - vehicle
            dot_ve_wp = wp_veh.x * wp_dir.x + wp_veh.y * wp_dir.y + wp_veh.z * wp_dir.z
            if dot_ve_wp > 0 or index == self._current_index:
                # DEBUG
                # print(f'wp idx:{index}')
                # increment completion based on cleared waypoint segment, include negative progress wrt current wp
                self._current_index = index
                self._percentage_route_completed = 100.0 * (
                            float(self._accum_meters[self._current_index]) + float(dot_ve_wp)) \
                                                   / float(self._accum_meters[-1])
                self._traffic_event.set_dict({
                    'route_completed': self._percentage_route_completed})
                self._traffic_event.set_message(
                    "Agent has completed > {:.2f}% of the route".format(
                        self._percentage_route_completed))

        if self._percentage_route_completed > 99.0 and location.distance(self.target) < self.DISTANCE_THRESHOLD:
            self.is_route_completed = True
            route_completion_event = TrafficEvent(event_type=TrafficEventType.ROUTE_COMPLETED)
            route_completion_event.set_message("Destination was successfully reached")
            self.list_traffic_events.append(route_completion_event)
            self._percentage_route_completed = 100

        self.actual_value = round(self._percentage_route_completed, 2)

        return self.actual_value, self._current_index, self.is_route_completed

class InfractionsTests():
    def __init__ (self,_car_agent,_map,_world,route_waypoints_unformatted,route_waypoints,_pre_ego_waypoint):
        self._car_agent = _car_agent
        self._map = _map
        self._world = _world
        self.route_waypoints_unformatted = route_waypoints_unformatted
        self.route_waypoints = route_waypoints
        self._pre_ego_waypoint = _pre_ego_waypoint

        self._list_traffic_lights = []
        self._list_stop_signs = []
        self._last_red_light_id = None
        self._target_stop_sign = None
        self._stop_completed = False
        self._affected_by_stop = False
        self.stop_actual_value = 0
        self.light_actual_value = 0
        self._outside_lane_active = False
        self._wrong_lane_active = False
        self._last_road_id = None
        self._last_lane_id = None
        self._total_distance = 0
        self._wrong_distance = 0
        self._current_index = 0
        self.events = []

        # some metrics for debugging
        self.colllided_w_static = False
        self.n_collisions = 0
        self.n_tafficlight_violations = 0
        self.n_stopsign_violations = 0
        self.n_route_violations = 0
        self.n_vehicle_blocked = 0

        self.DISTANCE_LIGHT = 15
        self.PROXIMITY_THRESHOLD = 50.0  # meters
        self.SPEED_THRESHOLD = 0.1
        self.WAYPOINT_STEP = 1.0  # meters
        self.ALLOWED_OUT_DISTANCE = 1.3          # At least 0.5, due to the mini-shoulder between lanes and sidewalks
        self.MAX_ALLOWED_VEHICLE_ANGLE = 120.0   # Maximum angle between the yaw and waypoint lane
        self.MAX_ALLOWED_WAYPOINT_ANGLE = 150.0  # Maximum change between the yaw-lane angle between frames
        self.WINDOWS_SIZE = 3   # Amount of additional waypoints checked (in case the first on fails)

        # TODO: decide whether to use actor lists or dict
        # Get all static actors in world
        all_actors = self._world.get_actors()
        for _actor in all_actors:
            if 'traffic_light' in _actor.type_id:
                center, waypoints = self.get_traffic_light_waypoints(_actor)
                self._list_traffic_lights.append((_actor, center, waypoints))
            if 'traffic.stop' in _actor.type_id:
                self._list_stop_signs.append(_actor)

    def check_traffic_light_infraction(self):
        transform = self._car_agent.get_transform()
        location = transform.location

        veh_extent = self._car_agent.bounding_box.extent.x
        tail_close_pt = self.rotate_point(carla.Vector3D(-0.8 * veh_extent, 0.0, location.z), transform.rotation.yaw)
        tail_close_pt = location + carla.Location(tail_close_pt)

        tail_far_pt = self.rotate_point(carla.Vector3D(-veh_extent - 1, 0.0, location.z), transform.rotation.yaw)
        tail_far_pt = location + carla.Location(tail_far_pt)

        for traffic_light, center, waypoints in self._list_traffic_lights:
            center_loc = carla.Location(center)
            if self._last_red_light_id and self._last_red_light_id == traffic_light.id:
                continue
            if center_loc.distance(location) > self.DISTANCE_LIGHT:
                continue
            if traffic_light.state != carla.TrafficLightState.Red:
                continue
            for wp in waypoints:
                tail_wp = self._map.get_waypoint(tail_far_pt)
                # Calculate the dot product (Might be unscaled, as only its sign is important)
                ve_dir = self._car_agent.get_transform().get_forward_vector()
                wp_dir = wp.transform.get_forward_vector()
                dot_ve_wp = ve_dir.x * wp_dir.x + ve_dir.y * wp_dir.y + ve_dir.z * wp_dir.z

                # Check the lane until all the "tail" has passed
                if tail_wp.road_id == wp.road_id and tail_wp.lane_id == wp.lane_id and dot_ve_wp > 0:
                    # This light is red and is affecting our lane
                    yaw_wp = wp.transform.rotation.yaw
                    lane_width = wp.lane_width
                    location_wp = wp.transform.location

                    lft_lane_wp = self.rotate_point(carla.Vector3D(0.4 * lane_width, 0.0, location_wp.z), yaw_wp + 90)
                    lft_lane_wp = location_wp + carla.Location(lft_lane_wp)
                    rgt_lane_wp = self.rotate_point(carla.Vector3D(0.4 * lane_width, 0.0, location_wp.z), yaw_wp - 90)
                    rgt_lane_wp = location_wp + carla.Location(rgt_lane_wp)

                    # Is the vehicle traversing the stop line?
                    if self.is_vehicle_crossing_line((tail_close_pt, tail_far_pt), (lft_lane_wp, rgt_lane_wp)):
                        self.light_actual_value += 1
                        # location = traffic_light.get_transform().location
                        #red_light_event = TrafficEvent(event_type=TrafficEventType.TRAFFIC_LIGHT_INFRACTION)
                        self.events.append([TrafficEventType.TRAFFIC_LIGHT_INFRACTION])
                        self._last_red_light_id = traffic_light.id
                        self.n_tafficlight_violations += 1
                        break

    def get_traffic_light_waypoints(self, traffic_light):
        """
        get area of a given traffic light
        """
        base_transform = traffic_light.get_transform()
        base_rot = base_transform.rotation.yaw
        area_loc = base_transform.transform(traffic_light.trigger_volume.location)

        # Discretize the trigger box into points
        area_ext = traffic_light.trigger_volume.extent
        x_values = np.arange(-0.9 * area_ext.x, 0.9 * area_ext.x, 1.0)  # 0.9 to avoid crossing to adjacent lanes

        area = []
        for x in x_values:
            point = self.rotate_point(carla.Vector3D(x, 0, area_ext.z), base_rot)
            point_location = area_loc + carla.Location(x=point.x, y=point.y)
            area.append(point_location)

        # Get the waypoints of these points, removing duplicates
        ini_wps = []
        for pt in area:
            wpx = self._map.get_waypoint(pt)
            # As x_values are arranged in order, only the last one has to be checked
            if not ini_wps or ini_wps[-1].road_id != wpx.road_id or ini_wps[-1].lane_id != wpx.lane_id:
                ini_wps.append(wpx)

        # Advance them until the intersection
        wps = []
        for wpx in ini_wps:
            while not wpx.is_intersection:
                next_wp = wpx.next(0.5)[0]
                if next_wp and not next_wp.is_intersection:
                    wpx = next_wp
                else:
                    break
            wps.append(wpx)

        return area_loc, wps

    def check_stop_sign_infraction(self):
        transform = self._car_agent.get_transform()
        location = transform.location
        if not self._target_stop_sign:
            # scan for stop signs
            self._target_stop_sign = self._scan_for_stop_sign()
        else:
            # we were in the middle of dealing with a stop sign
            if not self._stop_completed:
                # did the ego-vehicle stop?
                velocity = self._car_agent.get_velocity()
                current_speed = np.sqrt(np.power(velocity.x, 2) + np.power(velocity.y, 2) + np.power(velocity.z, 2))
                if current_speed < self.SPEED_THRESHOLD:
                    self._stop_completed = True

            if not self._affected_by_stop:
                stop_location = self._target_stop_sign.get_location()
                stop_extent = self._target_stop_sign.trigger_volume.extent

                if self.point_inside_boundingbox(location, stop_location, stop_extent):
                    self._affected_by_stop = True

            if not self.is_actor_affected_by_stop(self._car_agent, self._target_stop_sign):
                # is the vehicle out of the influence of this stop sign now?
                if not self._stop_completed and self._affected_by_stop:
                    # did we stop?
                    self.stop_actual_value += 1
                    #stop_location = self._target_stop_sign.get_transform().location
                    self.events.append([TrafficEventType.STOP_INFRACTION])
                    self.n_stopsign_violations += 1

                # reset state
                self._target_stop_sign = None
                self._stop_completed = False
                self._affected_by_stop = False

    def is_actor_affected_by_stop(self, actor, stop, multi_step=20):
        """
        Check if the given actor is affected by the stop
        """
        affected = False
        # first we run a fast coarse test
        current_location = actor.get_location()
        stop_location = stop.get_transform().location
        if stop_location.distance(current_location) > self.PROXIMITY_THRESHOLD:
            return affected

        stop_t = stop.get_transform()
        transformed_tv = stop_t.transform(stop.trigger_volume.location)

        # slower and accurate test based on waypoint's horizon and geometric test
        list_locations = [current_location]
        waypoint = self._map.get_waypoint(current_location)
        for _ in range(multi_step):
            if waypoint:
                next_wps = waypoint.next(self.WAYPOINT_STEP)
                if not next_wps:
                    break
                waypoint = next_wps[0]
                if not waypoint:
                    break
                list_locations.append(waypoint.transform.location)

        for actor_location in list_locations:
            if self.point_inside_boundingbox(actor_location, transformed_tv, stop.trigger_volume.extent):
                affected = True

        return affected

    def _scan_for_stop_sign(self):
        target_stop_sign = None

        ve_tra = self._car_agent.get_transform()
        ve_dir = ve_tra.get_forward_vector()

        wp = self._map.get_waypoint(ve_tra.location)
        wp_dir = wp.transform.get_forward_vector()

        dot_ve_wp = ve_dir.x * wp_dir.x + ve_dir.y * wp_dir.y + ve_dir.z * wp_dir.z

        if dot_ve_wp > 0:  # Ignore all when going in a wrong lane
            for stop_sign in self._list_stop_signs:
                if self.is_actor_affected_by_stop(self._car_agent, stop_sign):
                    # this stop sign is affecting the vehicle
                    target_stop_sign = stop_sign
                    break

        return target_stop_sign

    def check_outside_route_lane(self):
        self.test_status = None
        _waypoints = self.route_waypoints_unformatted #not sure if this is correct
        _route_length = len(self.route_waypoints)
        location = self._car_agent.get_location()
         # 1) Check if outside route lanes
        self._is_outside_driving_lanes(location)
        self._is_at_wrong_lane(location)
        if self._outside_lane_active or self._wrong_lane_active:
            self.test_status = "FAILURE"
        # 2) Get the traveled distance
        for index in range(self._current_index + 1,
                           min(self._current_index + self.WINDOWS_SIZE + 1, _route_length)):
            # Get the dot product to know if it has passed this location
            index_location = _waypoints[index].transform.location
            index_waypoint = self._map.get_waypoint(index_location)

            wp_dir = index_waypoint.transform.get_forward_vector()  # Waypoint's forward vector
            wp_veh = location - index_location  # vector waypoint - vehicle
            dot_ve_wp = wp_veh.x * wp_dir.x + wp_veh.y * wp_dir.y + wp_veh.z * wp_dir.z

            if dot_ve_wp > 0:
                # Get the distance traveled
                index_location = _waypoints[index].transform.location
                current_index_location = _waypoints[self._current_index].transform.location
                new_dist = current_index_location.distance(index_location)

                # Add it to the total distance
                self._current_index = index
                self._total_distance += new_dist

                # And to the wrong one if outside route lanes
                if self._outside_lane_active or self._wrong_lane_active:
                    self._wrong_distance += new_dist
        if self.test_status == "FAILURE" and self._total_distance != 0:
            self.events.append([TrafficEventType.OUTSIDE_ROUTE_LANES_INFRACTION,self._wrong_distance / self._total_distance * 100])
            self.n_route_violations += 1

    def _is_outside_driving_lanes(self, location):
        """
        Detects if the ego_vehicle is outside driving lanes
        """

        current_driving_wp = self._map.get_waypoint(location, lane_type=carla.LaneType.Driving, project_to_road=True)
        current_parking_wp = self._map.get_waypoint(location, lane_type=carla.LaneType.Parking, project_to_road=True)

        driving_distance = location.distance(current_driving_wp.transform.location)
        if current_parking_wp is not None:  # Some towns have no parking
            parking_distance = location.distance(current_parking_wp.transform.location)
        else:
            parking_distance = float('inf')

        if driving_distance >= parking_distance:
            distance = parking_distance
            lane_width = current_parking_wp.lane_width
        else:
            distance = driving_distance
            lane_width = current_driving_wp.lane_width

        self._outside_lane_active = bool(distance > (lane_width / 2 + self.ALLOWED_OUT_DISTANCE))

    def _is_at_wrong_lane(self, location):
        """
        Detects if the ego_vehicle has invaded a wrong lane
        """

        current_waypoint = self._map.get_waypoint(location, lane_type=carla.LaneType.Driving, project_to_road=True)
        current_lane_id = current_waypoint.lane_id
        current_road_id = current_waypoint.road_id

        # Lanes and roads are too chaotic at junctions
        if current_waypoint.is_junction:
            self._wrong_lane_active = False
        elif self._last_road_id != current_road_id or self._last_lane_id != current_lane_id:

            # Route direction can be considered continuous, except after exiting a junction.
            if self._pre_ego_waypoint.is_junction:
                yaw_waypt = current_waypoint.transform.rotation.yaw % 360
                yaw_actor = self._car_agent.get_transform().rotation.yaw % 360

                vehicle_lane_angle = (yaw_waypt - yaw_actor) % 360

                if vehicle_lane_angle < self.MAX_ALLOWED_VEHICLE_ANGLE \
                        or vehicle_lane_angle > (360 - self.MAX_ALLOWED_VEHICLE_ANGLE):
                    self._wrong_lane_active = False
                else:
                    self._wrong_lane_active = True

            else:
                # Check for a big gap in waypoint directions.
                yaw_pre_wp = self._pre_ego_waypoint.transform.rotation.yaw % 360
                yaw_cur_wp = current_waypoint.transform.rotation.yaw % 360

                waypoint_angle = (yaw_pre_wp - yaw_cur_wp) % 360

                if self.MAX_ALLOWED_WAYPOINT_ANGLE <= waypoint_angle <= (360 - self.MAX_ALLOWED_WAYPOINT_ANGLE):

                    # Is the ego vehicle going back to the lane, or going out? Take the opposite
                    self._wrong_lane_active = not bool(self._wrong_lane_active)
                else:

                    # Changing to a lane with the same direction
                    self._wrong_lane_active = False

        # Remember the last state
        self._last_lane_id = current_lane_id
        self._last_road_id = current_road_id
        self._pre_ego_waypoint = current_waypoint

    @staticmethod
    def rotate_point(point, angle):
        """
        rotate a given point by a given angle
        """
        x_ = math.cos(math.radians(angle)) * point.x - math.sin(math.radians(angle)) * point.y
        y_ = math.sin(math.radians(angle)) * point.x + math.cos(math.radians(angle)) * point.y
        return carla.Vector3D(x_, y_, point.z)

    @staticmethod
    def is_vehicle_crossing_line(seg1, seg2):
        """
        check if vehicle crosses a line segment
        """
        line1 = LineString([(seg1[0].x, seg1[0].y), (seg1[1].x, seg1[1].y)])
        line2 = LineString([(seg2[0].x, seg2[0].y), (seg2[1].x, seg2[1].y)])
        inter = line1.intersection(line2)

        return not inter.is_empty

    @staticmethod
    def point_inside_boundingbox(point, bb_center, bb_extent):
        """
        X
        :param point:
        :param bb_center:
        :param bb_extent:
        :return:
        """

        # pylint: disable=invalid-name
        A = carla.Vector2D(bb_center.x - bb_extent.x, bb_center.y - bb_extent.y)
        B = carla.Vector2D(bb_center.x + bb_extent.x, bb_center.y - bb_extent.y)
        D = carla.Vector2D(bb_center.x - bb_extent.x, bb_center.y + bb_extent.y)
        M = carla.Vector2D(point.x, point.y)

        AB = B - A
        AD = D - A
        AM = M - A
        am_ab = AM.x * AB.x + AM.y * AB.y
        ab_ab = AB.x * AB.x + AB.y * AB.y
        am_ad = AM.x * AD.x + AM.y * AD.y
        ad_ad = AD.x * AD.x + AD.y * AD.y

        return am_ab > 0 and am_ab < ab_ab and am_ad > 0 and am_ad < ad_ad
