#!/usr/bin/env python

# Customized scenarios: Intention aware lane change
# Author: Peng Xu
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Intention aware lane change scenario:

The scenario realizes a relative lane change behavior, under circumstance
that vehicles in the target lane don't have enough gap to merge in initially.
The ego vehicle is expected to move closer to remind the neighboring vehicle
to open a bigger gap until it is safe to complete the lane change.

The scenario ends either via a timeout, or if the ego vehicle stopped close
enough to the leading vehicle
"""

import random

import py_trees

import carla

from srunner.scenariomanager.atomic_scenario_behavior import *
from srunner.scenariomanager.atomic_scenario_criteria import *
from srunner.scenariomanager.timer import TimeOut
from srunner.scenarios.basic_scenario import *
from srunner.tools.scenario_helper import *

INTENTION_AWARE_LANE_CHANGE_SCENARIOS = [
    "IntentionAwareLaneChange"
]


class IntentionAwareLaneChange(BasicScenario):

    """
    This is a single ego vehicle scenario
    """
    category = "IntentionAwareLaneChange"

    timeout = 120            # Timeout of scenario in seconds

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True):
        """
        Setup all relevant parameters and create scenario
        """
        self._map = CarlaDataProvider.get_map()

        self._first_actor_location = 50
        self._second_actor_location = self._first_actor_location + 10

        self._other_one_actor_location = self._first_actor_location -20
        self._other_two_actor_location = self._first_actor_location - 30
        self._other_three_actor_location = self._first_actor_location - 40

        self._first_actor_speed = 10
        self._second_actor_speed = 5
        self._other_actor_speed = 10

        self._reference_waypoint = self._map.get_waypoint(config.trigger_points[0].location)
        self._other_actor_max_brake = 1.0

        self._first_actor_transform = None
        self._second_actor_transform = None

        self._other_one_actor_transform = None
        self._other_two_actor_transform = None
        self._other_three_actor_transform = None

        super(IntentionAwareLaneChange, self).__init__("IntentionAwareLaneChange",
                                                       ego_vehicles,
                                                       config,
                                                       world,
                                                       debug_mode,
                                                       criteria_enable=criteria_enable)
        if randomize:
            self._ego_other_distance_start = random.randint(4, 8)

    def _initialize_actors(self, config):
        """
        Custom initialization
        """

        first_actor_waypoint, _ = get_waypoint_in_distance(self._reference_waypoint, self._first_actor_location)
        second_actor_waypoint, _ = get_waypoint_in_distance(self._reference_waypoint, self._second_actor_location)

        other_one_actor_waypoint, _ = get_waypoint_in_distance(self._reference_waypoint, self._other_one_actor_location)
        other_two_actor_waypoint, _ = get_waypoint_in_distance(self._reference_waypoint, self._other_two_actor_location)
        other_three_actor_waypoint, _ = get_waypoint_in_distance(self._reference_waypoint, self._other_three_actor_location)

        first_actor_transform = carla.Transform(
            carla.Location(first_actor_waypoint.transform.location.x,
                           first_actor_waypoint.transform.location.y,
                           first_actor_waypoint.transform.location.z - 500),
            first_actor_waypoint.transform.rotation)
        self._first_actor_transform = carla.Transform(
            carla.Location(first_actor_waypoint.transform.location.x,
                           first_actor_waypoint.transform.location.y,
                           first_actor_waypoint.transform.location.z + 1),
            first_actor_waypoint.transform.rotation)
        yaw_1 = second_actor_waypoint.transform.rotation.yaw + 90
        second_actor_transform = carla.Transform(
            carla.Location(second_actor_waypoint.transform.location.x,
                           second_actor_waypoint.transform.location.y,
                           second_actor_waypoint.transform.location.z - 500),
            carla.Rotation(second_actor_waypoint.transform.rotation.pitch, yaw_1,
                           second_actor_waypoint.transform.rotation.roll))
        self._second_actor_transform = carla.Transform(
            carla.Location(second_actor_waypoint.transform.location.x,
                           second_actor_waypoint.transform.location.y,
                           second_actor_waypoint.transform.location.z + 1),
            carla.Rotation(second_actor_waypoint.transform.rotation.pitch, yaw_1,
                           second_actor_waypoint.transform.rotation.roll))
        # three other vehicles in left lane
        other_one_actor_transform = carla.Transform(
            carla.Location(other_one_actor_waypoint.transform.location.x,
                           other_one_actor_waypoint.transform.location.y + 3.8,
                           other_one_actor_waypoint.transform.location.z - 500),
            other_one_actor_waypoint.transform.rotation)
        self._other_one_actor_transform = carla.Transform(
            carla.Location(other_one_actor_waypoint.transform.location.x,
                           other_one_actor_waypoint.transform.location.y + 3.8,
                           other_one_actor_waypoint.transform.location.z + 1),
            other_one_actor_waypoint.transform.rotation)
        other_two_actor_transform = carla.Transform(
            carla.Location(other_two_actor_waypoint.transform.location.x,
                           other_two_actor_waypoint.transform.location.y + 3.8,
                           other_two_actor_waypoint.transform.location.z - 500),
            other_one_actor_waypoint.transform.rotation)
        self._other_two_actor_transform = carla.Transform(
            carla.Location(other_two_actor_waypoint.transform.location.x,
                           other_two_actor_waypoint.transform.location.y + 3.8,
                           other_two_actor_waypoint.transform.location.z + 1),
            other_two_actor_waypoint.transform.rotation)
        other_three_actor_transform = carla.Transform(
            carla.Location(other_three_actor_waypoint.transform.location.x,
                           other_three_actor_waypoint.transform.location.y + 3.8,
                           other_three_actor_waypoint.transform.location.z - 500),
            other_three_actor_waypoint.transform.rotation)
        self._other_three_actor_transform = carla.Transform(
            carla.Location(other_three_actor_waypoint.transform.location.x,
                           other_three_actor_waypoint.transform.location.y + 3.8,
                           other_three_actor_waypoint.transform.location.z + 1),
            other_three_actor_waypoint.transform.rotation)

        first_actor = CarlaActorPool.request_new_actor('vehicle.tesla.model3', first_actor_transform)
        second_actor = CarlaActorPool.request_new_actor('vehicle.diamondback.century',
                                                        second_actor_transform)

        other_one_actor = CarlaActorPool.request_new_actor('vehicle.toyota.prius',
                                                           other_one_actor_transform)
        other_two_actor = CarlaActorPool.request_new_actor('vehicle.bmw.grandtourer',
                                                           other_two_actor_transform)
        other_three_actor = CarlaActorPool.request_new_actor('vehicle.nissan.micra',
                                                             other_three_actor_transform)

        self.other_actors.append(first_actor)
        self.other_actors.append(second_actor)

        self.other_actors.append(other_one_actor)
        self.other_actors.append(other_two_actor)
        self.other_actors.append(other_three_actor)

    def _create_behavior(self):
        """
        The scenario defined after is a "follow leading vehicle" scenario. After
        invoking this scenario, it will wait for the user controlled vehicle to
        enter the start region, then make the other actor to drive towards obstacle.
        Once obstacle clears the road, make the other actor to drive towards the
        next intersection. Finally, the user-controlled vehicle has to be close
        enough to the other actor to end the scenario.
        If this does not happen within 60 seconds, a timeout stops the scenario
        """

        one_in_left_lane = py_trees.composites.Parallel(
                    "One In Left Lane",
                    policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        one_in_left_lane.add_child(WaypointFollower(self.other_actors[2], self._other_actor_speed))
        one_in_left_lane.add_child(InTriggerDistanceToNextIntersection(self.other_actors[2], 5))

        two_in_left_lane = py_trees.composites.Parallel(
                    "One In Left Lane",
                    policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        two_in_left_lane.add_child(UseAutoPilot(self.other_actors[3]))
        two_in_left_lane.add_child(InTriggerDistanceToNextIntersection(self.other_actors[2], 5))

        three_in_left_lane = py_trees.composites.Parallel(
                    "One In Left Lane",
                    policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        three_in_left_lane.add_child(UseAutoPilot(self.other_actors[4]))
        three_in_left_lane.add_child(InTriggerDistanceToNextIntersection(self.other_actors[2], 5))

        driving_in_left_lane = py_trees.composites.Parallel(
            "Driving in Left Lane",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        driving_in_left_lane.add_child(one_in_left_lane)
        driving_in_left_lane.add_child(two_in_left_lane)
        driving_in_left_lane.add_child(three_in_left_lane)

        # end condition
        endcondition = py_trees.composites.Parallel("Waiting for end position",
                                                    policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL)
        endcondition_part1 = InTriggerDistanceToVehicle(self.other_actors[2],
                                                        self.ego_vehicles[0],
                                                        distance=15,
                                                        name="FinalDistance")
        endcondition_part2 = StandStill(self.ego_vehicles[0], name="FinalSpeed")
        endcondition.add_child(endcondition_part1)
        endcondition.add_child(endcondition_part2)

        # Build behavior tree
        sequence = py_trees.composites.Sequence("Sequence Behavior")
        sequence.add_child(ActorTransformSetter(self.other_actors[0], self._first_actor_transform))
        sequence.add_child(ActorTransformSetter(self.other_actors[1], self._second_actor_transform))
        sequence.add_child(ActorTransformSetter(self.other_actors[2], self._other_one_actor_transform))
        sequence.add_child(ActorTransformSetter(self.other_actors[3], self._other_two_actor_transform))
        sequence.add_child(ActorTransformSetter(self.other_actors[4], self._other_three_actor_transform))
        sequence.add_child(driving_in_left_lane)
        sequence.add_child(StopVehicle(self.other_actors[2], self._other_actor_max_brake))
        sequence.add_child(endcondition)

        return sequence

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        criteria = []

        collision_criterion = CollisionTest(self.ego_vehicles[0])

        criteria.append(collision_criterion)

        return criteria

    def __del__(self):
        """
        Remove all actors upon deletion
        """
        self.remove_all_actors()
