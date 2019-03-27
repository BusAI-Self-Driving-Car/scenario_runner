import unittest
import os

import srunner.challenge.utils.route_configuration_parser as parser
from srunner.challenge.challenge_evaluator_routes import ChallengeEvaluator

from srunner.scenariomanager.carla_data_provider import CarlaActorPool

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.challenge.utils.route_manipulation import interpolate_trajectory
import carla




class Arguments():

    def __init__(self):
        self.agent = None
        self.use_docker = False
        self.host = '127.0.0.1'
        self.port = 2000
        self.split = 'dev_track_1'

class TestScenarioBuilder(unittest.TestCase):

    def __init__(self, name='runTest'):
        unittest.TestCase.__init__(self, name)
        self.root_route_file_position = 'srunner/challenge/'


    def test_build_scenarios(self):

        args = Arguments()
        client = carla.Client(args.host, int(args.port))
        client.set_timeout(25.0)
        challenge = ChallengeEvaluator(args)

        filename = os.path.join(self.root_route_file_position, 'all_towns_traffic_scenarios.json')
        world_annotations = parser.parse_annotations_file(filename)
        # retrieve routes
        # Which type of file is expected ????

        filename = os.path.join(self.root_route_file_position, 'routes_training.xml')
        list_route_descriptions = parser.parse_routes_file(filename)

        # For each of the routes to be evaluated.
        for route_description in list_route_descriptions:

            if route_description['town_name'] == 'Town03' or route_description['town_name'] == 'Town04':
                continue
            challenge.world = client.load_world(route_description['town_name'])

            # Set the actor pool so the scenarios can prepare themselves when needed
            CarlaActorPool.set_world(challenge.world)

            CarlaDataProvider.set_world(challenge.world)
            print (" WE HAVE THIS   ACTORS ", challenge.world.get_actors())
            for actor in challenge.world.get_actors():
                if 'static.prop' in actor.type_id:
                    print (actor.get_transform().location)
            # find and filter potential scenarios
            # Returns the iterpolation in a different format

            challenge.world.wait_for_tick()
            gps_route, route_description['trajectory'] = interpolate_trajectory(challenge.world,
                                                                                route_description['trajectory'])


            potential_scenarios_definitions, existent_triggers = parser.scan_route_for_scenarios(route_description,
                                                                                                 world_annotations)
            # Sample the scenarios
            sampled_scenarios = challenge.scenario_sampling(potential_scenarios_definitions)

            # prepare route's trajectory
            challenge.prepare_ego_car(route_description['trajectory'][0][0].transform)

            # build the master scenario based on the route and the target.
            master_scenario = challenge.build_master_scenario(route_description['trajectory'], route_description['town_name'])
            list_scenarios = [master_scenario]
            print (" TOWN  ", route_description['town_name'])
            print (" Built the master scenario ")
            # build the instance based on the parsed definitions.
            print (sampled_scenarios)
            list_scenarios += challenge.build_scenario_instances(sampled_scenarios, route_description['town_name'])
            print (" Scenarios present ", list_scenarios)

            challenge.cleanup(ego=True)
