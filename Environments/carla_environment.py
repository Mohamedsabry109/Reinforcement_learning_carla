
import random
import sys
from os import path, environ
from typing import Union, Dict
import time
from  collections import namedtuple

try:
    if 'CARLA_ROOT' in environ:
        sys.path.append(path.join(environ.get('CARLA_ROOT'), 'PythonClient'))
    else:
        screen.error("CARLA_ROOT was not defined. Please set it to point to the CARLA root directory and try again.", crash=False)

    from carla.client import CarlaClient
    from carla.settings import CarlaSettings
    from carla.tcp import TCPConnectionError
    from carla.sensor import Camera
    from carla.image_converter import *
    from carla.client import VehicleControl
    from carla.planner.planner import Planner
    from carla.driving_benchmark.experiment_suites.experiment_suite import ExperimentSuite
except ImportError:
    print("Error importing carla")

import os
import signal
import logging
import subprocess
import numpy as np
from enum import Enum
from typing import List, Union
import pygame

# enum of the available levels and their path
class CarlaLevel(Enum):
    TOWN1 = {"map_name": "Town01", "map_path": "/Game/Maps/Town01"}
    TOWN2 = {"map_name": "Town02", "map_path": "/Game/Maps/Town02"}


key_map = {
    'BRAKE': (274,),  # down arrow
    'GAS': (273,),  # up arrow
    'TURN_LEFT': (276,),  # left arrow
    'TURN_RIGHT': (275,),  # right arrow
    'GAS_AND_TURN_LEFT': (273, 276),
    'GAS_AND_TURN_RIGHT': (273, 275),
    'BRAKE_AND_TURN_LEFT': (274, 276),
    'BRAKE_AND_TURN_RIGHT': (274, 275),
}



class CameraTypes(Enum):
    FRONT = "forward_camera"
    LEFT = "left_camera"
    RIGHT = "right_camera"
    SEGMENTATION = "segmentation"
    DEPTH = "depth"
    LIDAR = "lidar"



class State(Enum):
    state = None


# class state_space(Enum):
#     shape = None


class ActionSpace(object):
    def __init__(self, shape = None, low= None , high = None, descriptions = None):    
        self.shape = shape
        self.low = low
        self.high = high
        self.descriptions = descriptions



class EnvironmentInterface(object):
    def __init__(self):
        pass

    def get_random_action(self):
        """
            This function returns the randomized action 
        """
        raise NotImplementedError("")

    def step(self):

        raise NotImplementedError("")

    def reset(self):

        raise NotImplementedError("")

map_path_mapper = {"Town01" : "/Game/Maps/Town01" , "Town02":"/Game/Maps/Town02"}

class CarlaEnvironment(EnvironmentInterface):


    def __init__ (self, experiment_path = None , frame_skip = 1, server_height = 512,
        server_width = 720, camera_height = 88, camera_width = 200, experiment_suite = None,
        quality = "low", cameras = [CameraTypes.FRONT] , weather_id = [1],  episode_max_time = 100000,
        max_speed = 35.0, port = 2000, map_name = "Town01", verbose=True,
        seed = None, is_rendered = True, num_speedup_steps = 30 ,separate_actions_for_throttle_and_brake = False, rendred_image_type = 'forward_camera'):

        self.frame_skip = frame_skip  # the frame skip affects the fps of the server directly. fps = 30 / frameskip
        self.server_height = server_height
        self.server_width = server_width
        self.camera_height = camera_height
        self.camera_width = camera_width
        self.experiment_suite = experiment_suite  # an optional CARLA experiment suite to use
        self.quality = quality
        self.cameras = cameras
        self.weather_id = weather_id
        self.episode_max_time =  episode_max_time # miliseconds for each episode
        self.max_speed = max_speed  # km/h
        self.port = port
        self.host = 'localhost'
        self.map_name = map_name
        self.map_path = map_path_mapper[self.map_name]
        self.experiment_path = experiment_path
        self.current_episode_steps_counter = 1
        # client configuration
        self.verbose = verbose
        self.episode_idx = 0
        self.num_speedup_steps = num_speedup_steps
        self.max_speed = max_speed
        self.is_rendered = is_rendered
        # setup server settings
        self.experiment_suite = experiment_suite
        self.separate_actions_for_throttle_and_brake = separate_actions_for_throttle_and_brake
        self.rendred_image_type = rendred_image_type
        self.left_poses = []
        self.right_poses = []
        self.follow_poses = []
        self.Straight_poses = []


        self.settings = CarlaSettings()
        self.settings.set(
                            SynchronousMode=True,
                            SendNonPlayerAgentsInfo=False,
                            NumberOfVehicles=15,
                            NumberOfPedestrians=30,
                            WeatherId=self.weather_id,
                            QualityLevel=self.quality,
                            SeedVehicles=seed,
                            SeedPedestrians=seed)
        if seed is None:
            self.settings.randomize_seeds()

        self.settings = self.add_cameras(self.settings, self.cameras, self.camera_width, self.camera_height)

        # open the server
        self.server = self.open_server()
        logging.disable(40)
        print("Successfully opened the server")

        # open the client
        self.game = CarlaClient(self.host, self.port, timeout=99999999)
        print("Successfully opened the client")

        self.game.connect()
        print("Successfull Connection")

        if self.experiment_suite:
            self.current_experiment_idx = 0
            self.current_experiment = self.experiment_suite.get_experiments()[self.current_experiment_idx]
            self.scene = self.game.load_settings(self.current_experiment.conditions)
        else:
            self.scene = self.game.load_settings(self.settings)

        # get available start positions
        self.positions = self.scene.player_start_spots
        self.num_positions = len(self.positions)
        self.current_start_position_idx = 0
        self.current_pose = 0


        self.action_space = ActionSpace(shape=2, low= np.array([-1, -1]), high=np.array([1, 1]), descriptions = ["steer", "gas_and_brake"])

        # state space
        #define all measurments
        self.state_space = {
            "measurements": {"forward_speed" : np.array([1]), "x" : np.array([1]), "y" : np.array([1]), "z" : np.array([1])}
        }
        #define all cameras
        for camera in self.scene.sensors:
            self.state_space[camera.name] = {"data": np.array([self.camera_height, self.camera_width, 3])}
            print("Define " , camera.name , " Camera")

        # measurements
        self.autopilot = None
        self.planner = Planner(self.map_name)


        #rendering 
        if self.is_rendered:
            pygame.init()
            pygame.font.init()
            self.display = pygame.display.set_mode(
            (self.camera_width, self.camera_height))


        # env initialization
        self.reset(True)




    def reset(self, force_environment_reset=False) :
        """
        Reset the environment and all the variable of the wrapper

        :param force_environment_reset: forces environment reset even when the game did not end
        :return: A dictionary containing the observation, reward, done flag, action and measurements
        """
        self.reset_ep = True
        self.restart_environment_episode(force_environment_reset)
        self.last_episode_time = time.time()

        if self.current_episode_steps_counter > 0 :
            self.episode_idx += 1

        self.done = False
        self.total_reward_in_current_episode = self.reward = 0.0
        self.last_action = 0
        self.current_episode_steps_counter = 0
        self.last_episode_images = []
        self.step([0,1,0])


        self.last_env_response = {"reward":self.reward,
                "next_state":self.state,
                "goal":self.current_goal,
                "game_over":self.done}

        return self.last_env_response


    def add_cameras(self, settings, cameras, camera_width, camera_height):
        # add a front facing camera
        print("Available Cameras are ",cameras)
        if CameraTypes.FRONT in cameras:
            camera = Camera(CameraTypes.FRONT.value)
            camera.set(FOV=100)
            camera.set_image_size(camera_width, camera_height)
            camera.set_position(2.0, 0, 1.4)
            camera.set_rotation(-15.0, 0, 0)
            settings.add_sensor(camera)

        # add a left facing camera
        if CameraTypes.LEFT in cameras:
            camera = Camera(CameraTypes.LEFT.value)
            camera.set(FOV=100)
            camera.set_image_size(camera_width, camera_height)
            camera.set_position(2.0, 0, 1.4)
            camera.set_rotation(-15.0, -30, 0)
            settings.add_sensor(camera)

        # add a right facing camera
        if CameraTypes.RIGHT in cameras:
            camera = Camera(CameraTypes.RIGHT.value)
            camera.set(FOV=100)
            camera.set_image_size(camera_width, camera_height)
            camera.set_position(2.0, 0, 1.4)
            camera.set_rotation(-15.0, 30, 0)
            settings.add_sensor(camera)

        # add a front facing depth camera
        if CameraTypes.DEPTH in cameras:
            camera = Camera(CameraTypes.DEPTH.value)
            camera.set_image_size(camera_width, camera_height)
            camera.set_position(0.2, 0, 1.3)
            camera.set_rotation(8, 30, 0)
            camera.PostProcessing = 'Depth'
            settings.add_sensor(camera)

        # add a front facing semantic segmentation camera
        if CameraTypes.SEGMENTATION in cameras:
            camera = Camera(CameraTypes.SEGMENTATION.value)
            camera.set_image_size(camera_width, camera_height)
            camera.set_position(0.2, 0, 1.3)
            camera.set_rotation(8, 30, 0)
            camera.PostProcessing = 'SemanticSegmentation'
            settings.add_sensor(camera)
            print("Successfully adding a SemanticSegmentation camera")

        return settings

    def get_directions(self, current_point, end_point):
        """
        Class that should return the directions to reach a certain goal
        """

        directions = self.planner.get_next_command(
            (current_point.location.x,
             current_point.location.y, 0.22),
            (current_point.orientation.x,
             current_point.orientation.y,
             current_point.orientation.z),
            (end_point.location.x, end_point.location.y, 0.22),
            (end_point.orientation.x, end_point.orientation.y, end_point.orientation.z))
        return directions

    def open_server(self):
        log_path = path.join(self.experiment_path if self.experiment_path is not None else '.', 'logs',
                             "CARLA_LOG_{}.txt".format(self.port))
        if not os.path.exists(os.path.dirname(log_path)):
            os.makedirs(os.path.dirname(log_path))
        with open(log_path, "wb") as out:
            cmd = [path.join(environ.get('CARLA_ROOT'), 'CarlaUE4.sh'), self.map_path,
                   "-benchmark", "-carla-server", "-fps={}".format(30 / self.frame_skip),
                   "-world-port={}".format(self.port),
                   "-windowed -ResX={} -ResY={}".format(self.server_width, self.server_height),
                   "-carla-no-hud"]
            print("CMD is : ",cmd)

            # if self.config:
            #     cmd.append("-carla-settings={}".format(self.config))
            p = subprocess.Popen(cmd, stdout=out, stderr=out)

        return p

    def close_server(self):
        os.killpg(os.getpgid(self.server.pid), signal.SIGKILL)



    def step(self,action):
        # get measurements and observations
        measurements = []
        while type(measurements) == list:
            measurements, sensor_data = self.game.read_data()
        self.state = {}

        for camera in self.scene.sensors:
            if camera.name == 'segmentation':
                #labels_to_road_noroad taker Sensor.Image not numpy array
                self.state[camera.name] = labels_to_road_noroad(sensor_data[camera.name])
            else:
                self.state[camera.name] = sensor_data[camera.name].data
            #self.state[camera.name] = sensor_data[camera.name].data

        self.location = [measurements.player_measurements.transform.location.x,
                         measurements.player_measurements.transform.location.y,
                         measurements.player_measurements.transform.location.z]

        self.distance_from_goal = np.linalg.norm(np.array(self.location[:2]) -
                                                 [self.current_goal.location.x, self.current_goal.location.y])

        is_collision = measurements.player_measurements.collision_vehicles != 0 \
                       or measurements.player_measurements.collision_pedestrians != 0 \
                       or measurements.player_measurements.collision_other != 0

        speed_reward = measurements.player_measurements.forward_speed - 1
        if speed_reward > 30.:
            speed_reward = 30.
        self.reward = speed_reward \
                      - (measurements.player_measurements.intersection_otherlane * 5) \
                      - (measurements.player_measurements.intersection_offroad * 5) \
                      - is_collision * 100 \
                      - np.abs(self.control.steer) * 10

        # update measurements
        #self.measurements = [measurements.player_measurements.forward_speed] + self.location 
        #TODO, Add control signals to measurements
        control_signals = [np.clip(action[0], -1, 1) ,np.clip(action[1], 0, 1),np.clip(action[2], 0, 1)] #steer, throttle, brake
        self.measurements = [measurements.player_measurements.forward_speed] + self.location + control_signals 

        self.autopilot = measurements.player_measurements.autopilot_control

        # The directions to reach the goal (0 Follow lane, 1 Left, 2 Right, 3 Straight)
        directions = int(self.get_directions(measurements.player_measurements.transform, self.current_goal) - 2)

        # if directions == 0:
        #     if self.reset_ep:
        #         self.follow_poses.append((self.current_start,self.current_goal))
        #         self.reset_ep = False
        #     elif directions == 1:
        #         self.left_poses.append((self.current_start,self.current_goal))
        #     elif directions == 2:
        #         self.right_poses.append((self.current_start,self.current_goal))
            
        # elif directions == 1:
        #     if self.reset_ep:
        #         self.left_poses.append((self.current_start,self.current_goal))
        #         self.reset_ep = False
        # elif directions == 2:
        #     if self.reset_ep:
        #         self.right_poses.append((self.current_start,self.current_goal))
        #         self.reset_ep = False
        # elif directions == 3:
        #     if self.reset_ep:
        #         self.Straight_poses.append((self.current_start,self.current_goal))
        #         self.reset_ep = False
        # else:
        #     self.reset_ep = False


        self.state['high_level_command'] = directions

        if (measurements.game_timestamp >= self.episode_max_time) or is_collision:
            self.done = True


        self.state['measurements'] = np.array(self.measurements)

        #prepare rendered image and update display
        if self.is_rendered:
            #check if user wants to close rendering, else continue rendering
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    self.is_rendered = False
                
            self.surface = pygame.surfarray.make_surface(self.state[self.rendred_image_type].swapaxes(0, 1))
            self.display.blit(self.surface, (0, 0))
            pygame.display.flip()


        self.take_action(action)

    def take_action(self, action):
        self.control = VehicleControl()

        # transform the 2 value action (steer, throttle - brake) into a 3 value action (steer, throttle, brake)
        self.control.steer = np.clip(action[0], -1, 1)
        self.control.throttle = np.clip(action[1], 0, 1)
        self.control.brake = np.abs(np.clip(action[2], 0, 1))

        # prevent braking
        # if not self.allow_braking or self.control.brake < 0.1 or self.control.throttle > self.control.brake:
        #     self.control.brake = 0

        # prevent over speeding
        if hasattr(self, 'measurements') and self.measurements[0] * 3.6 > self.max_speed and self.control.brake == 0:
            self.control.throttle = 0.0

        self.control.hand_brake = False
        self.control.reverse = False

        self.game.send_control(self.control)

    def load_experiment(self, experiment_idx):
        print("Loading the experiment")
        self.current_experiment = self.experiment_suite.get_experiments()[experiment_idx]
        self.scene = self.game.load_settings(self.current_experiment.conditions)
        self.positions = self.scene.player_start_spots
        self.num_positions = len(self.positions)
        self.current_start_position_idx = 0
        self.current_pose = 0

    def restart_environment_episode(self, force_environment_reset=False):
        # select start and end positions
        print("restarting new Episode")
        if self.experiment_suite:
            # if an expeirent suite is available, follow its given poses
            if self.current_pose >= len(self.current_experiment.poses):
                # load a new experiment
                self.current_experiment_idx = (self.current_experiment_idx + 1) % len(self.experiment_suite.get_experiments())
                self.load_experiment(self.current_experiment_idx)

            self.current_start_position_idx = self.current_experiment.poses[self.current_pose][0]
            self.current_start = self.positions[self.current_experiment.poses[self.current_pose][0]]
            self.current_goal = self.positions[self.current_experiment.poses[self.current_pose][1]]
            self.current_pose += 1
        else:
            # go over all the possible positions in a cyclic manner
            self.current_start_position_idx = (self.current_start_position_idx + 1) % self.num_positions
            self.current_start = self.positions[self.current_start_position_idx]
            # choose a random goal destination
            self.current_goal = random.choice(self.positions)

        try:
            self.game.start_episode(self.current_start_position_idx)
        except:
            self.game.connect()
            self.game.start_episode(self.current_start_position_idx)

        # start the game with some initial speed
        for i in range(self.num_speedup_steps):
            self.control = VehicleControl(throttle=1.0, brake=0, steer=0, hand_brake=False, reverse=False)
            self.game.send_control(VehicleControl())

    def get_rendered_image(self) -> np.ndarray:
        """
        Return a numpy array containing the image that will be rendered to the screen.
        This can be different from the observation. For example, mujoco's observation is a measurements vector.
        :return: numpy array containing the image that will be rendered to the screen
        """
        image = [self.state[camera.name] for camera in self.scene.sensors]
        image = np.vstack(image)
        return image

    # def get_target_success_rate(self) -> float:
    #     return self.target_success_rate


# env = CarlaEnvironment()
# print("number of available poses are : ",env.num_positions)
# for i in range(1):
#     print("iteration ",i)
#     while env.done == False:
#         env.step([0,1,0])

#     env.reset(True)
# print(env.action_space)
# print(env.state_space)
# env.close_server()

# print("number of available poses are : ",env.num_positions)
# print("follow poses are : ",env.follow_poses)
# print("left poses are : ",env.left_poses)
# print("right poses are : ", env.right_poses)
# print("straight poses are : ", env.Straight_poses)

