import argparse
import sys
import time
import numpy as np
import cv2
import math
import bosdyn.client
import bosdyn.client.util
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.robot_command import RobotCommandClient, RobotCommandBuilder, block_until_arm_arrives
from bosdyn.api import geometry_pb2
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.api import image_pb2
from bosdyn.client.network_compute_bridge_client import NetworkComputeBridgeClient
from bosdyn.api import network_compute_bridge_pb2
from google.protobuf import wrappers_pb2
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.api import manipulation_api_pb2
from bosdyn.api import basic_command_pb2
from bosdyn.client import frame_helpers
from bosdyn.client import math_helpers

def grasp_directions(object):
    if object == "drawer" or object == "coffee_cup":
        return geometry_pb2.Vec3(x=0, y=1, z=0), geometry_pb2.Vec3(x=0, y=0, z=1)
    else:
        raise Exception("Invalid object name")

def task_constructor(object):
    if object == "drawer":
        return construct_drawer_task(-VELOCITY, force_limit=FORCE_LIMIT), construct_drawer_task(VELOCITY, force_limit=FORCE_LIMIT)
    else:
        raise Exception("Invalid object name")
