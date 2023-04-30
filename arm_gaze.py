# Copyright (c) 2022 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).

"""Tutorial to show how to use Spot's arm.
"""
import argparse
import sys
from vision_model import VisionModel
from time import sleep

from google.protobuf import duration_pb2

import bosdyn.client
import bosdyn.client.lease
import bosdyn.client.util
from bosdyn.api import (arm_command_pb2, geometry_pb2, robot_command_pb2, synchronized_command_pb2,
                        trajectory_pb2)
from bosdyn.client import math_helpers
from bosdyn.client.frame_helpers import GRAV_ALIGNED_BODY_FRAME_NAME, ODOM_FRAME_NAME, get_a_tform_b
from bosdyn.client.robot_command import (RobotCommandBuilder, RobotCommandClient,
                                         block_until_arm_arrives, blocking_stand)
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.api import gripper_camera_param_pb2, header_pb2
from google.protobuf import wrappers_pb2
from bosdyn.client.gripper_camera_param import GripperCameraParamClient


HOSTNAME = "tusker.rlab.cs.brown.edu"


def do_stuff(model,robot,robot_state_client,command_client,traj_data,move_data):

        ## Convert the location from the moving base frame to the world frame.
        robot_state = robot_state_client.get_robot_state()
        odom_T_flat_body = get_a_tform_b(robot_state.kinematic_state.transforms_snapshot,
                                         ODOM_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME)

        # Look to the left and the right with the hand.
        # Robot's frame is X+ forward, Z+ up, so left and right is +/- in Y.
        start_x = traj_data[0]  # look 4 meters ahead
        end_x = traj_data[1]  # look 4 meters ahead
        start_y = traj_data[2]
        end_y = traj_data[3]
        start_z = traj_data[4]  # Look ahead, not up or down
        end_z = traj_data[5]  # Look ahead, not up or down


        traj_time = 10  # take 5.5 seconds to look from left to right.

        start_pos_in_odom_tuple = odom_T_flat_body.transform_point(x=start_x, y=start_y, z=start_z)
        start_pos_in_odom = geometry_pb2.Vec3(x=start_pos_in_odom_tuple[0],
                                              y=start_pos_in_odom_tuple[1],
                                              z=start_pos_in_odom_tuple[2])

        end_pos_in_odom_tuple = odom_T_flat_body.transform_point(x=start_x, y=end_y, z=end_z)
        end_pos_in_odom = geometry_pb2.Vec3(x=end_pos_in_odom_tuple[0], y=end_pos_in_odom_tuple[1],
                                            z=end_pos_in_odom_tuple[2])

        ## Create the trajectory points
        point1 = trajectory_pb2.Vec3TrajectoryPoint(point=start_pos_in_odom)

        duration_seconds = int(traj_time)
        duration_nanos = int((traj_time - duration_seconds) * 1e9)

        point2 = trajectory_pb2.Vec3TrajectoryPoint(
            point=end_pos_in_odom,
            time_since_reference=duration_pb2.Duration(seconds=duration_seconds,
                                                       nanos=duration_nanos))

        # Build the trajectory proto
        traj_proto = trajectory_pb2.Vec3Trajectory(points=[point1, point2])

        ## STUFF TO COMMENT
        ## Build the proto
        #gaze_cmd = arm_command_pb2.GazeCommand.Request(target_trajectory_in_frame1=traj_proto,
        #                                               frame1_name=ODOM_FRAME_NAME,
        #                                               frame2_name=ODOM_FRAME_NAME)
        #arm_command = arm_command_pb2.ArmCommand.Request(arm_gaze_command=gaze_cmd)
        #synchronized_command = synchronized_command_pb2.SynchronizedCommand.Request(
        #    arm_command=arm_command)
        #command = robot_command_pb2.RobotCommand(synchronized_command=synchronized_command)

        ## Make the open gripper RobotCommand
        #gripper_command = RobotCommandBuilder.claw_gripper_open_command()

        ## Combine the arm and gripper commands into one RobotCommand
        #synchro_command = RobotCommandBuilder.build_synchro_command(gripper_command, command)

        ## Send the request
        #gaze_command_id = command_client.robot_command(command)
        #robot.logger.info('Sending gaze trajectory.')

        ## Wait until the robot completes the gaze before issuing the next command.
        #block_until_arm_arrives(command_client, gaze_command_id, timeout_sec=traj_time + 3.0)

        #print("Done")

        # ------------- #

        # Now make a gaze trajectory that moves the hand around while maintaining the gaze.
        # We'll use the same trajectory as before, but add a trajectory for the hand to move to.

        traj_time = 10  # take 5.5 seconds to look from left to right.

        start_hand_x = move_data[0]
        end_hand_x = move_data[1]
        start_hand_y = move_data[2]
        end_hand_y = move_data[3]
        start_hand_z = move_data[4]
        end_hand_z = move_data[5]

        hand_vec3_start = geometry_pb2.Vec3(x=start_hand_x, y=start_hand_y, z=start_hand_z)
        hand_vec3_end = geometry_pb2.Vec3(x=end_hand_x, y=end_hand_y, z=end_hand_z)

        # We specify an orientation for the hand, which the robot will use its remaining degree
        # of freedom to achieve.  Most of it will be ignored in favor of the gaze direction.
        qw = 1
        qx = 0
        qy = 0
        qz = 0
        quat = geometry_pb2.Quaternion(w=qw, x=qx, y=qy, z=qz)

        # Build a trajectory
        hand_pose1_in_flat_body = geometry_pb2.SE3Pose(position=hand_vec3_start, rotation=quat)
        hand_pose2_in_flat_body = geometry_pb2.SE3Pose(position=hand_vec3_end, rotation=quat)

        hand_pose1_in_odom = odom_T_flat_body * math_helpers.SE3Pose.from_proto(
            hand_pose1_in_flat_body)
        hand_pose2_in_odom = odom_T_flat_body * math_helpers.SE3Pose.from_proto(
            hand_pose2_in_flat_body)

        traj_point1 = trajectory_pb2.SE3TrajectoryPoint(pose=hand_pose1_in_odom.to_proto())

        # We'll make this trajectory the same length as the one above.
        traj_point2 = trajectory_pb2.SE3TrajectoryPoint(
            pose=hand_pose2_in_odom.to_proto(),
            time_since_reference=duration_pb2.Duration(seconds=duration_seconds,
                                                       nanos=duration_nanos))

        hand_traj = trajectory_pb2.SE3Trajectory(points=[traj_point1, traj_point2])

        # Build the proto
        gaze_cmd = arm_command_pb2.GazeCommand.Request(target_trajectory_in_frame1=traj_proto,
                                                       frame1_name=ODOM_FRAME_NAME,
                                                       tool_trajectory_in_frame2=hand_traj,
                                                       frame2_name=ODOM_FRAME_NAME)

        arm_command = arm_command_pb2.ArmCommand.Request(arm_gaze_command=gaze_cmd)
        synchronized_command = synchronized_command_pb2.SynchronizedCommand.Request(
            arm_command=arm_command)
        command = robot_command_pb2.RobotCommand(synchronized_command=synchronized_command)

        # Make the open gripper RobotCommand
        gripper_command = RobotCommandBuilder.claw_gripper_open_command()

        # Combine the arm and gripper commands into one RobotCommand
        synchro_command = RobotCommandBuilder.build_synchro_command(gripper_command, command)

        # Send the request
        gaze_command_id = command_client.robot_command(synchro_command)
        robot.logger.info('Sending gaze trajectory with hand movement.')

        sleep(0.5)

        #TODO: build thread here
        thread = model.create_thread()
        model.start_taking_images(thread)

        # Wait until the robot completes the gaze before powering off.
        block_until_arm_arrives(command_client, gaze_command_id, timeout_sec=traj_time + 3.0)

        model.stop_taking_images(thread)
        sleep(1.0)
        
        
def gaze_control(config):
    """Commanding a gaze with Spot's arm."""



    # See hello_spot.py for an explanation of these lines.
    #bosdyn.client.util.setup_logging(config.verbose)

    sdk = bosdyn.client.create_standard_sdk('GazeDemoClient')
    robot = sdk.create_robot(config.hostname)
    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()

    assert robot.has_arm(), "Robot requires an arm to run this example."

    # Verify the robot is not estopped and that an external application has registered and holds
    # an estop endpoint.
    assert not robot.is_estopped(), "Robot is estopped. Please use an external E-Stop client, " \
                                    "such as the estop SDK example, to configure E-Stop."




    manual_focus = wrappers_pb2.FloatValue(value=0.5)
    auto_focus = wrappers_pb2.BoolValue(value=False)

    gripper_camera_param_client = robot.ensure_client(GripperCameraParamClient.default_service_name)

    params = gripper_camera_param_pb2.GripperCameraParams(focus_absolute=manual_focus, focus_auto=auto_focus)

    request = gripper_camera_param_pb2.GripperCameraParamRequest(params=params)

    # Send the request
    response = gripper_camera_param_client.set_camera_params(request)
    print('Sent request.')

    if response.header.error and response.header.error.code != header_pb2.CommonError.CODE_OK:
        print('Got an error:')
        print(response.header.error)

    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)

    lease_client = robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)
    with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
        # Now, we are ready to power on the robot. This call will block until the power
        # is on. Commands would fail if this did not happen. We can also check that the robot is
        # powered at any point.
        robot.logger.info("Powering on robot... This may take a several seconds.")
        robot.power_on(timeout_sec=20)
        assert robot.is_powered_on(), "Robot power on failed."
        robot.logger.info("Robot powered on.")

        # Tell the robot to stand up. The command service is used to issue commands to a robot.
        # The set of valid commands for a robot depends on hardware configuration. See
        # RobotCommandBuilder for more detailed examples on command building. The robot
        # command service requires timesync between the robot and the client.
        robot.logger.info("Commanding robot to stand...")
        command_client = robot.ensure_client(RobotCommandClient.default_service_name)
        blocking_stand(command_client, timeout_sec=10)
        robot.logger.info("Robot standing.")

        # Unstow the arm
        unstow = RobotCommandBuilder.arm_ready_command()

        # Issue the command via the RobotCommandClient
        unstow_command_id = command_client.robot_command(unstow)
        robot.logger.info("Unstow command issued.")

        block_until_arm_arrives(command_client, unstow_command_id, 3.0)

        #move-to-start
        #start_x,end_x,start_y,end_y,start_z,end_z
        #traj_data = [4.0,4.0,2.0,-2.0,0.0,0,0]
        #move_data = [0.75,0.75,-0.25,-0.25,0,0]

        ##start_x,end_x,start_y,end_y,start_z,end_z
        #do_stuff(robot,robot_state_client,command_client,traj_data,move_data)
        #sleep(1.0)
        #
        #exit()

        model = VisionModel(robot)



        # right-to-left
        #start_x,end_x,start_y,end_y,start_z,end_z
        traj_data = [4.0,4.0,2.0,-2.0,0.0,0,0]
        move_data = [0.75,0.75,-0.25,0.25,0,0]

        #start_x,end_x,start_y,end_y,start_z,end_z
        do_stuff(model,robot,robot_state_client,command_client,traj_data,move_data)

        # left-to-right
        #start_x,end_x,start_y,end_y,start_z,end_z
        traj_data = [4.0,4.0,-2.0,2.0,0.0,0,0]
        move_data = [0.75,0.75,0.25,-0.25,0,0]

        #start_x,end_x,start_y,end_y,start_z,end_z
        do_stuff(model,robot,robot_state_client,command_client,traj_data,move_data)

        # right-to-upcenter
        #start_x,end_x,start_y,end_y,start_z,end_z
        traj_data = [4.0,4.0,2.0,0.0,0,-2]
        move_data = [0.75,0.75,-0.25,0,0,0.25]

        #start_x,end_x,start_y,end_y,start_z,end_z
        do_stuff(model,robot,robot_state_client,command_client,traj_data,move_data)

        # upcenter-to-right
        #start_x,end_x,start_y,end_y,start_z,end_z
        traj_data = [4.0,4.0,0,2.0,-2,0]
        move_data = [0.75,0.75,0,-0.25,0.25,0]

        #start_x,end_x,start_y,end_y,start_z,end_z
        do_stuff(model,robot,robot_state_client,command_client,traj_data,move_data)

        # right-to-left
        #start_x,end_x,start_y,end_y,start_z,end_z
        traj_data = [4.0,4.0,2.0,-2.0,0.0,0,0]
        move_data = [0.75,0.75,-0.25,0.25,0,0]

        #start_x,end_x,start_y,end_y,start_z,end_z
        do_stuff(model,robot,robot_state_client,command_client,traj_data,move_data)

        # left-to-upcenter
        #start_x,end_x,start_y,end_y,start_z,end_z
        traj_data = [4.0,4.0,-2.0,0.0,0.0,-2]
        move_data = [0.75,0.75,0.25,0.0,0.0,0.25]

        #start_x,end_x,start_y,end_y,start_z,end_z
        do_stuff(model,robot,robot_state_client,command_client,traj_data,move_data)

        # upcenter-to-left
        #start_x,end_x,start_y,end_y,start_z,end_z
        traj_data = [4.0,4.0,0.0,-2,-2.0,0.0]
        move_data = [0.75,0.75,0.0,0.25,0.25,0.0]

        #start_x,end_x,start_y,end_y,start_z,end_z
        do_stuff(model,robot,robot_state_client,command_client,traj_data,move_data)

        model.data_capture_model.write_to_file()

        # Power the robot off. By specifying "cut_immediately=False", a safe power off command
        # is issued to the robot. This will attempt to sit the robot before powering off.
        robot.power_off(cut_immediately=False, timeout_sec=20)
        assert not robot.is_powered_on(), "Robot power off failed."
        robot.logger.info("Robot safely powered off.")



def main(argv):
    """Command line interface."""
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    options = parser.parse_args(argv)
    try:
        gaze_control(options)
        return True
    except Exception as exc:  # pylint: disable=broad-except
        logger = bosdyn.client.util.get_logger()
        logger.exception("Threw an exception")
        return False


if __name__ == '__main__':
    if not main(sys.argv[1:]):
        sys.exit(1)

