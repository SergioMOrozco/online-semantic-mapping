import math
import numpy as np
from bosdyn.client import math_helpers, frame_helpers
from bosdyn.client.robot_command import (RobotCommandBuilder, RobotCommandClient,
                                         block_until_arm_arrives, blocking_stand)


def get_pose_from_circle_vertical(vertical_angle_offset):

    # x, y, z
    # 4 in front of robot
    radius = 2

    angle_reference = 180

    angle = vertical_angle_offset + angle_reference

    x= (radius * math.cos(angle * math.pi / 180.))
    z= (radius * math.sin(angle * math.pi / 180.))

    if angle - angle_reference > 0:
        pitch = vertical_angle_offset 
    else:
        pitch = vertical_angle_offset 

    ## add offset
    x += radius

    print("X: ",x,"Z: ",z,"Pitch: ",pitch)

    rot = math_helpers.Quat().from_pitch(pitch)

    pose = math_helpers.SE3Pose(x=x,y=0,z=z,rot=rot)

    return pose

def get_pose_from_circle_horizontal(horizontal_angle_offset):

    # x, y, z
    # 4 in front of robot
    radius = 2

    angle_reference = 270
    angle = horizontal_angle_offset + angle_reference

    x= (radius * math.cos(angle * math.pi / 180.))
    y= (radius * math.sin(angle * math.pi / 180.))

    if angle - angle_reference > 0:
        yaw = -horizontal_angle_offset 
    else:
        yaw = -horizontal_angle_offset 
    roll = 0
    pitch = 0

    # go from cartesian coordinate system to gripper coordinate system
    tmp = x
    x = y 
    y = -tmp

    ## add offset
    x += radius

    #print("X: ",x,"Y: ",y,"Yaw: ",yaw)

    rot = math_helpers.Quat().from_yaw(yaw)

    pose = math_helpers.SE3Pose(x=x,y=y,z=0,rot=rot)


    return x,y,yaw

def stuff(robot,robot_state_client,command_client, start, end):

    x_start = 0
    y_start = 0
    z_start = 0

    start_horizontal_angle_offset = 0
    start_vertical_angle_offset = 0
    end_horizontal_angle_offset = 0
    end_vertical_angle_offset = 0

    seconds = 2


    if start == "left":
        start_horizontal_angle_offset = -30
        start_vertical_angle_offset = 0

    if start == "right":
        start_horizontal_angle_offset = 30
        start_vertical_angle_offset = 0

    if start == "center":
        start_horizontal_angle_offset = 0
        start_vertical_angle_offset = 0

    if start == "center_up":
        start_horizontal_angle_offset = 0
        start_vertical_angle_offset = -30

    if end == "left":
        end_horizontal_angle_offset = -30
        end_vertical_angle_offset = 0

    if end == "right":
        end_horizontal_angle_offset = 30
        end_vertical_angle_offset = 0

    if end == "center":
        end_horizontal_angle_offset = 0
        end_vertical_angle_offset = 0

    if end == "center_up":
        end_horizontal_angle_offset = 0
        end_vertical_angle_offset = -30

    if not start_horizontal_angle_offset - end_horizontal_angle_offset == 0:
        sample_angles = np.linspace(start_horizontal_angle_offset,end_horizontal_angle_offset, num=5)

        for sample in sample_angles:
            gripper_pose = get_pose_from_circle_horizontal(sample)

            arm_commad = RobotCommandBuilder.arm_pose_command(gripper_pose.x,gripper_pose.y,gripper_pose.z,gripper_pose.rot.w,gripper_pose.rot.x,gripper_pose.rot.y,gripper_pose.rot.z,frame_helpers.VISION_FRAME,NAME,seconds)

            cmd_id = command_client.robot_command(arm_command)

            block_until_arm_arrives(command_client, cmd_id)

    elif not start_vertical_angle_offset - end_vertical_angle_offset == 0:
        sample_angles = np.linspace(start_vertical_angle_offset,end_vertical_angle_offset, num=5)

        for sample in sample_angles:
            gripper_pose = get_pose_from_circle_vertical(sample)

            arm_command = RobotCommandBuilder.arm_pose_command(gripper_pose.x,gripper_pose.y,gripper_pose.z,gripper_pose.rot.w,gripper_pose.rot.x,gripper_pose.rot.y,gripper_pose.rot.z,frame_helpers.VISION_FRAME_NAME,seconds)

            cmd_id = command_client.robot_command(arm_command)

            block_until_arm_arrives(command_client, cmd_id)

stuff(None,None,None,"center","center_up")
