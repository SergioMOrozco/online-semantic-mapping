from threading import Thread
import pickle
import numpy as np
import cv2
import math
import time
import bosdyn.client
import bosdyn.client.util
import uuid
from bosdyn.client.image import ImageClient
from bosdyn.api import image_pb2
from bosdyn.api import network_compute_bridge_pb2
from google.protobuf import wrappers_pb2
from bosdyn.client import frame_helpers
from bosdyn.client import math_helpers
from bosdyn.client.network_compute_bridge_client import NetworkComputeBridgeClient
from bosdyn.client.graph_nav import GraphNavClient
from data_capture.data_capture_model import DataCaptureModel
from bosdyn.client.frame_helpers import get_a_tform_b
from scipy.spatial.transform import Rotation
from bosdyn.client.math_helpers import Quat

MODEL_NAME = "handle-model"
HAND_MODEL = 'object-hand-model'
SERVER_NAME = "fetch-server"
#HOSTNAME = "138.16.161.12"
HOSTNAME = "tusker.rlab.cs.brown.edu"
IMAGES_PER_SECOND = 0.5


class VisionModel:
    image_sources = [
            'frontleft_fisheye_image', 'frontright_fisheye_image',
            'left_fisheye_image', 'right_fisheye_image', 'back_fisheye_image'
        ]
    def __init__(self, robot):
        #self.graph_nav_client = graph_nav_client
        #self.network_compute_client = network_compute_client
        self.robot = robot
        self.image_client = robot.ensure_client(ImageClient.default_service_name)
        self.i = 0
        self.stopped = True
        self.data_capture_model = DataCaptureModel("transforms.json")
    def create_thread(self):
        return Thread(target = self.take_images)

    def start_taking_images(self,thread):
        print("I just ran: START")
        #self.stopped = False
        #self.image_thread = Thread(target = self.take_images)
        #self.image_thread.start()

        self.stopped = False
        thread.start()

    def stop_taking_images(self,thread):
        self.stopped = True 
        thread.join()
        #self.stopped = True
        #self.image_thread.join()
        print("I just ran: STOP")
    def take_images(self):
        while True:
            if self.stopped:
                print("STOPPED")
                break
            start = time.time() # gives current time in seconds since Jan 1, 1970 (in Unix)
            self.take_image("hand_color_image")
            while True:
                current_time = time.time()
                if current_time - start >= 1.0/IMAGES_PER_SECOND:
                    break
    def quat_to_eulerZYX(self,q):
        """Convert a Quat object into Euler yaw, pitch, roll angles (radians)."""
        pitch = math.asin(-2 * (q.x * q.z - q.w * q.y))
        if pitch > 0.9999:
            yaw = 2 * math.atan2(q.z, q.w)
            pitch = math.pi / 2
            roll = 0
        elif pitch < -0.9999:
            yaw = 2 * math.atan2(q.z, q.w)
            pitch = -math.pi / 2
            roll = 0
        else:
            yaw = math.atan2(2 * (q.x * q.y + q.w * q.z), q.w * q.w + q.x * q.x - q.y * q.y - q.z * q.z)
            roll = math.atan2(2 * (q.y * q.z + q.w * q.x),
                              q.w * q.w - q.x * q.x - q.y * q.y + q.z * q.z)
        return yaw, pitch, roll

    def take_image(self, source):
        # We want to capture from one camera at a time.

        # Capture and save images to disk
        file_name = "images/" + str(self.i) +".png"
        self.i += 1

        image_responses = self.image_client.get_image_from_sources([source])

        dtype = np.uint8

        img = np.frombuffer(image_responses[0].shot.image.data, dtype=dtype)
        img = cv2.imdecode(img, -1)

        frame_tree_snapshot = self.robot.get_frame_tree_snapshot()
        vision_tform_hand = get_a_tform_b(frame_tree_snapshot, "vision", "hand")

        
        
        tmpx = vision_tform_hand.x
        tmpy = vision_tform_hand.y
        tmpz = vision_tform_hand.z

        vision_tform_hand.x = tmpy * -1.0 * 10.0
        vision_tform_hand.y = tmpz * 10.0
        vision_tform_hand.z = tmpx * -1.0

        rtmpx = vision_tform_hand.rot.x
        rtmpy = vision_tform_hand.rot.y
        rtmpz = vision_tform_hand.rot.z

        vision_tform_hand.rot.x = rtmpy * -1
        vision_tform_hand.rot.y = rtmpz
        vision_tform_hand.rot.z = rtmpx * -1

        #vision_tform_hand.rot.x = 0
        #vision_tform_hand.rot.y = 0
        #vision_tform_hand.rot.z = 0


        #vision_tform_hand.y *= -1 
        #vision_tform_hand.z *= -1

        #yaw, pitch, roll = self.quat_to_eulerZYX(vision_tform_hand.rot)
        #roll += math.pi / 2.0
        #pitch += math.pi / 2.0

        # Create a rotation object from Euler angles specifying axes of rotation
        #rot = Rotation.from_euler('zyx', [yaw, pitch, roll], degrees=False)

        # Convert to quaternions and print
        #rot_quat = rot.as_quat()
        #rot_quat = Quat(rot_quat[0],rot_quat[1],rot_quat[2],rot_quat[3])

        #vision_tform_hand.rot = rot_quat

        self.data_capture_model.add_frame(file_name,vision_tform_hand.to_matrix())


        # # Approximately rotate the image to level.
        # if image_responses[0].source.name[0:5] == "front":
        #     img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        #
        # elif image_responses[0].source.name[0:5] == "right":
        #     img = cv2.rotate(img, cv2.ROTATE_180)

        # Show the image
        #cv2.imshow("Object", img)


        #cv2.waitKey(15)

        #vision_tform_body = bosdyn.client.frame_helpers.get_vision_tform_body(self.robot.get_frame_tree_snapshot())
        #body_tform_vision = vision_tform_body.inverse()
        #localization_state = self.graph_nav_client.get_localization_state()
        #seed_tform_body = localization_state.localization.seed_tform_body

        # need to convert from geometry_pb2.SE3Pose to math_helpers.SE3Pose
        #seed_tform_body =  math_helpers.SE3Pose(seed_tform_body.position.x,seed_tform_body.position.y,seed_tform_body.position.z, seed_tform_body.rotation)

        #if seed_tform_body == None:
        #    print("Forgot to upload map")
        #    return None, None

        cv2.imwrite(file_name, img)
        #seed_tform_vision = seed_tform_body * body_tform_vision
        #print("seed tfrom vision: ", seed_tform_vision)
        print("filename", file_name)

        #return seed_tform_vision, file_name
        return file_name

if __name__ == "__main__":
    sdk = bosdyn.client.create_standard_sdk('GraphNavClient')
    sdk.register_service_client(NetworkComputeBridgeClient)
    robot = sdk.create_robot(HOSTNAME)
    bosdyn.client.util.authenticate(robot)

    #graph_nav_client = robot.ensure_client(GraphNavClient.default_service_name)

    #network_compute_client = robot.ensure_client(NetworkComputeBridgeClient.default_service_name)

    vision_model = VisionModel(robot)
    while True:
        start = time.time() # gives current time in seconds since Jan 1, 1970 (in Unix)
        vision_model.take_image("hand_color_image")
        while True:
            current_time = time.time()
            if current_time - start >= 1.0/30.0:
                break
