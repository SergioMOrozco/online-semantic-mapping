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
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

MODEL_NAME = "handle-model"
HAND_MODEL = 'object-hand-model'
SERVER_NAME = "fetch-server"


class VisionModel:
    image_sources = [
            'frontleft_fisheye_image', 'frontright_fisheye_image',
            'left_fisheye_image', 'right_fisheye_image', 'back_fisheye_image'
        ]
    def __init__(self, graph_nav_client, network_compute_client, robot):
        self.graph_nav_client = graph_nav_client
        self.network_compute_client = network_compute_client
        self.robot = robot
        self.image_client = robot.ensure_client(ImageClient.default_service_name)

    def get_image(self, source):
        # We want to capture from one camera at a time.

        # Capture and save images to disk
        file_name = "images/" + str(uuid.uuid4()) + ".png"
        image_responses = self.image_client.get_image_from_sources([source])

        dtype = np.uint8

        img = np.frombuffer(image_responses[0].shot.image.data, dtype=dtype)
        img = cv2.imdecode(img, -1)

        # # Approximately rotate the image to level.
        # if image_responses[0].source.name[0:5] == "front":
        #     img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        #
        # elif image_responses[0].source.name[0:5] == "right":
        #     img = cv2.rotate(img, cv2.ROTATE_180)

        # Show the image
        cv2.imshow("Object", img)
        cv2.waitKey(15)

        vision_tform_body = bosdyn.client.frame_helpers.get_vision_tform_body(self.robot.get_frame_tree_snapshot())
        body_tform_vision = vision_tform_body.inverse()
        localization_state = self.graph_nav_client.get_localization_state()
        seed_tform_body = localization_state.localization.seed_tform_body

        # need to convert from geometry_pb2.SE3Pose to math_helpers.SE3Pose
        seed_tform_body =  math_helpers.SE3Pose(seed_tform_body.position.x,seed_tform_body.position.y,seed_tform_body.position.z, seed_tform_body.rotation)

        if seed_tform_body == None:
            print("Forgot to upload map")
            return None, None

        cv2.imwrite(file_name, img)
        seed_tform_vision = seed_tform_body * body_tform_vision
        print("seed tfrom vision: ", seed_tform_vision)

        return seed_tform_vision, file_name
