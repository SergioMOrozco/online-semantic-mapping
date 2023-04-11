import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseWithCovarianceStamped
from cv_bridge import CvBridge
from gazebo_msgs.msg import LinkStates
import cv2
import os
import uuid
import numpy as np
import math
import glob
from data_capture_model import DataCaptureModel
from scipy.spatial.transform import Rotation as R
#from bosdyn.client import math_helpers

class Nodo(object):
    def __init__(self, link_name):
        # Params
        self.link_name = link_name
        self.image = None
        self.transformation_matrix = None
        self.br = CvBridge()
        # Node cycle rate (in Hz).
        self.loop_rate = rospy.Rate(30)

        # Publishers
        self.pub = rospy.Publisher('imagetimer', Image,queue_size=10)

        # Subscribers
        rospy.Subscriber("/camera/rgb/image_raw",Image,self.callback_camera)
        rospy.Subscriber("/amcl_pose",PoseWithCovarianceStamped,self.callback_pose)

    def get_rotation_from(self, yaw, pitch, roll):
        yaw_rotation = np.array([[math.cos(yaw),-math.sin(yaw),0],
                        [math.sin(yaw),math.cos(yaw),0],
                        [0,0,1]])

        pitch_rotation = np.array([[math.cos(pitch),0,math.sin(pitch)],
                        [0,1,0],
                        [-math.sin(pitch),0,math.cos(pitch)]])

        roll_rotation = np.array([[1,0,0],
                        [0,math.cos(roll),-math.sin(roll)],
                        [0,math.sin(roll),math.cos(roll)]])

        return np.dot(yaw_rotation,np.dot(pitch_rotation,roll_rotation))

    def qvec2rotmat(self,qvec):
        return np.array([
		[
			1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
			2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
			2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]
		], [
			2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
			1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
			2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]
		], [
			2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
			2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
			1 - 2 * qvec[1]**2 - 2 * qvec[2]**2
		]
	])

    def quaternion_rotation_matrix(self,Q):
        """
        Covert a quaternion into a full three-dimensional rotation matrix.

        Input
        :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3)

        Output
        :return: A 3x3 element matrix representing the full 3D rotation matrix.
                 This rotation matrix converts a point in the local reference
                 frame to a point in the global reference frame.
        """
        # Extract the values from Q
        q0 = Q[0]
        q1 = Q[1]
        q2 = Q[2]
        q3 = Q[3]

        # First row of the rotation matrix
        r00 = 2 * (q0 * q0 + q1 * q1) - 1
        r01 = 2 * (q1 * q2 - q0 * q3)
        r02 = 2 * (q1 * q3 + q0 * q2)

        # Second row of the rotation matrix
        r10 = 2 * (q1 * q2 + q0 * q3)
        r11 = 2 * (q0 * q0 + q2 * q2) - 1
        r12 = 2 * (q2 * q3 - q0 * q1)

        # Third row of the rotation matrix
        r20 = 2 * (q1 * q3 - q0 * q2)
        r21 = 2 * (q2 * q3 + q0 * q1)
        r22 = 2 * (q0 * q0 + q3 * q3) - 1

        # 3x3 rotation matrix
        rot_matrix = np.array([[r00, r01, r02],
                               [r10, r11, r12],
                               [r20, r21, r22]])

        return rot_matrix

    def callback_pose(self, data):
        try:
          #ind = data.name.index(self.link_name)
          #self.link_pose = data.pose[ind]
          self.link_pose = data.pose.pose
          #print(data)

          bottom = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4])


          x = self.link_pose.position.x
          y = self.link_pose.position.y
          z = self.link_pose.position.z

          t = np.array([z,x,y]).reshape([3,1])
          #t = np.array([x,y,z]).reshape([3,1])

          rotation = [
                    self.link_pose.orientation.w,
                    self.link_pose.orientation.x,
                    self.link_pose.orientation.y,
                    self.link_pose.orientation.z,
                    ]

          qw = self.link_pose.orientation.w
          qx = self.link_pose.orientation.x
          qy = self.link_pose.orientation.y
          qz = self.link_pose.orientation.z


          #roll = math.atan2(2 * rotation[2] * rotation[0] - 2 * rotation[1] * rotation[3], 1 -2 *rotation[2] * rotation[2] - 2 * rotation[3] * rotation[3])
          #pitch = math.atan2(2 * rotation[1] * rotation[0] - 2 * rotation[2] * rotation[3], 1 -2 *rotation[1] * rotation[1] - 2 * rotation[3] * rotation[3])
          #yaw = math.asin(2 * rotation[1] * rotation[2] + 2 * rotation[3] * rotation[0])

          yaw = math.atan2(2*(rotation[0] * rotation[3] + rotation[1] * rotation[2]), 1 - 2 * (rotation[2] * rotation[2] + rotation[3] * rotation[3]))
          #yaw = math.atan2(2.0*(qy*qz + qw*qx), qw*qw - qx*qx - qy*qy + qz*qz)

          #rotation_matrix= np.array([[math.cos(yaw),-math.sin(yaw),0],
          #              [math.sin(yaw),math.cos(yaw),0],
          #              [0,0,1]])
          #rotation_matrix= np.array([[1,0,0],
          #              [0,1,0],
          #              [0,0,1]])
          rotation_matrix = self.get_rotation_from(-math.pi/2,math.pi/2 + yaw,0)

          #rotation_matrix = np.array([[1,0,0],
          #              [0,math.cos(yaw),-math.sin(yaw)],
          #              [0,math.sin(yaw),math.cos(yaw)]])



          #rotation_matrix = self.get_rotation_from(yaw,pitch,roll)

          #rotation_matrix = self.quaternion_rotation_matrix(rotation)

          #rotation_matrix = self.qvec2rotmat(rotation)

          self.transformation_matrix = np.concatenate([np.concatenate([rotation_matrix, t], 1), bottom], 0)
          #self.transformation_matrix = np.linalg.inv(self.transformation_matrix)






          #rotation_matrix = np.transpose(R.from_quat(rotation).as_matrix())
          #print(rotation_matrix)

          #T = np.array([x,y,z])
          #T.shape = (3,1)

          #tmp = np.hstack((rotation_matrix,T))

          #newrow = [0,0,0,1]
          #self.transformation_matrix = np.vstack([tmp, newrow])



        except ValueError:
          pass
    def callback_camera(self, msg):
        #rospy.loginfo('Image received...')
        self.image = self.br.imgmsg_to_cv2(msg)

        #cv2.imshow('image',self.image)
        #file_name = "images/" + str(uuid.uuid4()) + ".png"
        #cv2.imwrite(file_name, self.image)
        #cv2.waitKey(1)


    def start(self):


        files = glob.glob('/home/sorozco0612/dev/instant-ngp/data/nerf/simulator/images/*')
        for f in files:
            os.remove(f)
        data_capture_model = DataCaptureModel("/home/sorozco0612/dev/instant-ngp/data/nerf/simulator/transforms.json")
        rospy.loginfo("Timing images")
        #rospy.spin()
        while not rospy.is_shutdown():
            if self.image is not None and self.transformation_matrix is not None:
                #rospy.loginfo('publishing image')
                file_name = "images/" + str(uuid.uuid4()) + ".png"

                data_capture_model.add_frame(file_name, self.transformation_matrix)
                cv2.imwrite("/home/sorozco0612/dev/instant-ngp/data/nerf/simulator/" + file_name, self.image)

                cv2.imshow('image',self.image)
                data_capture_model.write_to_file()

                #print(self.transformation_matrix)

                self.image = None
                self.transformation_matrix = None

                cv2.waitKey(1)
            self.loop_rate.sleep()


if __name__ == '__main__':
    rospy.init_node("test", anonymous=True)
    my_node = Nodo("/::base_link")
    my_node.start()
