#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState
from cv_bridge import CvBridge
import cv2
import numpy as np
import random

# Load the dictionary for the AprilTag 36h11 family
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)

# Initialize the detector parameters
parameters = cv2.aruco.DetectorParameters()


camera_matrix = np.array([
            [599.22323485, 0.0, 339.16246969],
            [0.0, 600.35006132, 258.15884256],
            [0.0, 0.0, 1.0]
            ])

# intrinsic and extrinsic matrix from kalibr self calibrated 
# Intrinsics: via topic
            # [606.8311157226562, 0.0, 333.31500244140625],
            # [0.0, 606.0000610351562, 246.64346313476562],
            # [0.0, 0.0, 1.0]
    #  via 'kalibr_calibrate_cameras'
# [599.22323485 600.35006132 339.16246969 258.15884256]
# Distortion:
# [ 0.05088455 -0.13392238  0.0078341  -0.00211491]

dist_coeffs = np.array([ 0.05088455 ,-0.13392238 , 0.0078341,  -0.00211491, 0.0])  # Assuming no distortion based on the message

extrinsic_matrix = np.array([
    [-1.20040889, 15.74843836, 16.63322791, -3.28564496],
    [-0.36502823, 2.96074512, -0.45302241, 0.17169582],
    [-1.50155796, -21.47453118, -23.52317082, 4.91828151],
    [0.0, 0.0, 0.0, 1.0]
])



# ([
#     [0.7071, 0, -0.7071, 0.43],  # Rotation + Translation in X
#     [0, 1, 0, 0.23],             # Rotation + Translation in Y
#     [0.7071, 0, 0.7071, 0.26],   # Rotation + Translation in Z (26 cm height)
#     [0, 0, 0, 1]
# ])
def compute_correction_matrix(published_positions, actual_positions):
    P = np.hstack((published_positions, np.ones((published_positions.shape[0], 1))))
    A = np.hstack((actual_positions, np.ones((actual_positions.shape[0], 1))))
    centroid_P = np.mean(P[:, :3], axis=0)
    centroid_A = np.mean(A[:, :3], axis=0)
    P_centered = P[:, :3] - centroid_P
    A_centered = A[:, :3] - centroid_A
    H = np.dot(P_centered.T, A_centered)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)
    t = centroid_A - np.dot(R, centroid_P)
    correction_matrix = np.identity(4)
    correction_matrix[:3, :3] = R
    correction_matrix[:3, 3] = t
    return correction_matrix

class ImageDetectionAndControl:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('image_detection_and_control', anonymous=True)

        # Create a CvBridge object for image conversion
        self.bridge = CvBridge()

        # Publisher for goal position to Dobot's end effector pose
        self.goal_pub = rospy.Publisher('/dobot_magician/target_end_effector_pose', Pose, queue_size=10)

        # Subscribe to the RGB image topic from RealSense
        rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)
        rospy.Subscriber('/doobot_magician/joint_states', JointState, self.joint_state_callback)
        self.joint_angles = [0, 0, 0, 0]
        rospy.loginfo("Image Detection and Control Node Initialized")
        self.correction_matrix = np.identity(4)
    def joint_state_callback(self, msg):
        # Map the joint names to indices for better flexibility
        joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_joint"]
        name_to_index = {name: i for i, name in enumerate(joint_names)}

        # Extract joint angles based on the joint names in the message
        for joint in msg.name:
            if joint in name_to_index:
                idx = name_to_index[joint]
                self.joint_angles[idx] = msg.position[idx]
    
    
    def image_callback(self, data):
        try:
            # Convert the ROS Image message to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(data, 'bgr8')

            # Detect ArUco markers
            corners, ids, rejected = cv2.aruco.detectMarkers(cv_image, aruco_dict, parameters=parameters)

            if ids is not None:
                # Draw the detected markers
                cv2.aruco.drawDetectedMarkers(cv_image, corners, ids)
                rospy.loginfo(f"Detected ArUco IDs: {ids.flatten()}")

                # Transform the first detected marker's first corner to Dobot coordinates
                for i, marker_id in enumerate(ids):
                    # Pose estimation for each marker
                    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.031, camera_matrix, dist_coeffs)
                    marker_position_camera = tvecs[i][0]  # Extracting the translation vector
                    
                    rospy.loginfo(f"Marker ID {marker_id[0]} Position (Camera Frame): {marker_position_camera}")

                    # Transform to Dobot coordinates
                    dobot_coords = self.transform_to_dobot(marker_position_camera)

                    rospy.loginfo(f"Dobot coordinates for marker {marker_id[0]}: {dobot_coords}")

                    # Send goal to Dobot
                    self.send_goal(dobot_coords)

            else:
                rospy.logwarn("No ArUco markers detected")        
        except Exception as e:
                rospy.logerr(f"Error processing image: {e}")
                
                
    def dh_transformation(self, theta, d, a, alpha):
        """Return the DH transformation matrix."""
        return np.array([
            [np.cos(theta), -np.sin(theta) * np.cos(alpha), np.sin(theta) * np.sin(alpha), a * np.cos(theta)],
            [np.sin(theta), np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha), a * np.sin(theta)],
            [0, np.sin(alpha), np.cos(alpha), d],
            [0, 0, 0, 1]
        ])

    def forward_kinematics(self, theta1, theta2, theta3, theta4):
        """Calculate the forward kinematics for Dobot Magician."""
        # Define the DH parameters (lengths in meters)
        L1, L2, L3, L4 = 0.103, 0.135, 0.160, 0.015  # in meters

        # Create the transformation matrices using DH parameters
        T0_1 = self.dh_transformation(theta1, d=L1, a=0, alpha=np.pi/2)
        T1_2 = self.dh_transformation(theta2, d=0, a=L2, alpha=0)
        T2_3 = self.dh_transformation(theta3, d=0, a=L3, alpha=0)
        T3_4 = self.dh_transformation(theta4, d=0, a=0, alpha=0)

        # Compute the overall transformation from base to end-effector
        T_base_ee = T0_1 @ T1_2 @ T2_3 @ T3_4
        return T_base_ee

    def transform_to_dobot(self, camera_coordinates):
        """Transform the camera coordinates to Dobot end-effector coordinates."""
        # Step 1: Convert camera coordinates to homogeneous
        camera_coordinates_homogeneous = np.append(camera_coordinates, 1)

        # Step 2: Apply extrinsic transformation from camera to robot base
        base_coordinates_homogeneous = np.dot(extrinsic_matrix, camera_coordinates_homogeneous)

        # Step 3: Calculate the forward kinematics to get the end-effector pose        
        theta1, theta2, theta3, theta4 = self.joint_angles   
        T_base_ee = self.forward_kinematics(theta1, theta2, theta3, theta4)

        # Step 4: Transform base coordinates to end-effector frame
        end_effector_coordinates_homogeneous = np.dot(T_base_ee, base_coordinates_homogeneous)

        # Step 5: Extract the 3D coordinates
        dobot_coordinates = end_effector_coordinates_homogeneous[:3]
        return dobot_coordinates

    def normalize_quaternion(self,x, y, z, w):
        norm = np.sqrt(x**2 + y**2 + z**2 + w**2)
        return x / norm, y / norm, z / norm, w / norm

    
    def send_goal(self, target_coords):
        goal = Pose()
        goal.position.x, goal.position.y, goal.position.z = target_coords
        goal.position.z = -0.05
        
        # Adapt the goal positions within the min and max range
        goal.position.x = max(0.16, min(0.24, goal.position.x))  # Clamp x to be within 0.16 and 0.24
        goal.position.y = max(-0.1, min(0.13, goal.position.y))  # Clamp y to be within -0.1 and 0.13
        qx, qy, qz, qw = self.normalize_quaternion(0, 0, 0, 1)
        goal.orientation.x, goal.orientation.y, goal.orientation.z, goal.orientation.w = qx, qy, qz, qw



        # goal.position.x = random.uniform(0.16, 0.24)  # Min is 0.16 and max is 0.24
        # goal.position.y = random.uniform(-0.1, 0.13)  # Min is -0.1 and max is 0.13


        # Publish the goal to move the Dobot's end effector
        self.goal_pub.publish(goal)
        rospy.loginfo("Published goal to Dobot")
        rospy.sleep(3)
    def start(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        node = ImageDetectionAndControl()
        node.start()
    except rospy.ROSInterruptException:
        pass
