#!/usr/bin/env python

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge

class ArucoMarkerDetectionNode:
    def __init__(self):
        rospy.init_node('aruco_marker_detection_node', anonymous=True)
        
        # Create a CvBridge for image conversion
        self.bridge = CvBridge()

        # Camera matrix and distortion coefficients (replace with actual calibration results)
        self.camera_matrix = np.array([
            [606.8311157226562, 0.0, 333.31500244140625],
            [0.0, 606.0000610351562, 246.64346313476562],
            [0.0, 0.0, 1.0]
        ])

        self.dist_coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0])  # Assuming no distortion based on the message

        # Define the ArUco dictionary and parameters
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        self.parameters = cv2.aruco.DetectorParameters()

        # Subscribe to the camera's image topic
        rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)

        # Subscribe to the robot's joint and pose topics
        rospy.Subscriber('/dobot_magician/joint_states', JointState, self.joint_state_callback)
        rospy.Subscriber('/dobot_magician/end_effector_poses', PoseStamped , self.pose_callback)

        # Store positions and joint states
        self.marker_positions = []
        self.robot_poses = []
        self.joint_states = None
        self.current_pose = None

    def image_callback(self, image_msg):
        try:
            # Convert the ROS Image message to OpenCV format
            frame = self.bridge.imgmsg_to_cv2(image_msg, 'bgr8')

            # Detect ArUco markers
            corners, ids, _ = cv2.aruco.detectMarkers(frame, self.aruco_dict, parameters=self.parameters)

            if ids is not None:
                # Estimate the pose of the detected markers
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.031, self.camera_matrix, self.dist_coeffs)
                
                # Log the position of each detected marker
                for i, marker_id in enumerate(ids):
                    marker_position_camera = tvecs[i][0]
                    rospy.loginfo(f"Marker ID {marker_id[0]} Position (Camera Frame): {marker_position_camera}")
                    self.marker_positions.append(marker_position_camera)
            else:
                rospy.logwarn("No markers detected")
        
        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")

    def joint_state_callback(self, joint_state_msg):
        # Update the latest joint states
        self.joint_states = joint_state_msg.position

    def pose_callback(self, pose_msg):
        # Update the latest end-effector pose
        self.current_pose = pose_msg

    def record_position(self):
        # Wait for a key press to record the current position
        input("Press Enter to record the current ArUco position and robot pose...")

        if self.marker_positions:
            latest_marker_position = self.marker_positions[-1]
            if self.current_pose and self.joint_states:
                # Correctly reference the position attribute within PoseStamped
                rospy.loginfo(f"Recording ArUco Position: {latest_marker_position}")
                rospy.loginfo(f"Recording Robot Pose: {self.current_pose.pose.position.x}, {self.current_pose.pose.position.y}, {self.current_pose.pose.position.z}")
                rospy.loginfo(f"Recording Joint States: {self.joint_states}")
                
                # Store the correspondence
                self.robot_poses.append((latest_marker_position, self.current_pose.pose.position, self.joint_states))

                # Update the extrinsic matrix after recording each position
                self.estimate_extrinsic_matrix()
            else:
                rospy.logwarn("No robot pose or joint states available")
        else:
            rospy.logwarn("No ArUco marker positions detected")

    def estimate_extrinsic_matrix(self):
        if len(self.robot_poses) < 3:
            rospy.logwarn("Not enough points collected to estimate extrinsic matrix")
            return

        # Separate the collected data into camera and robot points
        camera_points = np.array([pose[0] for pose in self.robot_poses], dtype=np.float64)
        robot_points = np.array([[pose[1].x, pose[1].y, pose[1].z] for pose in self.robot_poses], dtype=np.float64)

        # Estimate the transformation using cv2.estimateAffine3D
        retval, R = cv2.estimateAffine3D(camera_points, robot_points)[:2]

        if retval:
            extrinsic_matrix = np.eye(4)
            extrinsic_matrix[:3, :3] = R[:, :3]  # Extract the 3x3 rotation part
            extrinsic_matrix[:3, 3] = R[:, 3]    # Extract the translation part
            rospy.loginfo(f"Updated Extrinsic Matrix:\n{extrinsic_matrix}")
        else:
            rospy.logwarn("Failed to estimate extrinsic matrix")

    def start(self):
        while not rospy.is_shutdown():
            self.record_position()  # Wait for key press and record the current position
            if rospy.is_shutdown():
                break

if __name__ == '__main__':
    try:
        node = ArucoMarkerDetectionNode()
        node.start()
    except rospy.ROSInterruptException:
        pass
