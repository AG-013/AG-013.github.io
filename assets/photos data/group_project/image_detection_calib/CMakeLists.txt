cmake_minimum_required(VERSION 3.0.2)
project(image_detection_calib)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  cv_bridge
  geometry_msgs
  image_transport
)

## Find OpenCV (required for your image processing)
find_package(OpenCV REQUIRED)

## Declare this as a catkin package
catkin_package()

## Specify additional locations of header files
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}  # Include OpenCV directories
)

## Add your Python scripts to the catkin package (for Python nodes)
catkin_install_python(PROGRAMS
  src/image_detect.py      # image detection node
  src/extrinsic_calibration_node.py # extrinsic calibration node
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
