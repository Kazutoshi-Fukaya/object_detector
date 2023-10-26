cmake_minimum_required(VERSION 3.0.2)
project(object_detector_msgs)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  message_generation
  sensor_msgs
)

add_message_files(
    FILES
    ObjectPosition.msg
    ObjectPositions.msg
    BoundingBox2D.msg
    BoundingBox2DArray.msg
    BoundingBox3D.msg
    BoundingBox3DArray.msg
    ObjectPositionWithImage.msg
    ObjectPositionsWithImage.msg
)

generate_messages(
    DEPENDENCIES
    std_msgs
    sensor_msgs
)

catkin_package(
    CATKIN_DEPENDS roscpp rospy std_msgs message_runtime sensor_msgs
)