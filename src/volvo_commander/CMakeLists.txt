cmake_minimum_required(VERSION 3.8)
project(volvo_commander)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(tf2_kdl REQUIRED)

add_executable(arm_target_tester src/arm_target_tester.cpp)
ament_target_dependencies(arm_target_tester
  rclcpp
  geometry_msgs
  std_msgs
  moveit_ros_planning_interface
  tf2_ros
  tf2_geometry_msgs
)

add_executable(shape_drawer src/shape_drawer.cpp)
ament_target_dependencies(shape_drawer
  rclcpp
  geometry_msgs
  std_msgs
  moveit_ros_planning_interface
  moveit_msgs
  tf2_ros
  tf2_geometry_msgs
  tf2_eigen
  tf2_kdl
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
  OPTIONAL
)

install(TARGETS
  shape_drawer
  arm_target_tester
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()