cmake_minimum_required(VERSION 3.8)
project(volvo_hardware_interface)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(hardware_interface REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(pluginlib REQUIRED)

# Build the hardware interface plugin as a shared library
add_library(volvo_hardware_interface SHARED
  src/volvo_hardware_interface.cpp
)

target_include_directories(volvo_hardware_interface PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(volvo_hardware_interface
  hardware_interface
  rclcpp
  rclcpp_lifecycle
  pluginlib
)

# Install the shared library
install(TARGETS volvo_hardware_interface
  DESTINATION lib
)

# Install headers
install(DIRECTORY include/
  DESTINATION include
)

# Export plugin description for pluginlib
pluginlib_export_plugin_description_file(hardware_interface volvo_hardware_interface.xml)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()