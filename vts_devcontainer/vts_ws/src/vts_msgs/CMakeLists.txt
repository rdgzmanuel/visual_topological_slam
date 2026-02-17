cmake_minimum_required(VERSION 3.8)
project(vts_msgs)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)  # Needed for ImageTensor.msg
find_package(builtin_interfaces REQUIRED)
find_package(nav_msgs REQUIRED)

# Generate messages
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/CustomOdometry.msg"
  "msg/ImageTensor.msg"
  "msg/GraphNode.msg"
  "msg/FullGraph.msg"
  "msg/CommandMessage.msg"
  DEPENDENCIES builtin_interfaces geometry_msgs std_msgs sensor_msgs nav_msgs
)

# Ensure dependencies are exported
ament_export_dependencies(rosidl_default_runtime)

ament_package()
