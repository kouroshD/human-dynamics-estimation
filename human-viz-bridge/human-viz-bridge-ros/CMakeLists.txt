cmake_minimum_required(VERSION 3.0)
project(human-viz-bridge-ros)

set (CMAKE_CXX_STANDARD 11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  message_generation
  roscpp
  rospy
  sensor_msgs
  std_msgs
  tf
  urdf
)

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES human-viz-bridge-ros
#  CATKIN_DEPENDS geometry_msgs message_generation roscpp rospy sensor_msgs std_msgs tf urdf
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

link_directories(${catkin_LIBRARY_DIRS})

## Declare a C++ executable
## With catkin_make all packages are built within a single CMake context
## The recommended prefix ensures that target names across packages don't collide
add_executable(${PROJECT_NAME} src/human-viz-bridge-ros.cpp)

## Specify libraries to link a library or executable target against
target_link_libraries(${PROJECT_NAME}
   ${catkin_LIBRARIES}
)

#############
## Install ##
#############

# Install the executables to <prefix>/lib/<project>, so that
# they can be found by rosrun/roslaunch.
install(TARGETS ${PROJECT_NAME} DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION})

# Create a simple package.xml, which is necessary to be treated as
# a ROS package.
install(FILES package.xml DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

# Install the .launch file; this could go anywhere, but `launch` is a
# good pattern to follow.
file(GLOB LAUNCH_FILES ${PROJECT_SOURCE_DIR}/launch/*)
install(FILES ${LAUNCH_FILES} DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

# Install the urdf files in a `urdfs` directory
file(GLOB URDF_FILES ${PROJECT_SOURCE_DIR}/urdfs/*)
install(FILES ${URDF_FILES} DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/urdfs)

# Create an empty .catkin marker file, which causes
# rosrun/roslaunch to look in <prefix>/lib/project for executables
# for each <prefix> in CMAKE_PREFIX_PATH
install(FILES .catkin DESTINATION ${CMAKE_INSTALL_PREFIX}/)
