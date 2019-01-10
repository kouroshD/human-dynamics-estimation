# Copyright (C) 2018 Istituto Italiano di Tecnologia (IIT). All rights reserved.
# This software may be modified and distributed under the terms of the
# GNU Lesser General Public License v2.1 or any later version.

project(ipopt-test)

find_package(iDynTree REQUIRED)

add_executable(${PROJECT_NAME}  "ipopt-test.cpp")

target_link_libraries(${PROJECT_NAME} PUBLIC
    YARP::YARP_OS
    YARP::YARP_dev
    YARP::YARP_init
    iDynTree::idyntree-model
    iDynTree::idyntree-inverse-kinematics
    iDynTree::idyntree-high-level)

install(TARGETS ${PROJECT_NAME}
        RUNTIME DESTINATION ${CMAKE_INSTALL_FULL_BINDIR})

