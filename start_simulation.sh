#!/bin/bash

# Stop leftover Gazebo Sim servers
pkill -f '^gz sim -r -s ' 2>/dev/null || true

sleep 2

# ROS 2 Jazzy setup
source /opt/ros/jazzy/setup.bash
source ~/turtlebot3_ws/install/setup.bash

export TURTLEBOT3_MODEL=burger

# Start Gazebo
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py