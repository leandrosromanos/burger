#!/bin/bash

# Stop leftover Cartographer processes
pkill -f 'cartographer_node' 2>/dev/null || true
pkill -f 'cartographer_occupancy_grid_node' 2>/dev/null || true

sleep 2

# ROS 2 Jazzy setup
source /opt/ros/jazzy/setup.bash
source ~/turtlebot3_ws/install/setup.bash

# TurtleBot3 model
export TURTLEBOT3_MODEL=burger

# Start Cartographer SLAM with simulation time
ros2 launch turtlebot3_cartographer cartographer.launch.py use_sim_time:=True