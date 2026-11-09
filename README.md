# burger

RaspberryPi
username: burger_pi
password: burger
IPv4: 192.168.0.200

Bringup
ssh ubuntu@{IP_ADDRESS_OF_RASPBERRY_PI}
export TURTLEBOT3_MODEL=burger
ros2 launch turtlebot3_bringup robot.launch.py

Keyboard Control (allo terminal)
export TURTLEBOT3_MODEL=burger
ros2 run turtlebot3_teleop teleop_keyboard

SLAM
export TURTLEBOT3_MODEL=burger
ros2 launch turtlebot3_cartographer cartographer.launch.py
