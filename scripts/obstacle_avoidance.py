import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import TwistStamped


class ObstacleAvoidance(Node):

    def __init__(self):
        super().__init__('obstacle_avoidance')

        # ==========================================
        # 1. ROBOT SETTINGS
        # ==========================================

        self.forward_speed = 0.06       # meters/second
        self.turn_speed = 0.4           # radians/second

        self.stop_distance = 0.45       # meters
        self.clear_distance = 0.60      # meters

        self.scan_timeout = 0.5         # seconds

        # ==========================================
        # 2. LIDAR MEASUREMENTS
        # ==========================================

        self.front = None
        self.left = None
        self.right = None

        self.last_scan_time = None

        # ==========================================
        # 3. ROBOT STATE
        # ==========================================

        self.turning = False
        self.turn_direction = 1.0

        # ==========================================
        # 4. SUBSCRIBE TO LIDAR DATA
        # ==========================================

        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            qos_profile_sensor_data
        )

        # ==========================================
        # 5. PUBLISH MOVEMENT COMMANDS
        # ==========================================

        self.publisher = self.create_publisher(
            TwistStamped,
            '/cmd_vel',
            10
        )

        # ==========================================
        # 6. CONTROL LOOP - 10 TIMES PER SECOND
        # ==========================================

        self.timer = self.create_timer(
            0.1,
            self.control_loop
        )

        self.get_logger().info(
            'Obstacle avoidance node started'
        )

    # ==============================================
    # 7. FIND CLOSEST OBSTACLE IN AN ANGULAR REGION
    # ==============================================

    def get_distance(self, msg, min_angle, max_angle):

        distances = []

        for i, distance in enumerate(msg.ranges):

            # Calculate the angle of this measurement
            angle = math.degrees(
                msg.angle_min + i * msg.angle_increment
            )

            # Convert angle to -180° to +180°
            angle = (angle + 180) % 360 - 180

            # Check whether the measurement belongs
            # to the selected region
            if min_angle <= angle <= max_angle:

                # Keep only valid LiDAR measurements
                if (
                    math.isfinite(distance)
                    and msg.range_min <= distance <= msg.range_max
                ):
                    distances.append(distance)

        # No sufficient valid measurements
        if len(distances) < 3:
            return None

        # Return distance to closest obstacle
        return min(distances)

    # ==============================================
    # 8. RECEIVE AND PROCESS LIDAR MEASUREMENTS
    # ==============================================

    def scan_callback(self, msg):

        # Front: -35° to +35°
        self.front = self.get_distance(
            msg, -35, 35
        )

        # Left: +20° to +100°
        self.left = self.get_distance(
            msg, 20, 100
        )

        # Right: -100° to -20°
        self.right = self.get_distance(
            msg, -100, -20
        )

        # Save the arrival time of the latest scan
        self.last_scan_time = time.monotonic()

    # ==============================================
    # 9. SEND MOVEMENT COMMANDS
    # ==============================================

    def move(self, linear, angular):

        command = TwistStamped()

        # Set timestamp
        command.header.stamp = (
            self.get_clock().now().to_msg()
        )

        command.header.frame_id = 'base_link'

        # Forward / backward velocity
        command.twist.linear.x = float(linear)

        # Rotational velocity
        command.twist.angular.z = float(angular)

        # Publish movement command to Gazebo
        self.publisher.publish(command)

    # ==============================================
    # 10. MAIN OBSTACLE AVOIDANCE ALGORITHM
    # ==============================================

    def control_loop(self):

        # CASE 1: No LiDAR data received
        if self.last_scan_time is None:

            self.move(0.0, 0.0)
            return

        # CASE 2: LiDAR data is outdated
        if (
            time.monotonic() - self.last_scan_time
            > self.scan_timeout
        ):

            self.move(0.0, 0.0)
            return

        # CASE 3: Missing valid measurements
        if any(
            d is None
            for d in (self.front, self.left, self.right)
        ):

            self.move(0.0, 0.0)
            return

        # CASE 4: Robot is already turning
        if self.turning:

            if self.front < self.clear_distance:

                self.move(
                    0.0,
                    self.turn_direction * self.turn_speed
                )

                return

            # Front is clear
            self.turning = False

        # CASE 5: Obstacle detected ahead
        if self.front < self.stop_distance:

            self.turning = True

            # Choose side with more space
            if self.left >= self.right:

                self.turn_direction = 1.0  # Left

            else:

                self.turn_direction = -1.0  # Right

            self.get_logger().info(
                f'Obstacle detected at {self.front:.2f} m'
            )

            # Rotate without moving forward
            self.move(
                0.0,
                self.turn_direction * self.turn_speed
            )

        # CASE 6: Path is clear
        else:

            self.move(
                self.forward_speed,
                0.0
            )

    # ==============================================
    # 11. STOP ROBOT
    # ==============================================

    def stop(self):
        self.move(0.0, 0.0)


def main(args=None):

    rclpy.init(args=args)

    node = ObstacleAvoidance()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()