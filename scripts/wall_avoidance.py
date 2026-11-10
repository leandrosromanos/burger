import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan


def angle_difference(a, b):
    """Signed angular difference in radians."""
    return math.atan2(math.sin(a - b), math.cos(a - b))


class WallFollowing(Node):
    def __init__(self):
        super().__init__('wall_following')

        # Movement settings: starting values for Gazebo.
        self.speed = 0.05                 # m/s
        self.wall_distance = 0.40         # m from LiDAR
        self.front_stop = 0.45            # m
        self.turn_speed = 0.35            # rad/s
        self.gain = 1.2                   # wall-distance correction

        # Loop-detection settings.
        self.minimum_travel = 4.0         # m before accepting a loop
        self.return_radius = 0.30         # m from starting position
        self.heading_tolerance = math.radians(30)
        self.maximum_duration = 300.0     # simulated seconds

        self.scan = None
        self.scan_received = None
        self.pose = None
        self.odom_received = None
        self.previous_position = None

        self.start_pose = None
        self.start_time = None
        self.travelled = 0.0
        self.left_start_area = False

        self.state = 'SEEK_WALL'
        self.turning_left = False

        self.publisher = self.create_publisher(
            TwistStamped, '/cmd_vel', 10
        )

        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback,
            qos_profile_sensor_data
        )

        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback,
            qos_profile_sensor_data
        )

        self.timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info(
            'Waiting for /scan and /odom. Simulation test only.'
        )

    def scan_callback(self, msg):
        self.scan = msg
        self.scan_received = time.monotonic()

    def odom_callback(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation

        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )

        position = (p.x, p.y)

        if self.state == 'FOLLOW_WALL':
            if self.previous_position is not None:
                self.travelled += math.hypot(
                    p.x - self.previous_position[0],
                    p.y - self.previous_position[1]
                )

        self.previous_position = position
        self.pose = (p.x, p.y, yaw)
        self.odom_received = time.monotonic()

    def sector_distance(self, low, high):
        """Nearest usable range in a sector, angles in degrees."""
        distances = []

        for i, distance in enumerate(self.scan.ranges):
            angle = math.degrees(
                self.scan.angle_min
                + i * self.scan.angle_increment
            )
            angle = (angle + 180.0) % 360.0 - 180.0

            if not low <= angle <= high:
                continue

            if (
                math.isfinite(distance)
                and self.scan.range_min
                <= distance <= self.scan.range_max
            ):
                distances.append(distance)

            elif distance == math.inf:
                # Gazebo-only assumption:
                # positive infinity means no hit within sensor range.
                distances.append(self.scan.range_max)

        if len(distances) < 3:
            return None

        return min(distances)

    def move(self, linear, angular):
        command = TwistStamped()
        command.header.stamp = self.get_clock().now().to_msg()
        command.header.frame_id = 'base_link'
        command.twist.linear.x = float(linear)
        command.twist.angular.z = float(angular)
        self.publisher.publish(command)

    def finish(self, reason):
        self.state = 'DONE'
        self.move(0.0, 0.0)
        self.get_logger().info(reason)

    def control_loop(self):
        if self.state == 'DONE':
            self.move(0.0, 0.0)
            return

        now = time.monotonic()

        # Wait for fresh sensor and position data.
        if (
            self.scan_received is None
            or self.odom_received is None
            or now - self.scan_received > 0.5
            or now - self.odom_received > 0.5
        ):
            self.move(0.0, 0.0)
            return

        sim_time = self.get_clock().now().nanoseconds / 1e9

        if self.start_time is None:
            self.start_time = sim_time

        if sim_time - self.start_time > self.maximum_duration:
            self.finish('Time limit reached. Map completion not determined.')
            return

        front = self.sector_distance(-35, 35)
        right = self.sector_distance(-100, -70)

        if front is None or right is None:
            self.move(0.0, 0.0)
            return

        # Initially approach an obstacle, then turn left
        # to place its boundary on the robot's right.
        if self.state == 'SEEK_WALL':
            if front < self.front_stop:
                self.state = 'ALIGN'
                self.move(0.0, self.turn_speed)
            else:
                self.move(self.speed, 0.0)
            return

        if self.state == 'ALIGN':
            if front > 0.60 and right < 0.65:
                self.state = 'FOLLOW_WALL'
                self.start_pose = self.pose
                self.previous_position = self.pose[:2]
                self.travelled = 0.0
                self.move(0.0, 0.0)
                self.get_logger().info('Following right wall.')
            else:
                self.move(0.0, self.turn_speed)
            return

        # Check whether we have returned to the loop start.
        x, y, yaw = self.pose
        sx, sy, start_yaw = self.start_pose
        distance_from_start = math.hypot(x - sx, y - sy)

        if distance_from_start > 0.80:
            self.left_start_area = True

        if (
            self.left_start_area
            and self.travelled >= self.minimum_travel
            and distance_from_start < self.return_radius
            and abs(angle_difference(yaw, start_yaw))
            < self.heading_tolerance
        ):
            self.finish(
                'Possible boundary loop completed. '
                'Stopped for inspection; map may still be incomplete.'
            )
            return

        # Use two front thresholds to avoid rapid switching.
        if front < self.front_stop:
            self.turning_left = True

        if self.turning_left:
            if front < 0.60:
                self.move(0.0, self.turn_speed)
                return
            self.turning_left = False

        # Right wall lost: slowly curve right to find it.
        if right > 0.80:
            self.move(0.025, -0.30)
            return

        # Too close to right wall -> positive correction -> left.
        # Too far from right wall -> negative correction -> right.
        angular = self.gain * (self.wall_distance - right)
        angular = max(-self.turn_speed, min(self.turn_speed, angular))

        self.move(self.speed, angular)


def main(args=None):
    rclpy.init(args=args)
    node = WallFollowing()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.move(0.0, 0.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()