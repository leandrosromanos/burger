"""TurtleBot3 Jazzy: right-hand wall following with progressively inward laps.

This is a boundary-offset coverage heuristic, NOT complete map-based exploration.
Requires /scan, /odom and a robot accepting TwistStamped on /cmd_vel.
"""

import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistStamped


def wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class ObstacleAvoidance(Node):
    def __init__(self):
        super().__init__('obstacle_avoidance')

        # Tunable settings, in metres / seconds / radians.
        self.forward_speed = 0.06
        self.turn_speed = 0.4
        self.stop_distance = 0.40
        self.clear_distance = 0.55
        self.scan_timeout = 0.5
        self.odom_timeout = 0.5

        self.initial_wall_distance = 0.40
        self.inward_step = 0.25        # Extra wall standoff after each lap
        self.max_wall_distance = 2.00  # End instead of blindly losing the wall
        self.wall_kp = 1.7
        self.max_steering = 0.45
        self.offset_change_per_tick = 0.004  # Ramp new target gently (10 Hz)
        self.wall_lost_timeout = 5.0

        # Lap detection requires returning near the same position AND heading,
        # after actually leaving the start region and travelling far enough.
        self.lap_near_radius = 0.40
        self.lap_leave_radius = 0.85
        self.lap_heading_tolerance = math.radians(35)
        self.min_lap_travel = 2.5

        self.front = None
        self.front_right = None
        self.right = None
        self.last_scan_time = None
        self.last_odom_time = None
        self.pose = None  # (x, y, yaw)
        self.last_xy = None

        self.state = 'FOLLOW_WALL'
        self.wall_lost_since = None
        self.target_distance = self.initial_wall_distance
        self.active_distance = self.initial_wall_distance
        self.lap_count = 0
        self.lap_start = None
        self.lap_travel = 0.0
        self.left_start_area = False

        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, qos_profile_sensor_data
        )
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, qos_profile_sensor_data
        )
        self.publisher = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info('Wall-following inward-pass controller started')

    @staticmethod
    def get_distance(msg, min_angle, max_angle):
        valid = []
        for i, distance in enumerate(msg.ranges):
            angle = math.degrees(msg.angle_min + i * msg.angle_increment)
            angle = (angle + 180.0) % 360.0 - 180.0
            if (min_angle <= angle <= max_angle
                    and math.isfinite(distance)
                    and msg.range_min <= distance <= msg.range_max):
                valid.append(distance)
        return min(valid) if len(valid) >= 3 else None

    def scan_callback(self, msg):
        self.front = self.get_distance(msg, -20, 20)
        self.front_right = self.get_distance(msg, -50, -20)
        self.right = self.get_distance(msg, -105, -75)
        self.last_scan_time = time.monotonic()

    def odom_callback(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        self.pose = (p.x, p.y, yaw)
        self.last_odom_time = time.monotonic()
        if self.last_xy is not None and self.lap_start is not None:
            travelled = math.hypot(p.x - self.last_xy[0], p.y - self.last_xy[1])
            # Avoid counting sudden odometry resets as travel.
            if travelled < 0.3:
                self.lap_travel += travelled
        self.last_xy = (p.x, p.y)

    def move(self, linear, angular):
        cmd = TwistStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = 'base_link'
        cmd.twist.linear.x = float(linear)
        cmd.twist.angular.z = float(angular)
        self.publisher.publish(cmd)

    def update_lap(self):
        if self.lap_start is None:
            self.lap_start = self.pose
            return

        x, y, yaw = self.pose
        x0, y0, yaw0 = self.lap_start
        separation = math.hypot(x - x0, y - y0)
        if separation >= self.lap_leave_radius:
            self.left_start_area = True

        if (self.left_start_area
                and self.lap_travel >= self.min_lap_travel
                and separation <= self.lap_near_radius
                and abs(wrap_angle(yaw - yaw0)) <= self.lap_heading_tolerance):
            self.lap_count += 1
            next_distance = self.initial_wall_distance + self.lap_count * self.inward_step
            self.get_logger().info(
                f'Completed lap {self.lap_count}; next standoff {next_distance:.2f} m'
            )
            if next_distance > self.max_wall_distance:
                self.get_logger().info('Maximum inward offset reached; stopping.')
                self.state = 'DONE'
                return
            self.target_distance = next_distance
            self.lap_start = self.pose
            self.lap_travel = 0.0
            self.left_start_area = False

    def control_loop(self):
        if self.state == 'DONE':
            self.move(0.0, 0.0)
            return

        now = time.monotonic()
        if (self.last_scan_time is None or self.last_odom_time is None
                or now - self.last_scan_time > self.scan_timeout
                or now - self.last_odom_time > self.odom_timeout
                or self.pose is None):
            self.move(0.0, 0.0)
            return

        # A missing front measurement is not proof of empty space.
        if self.front is None:
            self.move(0.0, 0.0)
            return

        self.update_lap()
        if self.state == 'DONE':
            self.move(0.0, 0.0)
            return

        # Apply inward changes gradually instead of steering abruptly.
        self.active_distance = min(
            self.target_distance,
            self.active_distance + self.offset_change_per_tick,
        )

        # On a front collision risk, turn LEFT (away from a right-side wall).
        # Hysteresis prevents oscillation around stop_distance.
        if self.state == 'TURN_LEFT':
            if self.front < self.clear_distance:
                self.move(0.0, self.turn_speed)
                return
            self.state = 'FOLLOW_WALL'

        if (self.front < self.stop_distance
                or (self.front_right is not None
                    and self.front_right < self.stop_distance * 0.80)):
            self.state = 'TURN_LEFT'
            self.move(0.0, self.turn_speed)
            return

        if self.right is None:
            if self.wall_lost_since is None:
                self.wall_lost_since = now
            if now - self.wall_lost_since > self.wall_lost_timeout:
                self.get_logger().warning(
                    'Right wall not found for 5 seconds; stopping (coverage unverified).'
                )
                self.state = 'DONE'
                self.move(0.0, 0.0)
                return
            # Search gently for the wall on the right, with forward LiDAR safety.
            self.move(self.forward_speed * 0.5, -0.20)
            return

        self.wall_lost_since = None
        # Too close to right wall -> steer left (positive yaw).
        # Too far -> steer right (negative yaw).
        error = self.active_distance - self.right
        angular = max(-self.max_steering,
                      min(self.max_steering, self.wall_kp * error))
        # Reduce translational speed when making sharp corrections.
        speed = self.forward_speed * (0.6 if abs(angular) > 0.30 else 1.0)
        self.move(speed, angular)

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
