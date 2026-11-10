"""TurtleBot3 Burger / ROS 2 Jazzy: straight search, gentle left avoidance,
right-wall tracking and progressively larger right-wall clearance per lap.

Run alongside Gazebo and Cartographer (uses /scan and /odom, not /map).
This is a coverage heuristic, NOT proof of complete exploration. A nearby
obstacle can be mistaken for an outer wall. Verify cmd_vel message type.
"""
import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistStamped


def wrap_angle(a):
    return math.atan2(math.sin(a), math.cos(a))


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class WallCoverage(Node):
    def __init__(self):
        super().__init__('wall_coverage')

        # ---- Adjust these first ----
        self.forward_speed = 0.075       # m/s in open space / normal following
        self.cautious_speed = 0.045      # m/s when near a front obstacle
        self.left_turn_speed = 0.28     # rad/s: slow, controlled LEFT turn
        self.front_stop = 0.29          # m: rotate only when VERY close
        self.front_resume = 0.37        # m: resume after enough room opens
        self.front_right_stop = 0.23    # m: protect the front-right corner
        self.first_wall_distance = 0.36 # m: desired initial right-wall gap
        self.wall_acquire_range = 0.70  # m: right wall must be detected this close
        self.inward_step = 0.20         # m: extra gap after each lap
        self.max_wall_distance = 1.56   # m: stop BEFORE requesting a larger gap
        self.offset_ramp = 0.004        # m per controller tick (10 Hz)
        self.wall_kp = 0.9             # proportional right-wall steering gain
        self.max_right_steer = 0.10    # rad/s: shallow right arcs, not circles
        self.max_left_steer = 0.20     # rad/s: more authority to avoid right wall
        self.wall_lost_timeout = 8.0   # s: stop if right wall cannot be recovered
        self.wall_range_extra = 0.70  # m: ignore far readings while following
        self.scan_timeout = 0.7
        self.odom_timeout = 0.7

        # Lap completion is based on right-wall tracking + odometry loop closure.
        self.lap_leave_radius = 0.90    # m: must leave start neighbourhood
        self.lap_return_radius = 0.40   # m: must return near start
        self.lap_heading_tolerance = math.radians(40)
        self.min_lap_distance = 3.0     # m: minimum actual distance driven
        self.min_lap_seconds = 35.0    # s: must follow long enough
        self.min_wall_fraction = 0.65  # right wall seen at least 65% of lap time

        # Sensor readings and odometry.
        self.front = None
        self.front_right = None
        self.right = None
        self.last_scan = None
        self.last_odom = None
        self.pose = None
        self.previous_xy = None

        # States: SEARCH, FOLLOW, AVOID_LEFT, SHIFT_INWARD, DONE.
        self.state = 'SEARCH'
        self.resume_after_avoid = 'SEARCH'
        self.wall_missing_since = None
        self.active_distance = self.first_wall_distance
        self.target_distance = self.first_wall_distance
        self.lap_count = 0
        self.lap_origin = None
        self.lap_distance = 0.0
        self.lap_start_time = None
        self.lap_seen_ticks = 0
        self.lap_total_ticks = 0
        self.left_lap_origin = False

        self.create_subscription(LaserScan, '/scan', self.on_scan,
                                 qos_profile_sensor_data)
        self.create_subscription(Odometry, '/odom', self.on_odom,
                                 qos_profile_sensor_data)
        self.publisher = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.create_timer(0.1, self.control)
        self.get_logger().info('Started: go straight, avoid left, follow right wall.')

    @staticmethod
    def sector(scan, low, high):
        """Closest range in angular sector. +inf means clear rays, None invalid."""
        valid = []
        clear = 0
        for i, d in enumerate(scan.ranges):
            angle = math.degrees(scan.angle_min + i * scan.angle_increment)
            angle = (angle + 180) % 360 - 180
            if not low <= angle <= high:
                continue
            if math.isfinite(d) and scan.range_min <= d <= scan.range_max:
                valid.append(d)
            elif math.isinf(d) and d > 0:
                clear += 1
        if valid:
            return min(valid)
        if clear >= 3:
            return math.inf
        return None

    def on_scan(self, scan):
        self.front = self.sector(scan, -18, 18)
        self.front_right = self.sector(scan, -55, -20)
        self.right = self.sector(scan, -105, -75)
        self.last_scan = time.monotonic()

    def on_odom(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = math.atan2(2 * (q.w*q.z + q.x*q.y),
                         1 - 2 * (q.y*q.y + q.z*q.z))
        self.pose = (p.x, p.y, yaw)
        self.last_odom = time.monotonic()
        if self.previous_xy is not None and self.lap_origin is not None:
            delta = math.hypot(p.x - self.previous_xy[0],
                               p.y - self.previous_xy[1])
            if delta < 0.20:  # ignore implausible odometry jumps
                self.lap_distance += delta
        self.previous_xy = (p.x, p.y)

    def move(self, linear, angular):
        cmd = TwistStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = 'base_link'
        cmd.twist.linear.x = float(linear)
        cmd.twist.angular.z = float(angular)
        self.publisher.publish(cmd)

    def begin_lap(self, now):
        self.lap_origin = self.pose
        self.lap_distance = 0.0
        self.lap_start_time = now
        self.lap_seen_ticks = 0
        self.lap_total_ticks = 0
        self.left_lap_origin = False
        self.get_logger().info(
            f'Lap {self.lap_count + 1} start; target gap {self.target_distance:.2f} m')

    def check_lap(self, now, wall_seen):
        if self.lap_origin is None:
            self.begin_lap(now)
            return
        self.lap_total_ticks += 1
        if wall_seen:
            self.lap_seen_ticks += 1
        x, y, yaw = self.pose
        x0, y0, yaw0 = self.lap_origin
        separation = math.hypot(x - x0, y - y0)
        if separation >= self.lap_leave_radius:
            self.left_lap_origin = True
        fraction = self.lap_seen_ticks / max(self.lap_total_ticks, 1)
        if not (self.left_lap_origin
                and self.lap_distance >= self.min_lap_distance
                and now - self.lap_start_time >= self.min_lap_seconds
                and fraction >= self.min_wall_fraction
                and wall_seen
                and separation <= self.lap_return_radius
                and abs(wrap_angle(yaw - yaw0)) <= self.lap_heading_tolerance):
            return

        self.lap_count += 1
        next_target = self.first_wall_distance + self.lap_count * self.inward_step
        self.get_logger().info(
            f'Completed lap {self.lap_count}, wall visible {fraction:.0%} of the time.')
        if next_target > self.max_wall_distance + 1e-6:
            self.state = 'DONE'
            self.get_logger().info('Maximum inward target reached; STOP (coverage unverified).')
            self.move(0, 0)
            return
        self.target_distance = next_target
        self.state = 'SHIFT_INWARD'
        self.lap_origin = None  # start next lap AFTER new clearance is achieved
        self.get_logger().info(f'Increasing right-wall clearance to {next_target:.2f} m.')

    def control(self):
        if self.state == 'DONE':
            self.move(0, 0)
            return
        now = time.monotonic()
        if (self.last_scan is None or self.last_odom is None
                or self.pose is None or now - self.last_scan > self.scan_timeout
                or now - self.last_odom > self.odom_timeout
                or self.front is None):
            self.move(0, 0)
            return

        # Clear sensors can return infinity; this is NOT a nearby wall.
        right_seen = (self.right is not None
                      and math.isfinite(self.right)
                      and self.right <= max(self.wall_acquire_range,
                                            self.active_distance + self.wall_range_extra))
        front_close = (self.front < self.front_stop
                       or (self.front_right is not None
                           and self.front_right < self.front_right_stop))

        # SEARCH: travel STRAIGHT, left turn only when a wall is really close.
        if self.state == 'SEARCH':
            if front_close:
                self.resume_after_avoid = 'SEARCH'
                self.state = 'AVOID_LEFT'
                self.move(0, self.left_turn_speed)
                return
            if (self.right is not None and math.isfinite(self.right)
                    and self.right <= self.wall_acquire_range):
                self.state = 'FOLLOW'
                self.wall_missing_since = None
                self.target_distance = self.first_wall_distance
                self.active_distance = self.first_wall_distance
                self.begin_lap(now)  # lap origin is at the wall, NOT spawn point
            else:
                self.move(self.forward_speed, 0)
                return

        # AVOID_LEFT: turn on spot only until front and front-right clear.
        if self.state == 'AVOID_LEFT':
            if (self.front < self.front_resume
                    or (self.front_right is not None
                        and self.front_right < self.front_right_stop + 0.05)):
                self.move(0, self.left_turn_speed)
                return
            # If searching, go back to straight movement; otherwise resume wall.
            self.state = self.resume_after_avoid
            self.move(self.cautious_speed, 0)
            return

        # FOLLOW / SHIFT_INWARD: obstacles still take priority over wall steering.
        if front_close:
            self.resume_after_avoid = self.state
            self.state = 'AVOID_LEFT'
            self.move(0, self.left_turn_speed)
            return

        # A little more clearance is the only difference between successive laps.
        if self.state == 'SHIFT_INWARD':
            self.active_distance = min(self.target_distance,
                                       self.active_distance + self.offset_ramp)
            if self.active_distance >= self.target_distance - 1e-6:
                self.state = 'FOLLOW'
                self.begin_lap(now)  # new origin on the new offset trajectory

        # Don't blindly drive into unknown space indefinitely when right wall lost.
        if not right_seen:
            if self.wall_missing_since is None:
                self.wall_missing_since = now
            if now - self.wall_missing_since > self.wall_lost_timeout:
                self.get_logger().warning('Lost right wall: STOP (not complete).')
                self.state = 'DONE'
                self.move(0, 0)
                return
            # Move straight briefly to get past corners/obstacles, not in circles.
            self.move(self.cautious_speed, 0)
            if self.state == 'FOLLOW':
                self.check_lap(now, False)
            return
        self.wall_missing_since = None

        # Right wall closer than target -> positive angle -> LEFT.
        # Right wall farther than target -> negative angle -> gentle RIGHT.
        error = self.active_distance - self.right
        angular = clamp(self.wall_kp * error,
                        -self.max_right_steer, self.max_left_steer)
        speed = self.cautious_speed if abs(angular) > 0.14 else self.forward_speed
        self.move(speed, angular)
        if self.state == 'FOLLOW':
            self.check_lap(now, True)

    def stop(self):
        self.move(0, 0)


def main(args=None):
    rclpy.init(args=args)
    node = WallCoverage()
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
