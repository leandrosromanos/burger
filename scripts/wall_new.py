
import math
import time
from collections import deque

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

        # =====================================
        # MOVEMENT PARAMETERS
        # =====================================
        self.forward_speed = 0.075
        self.cautious_speed = 0.045
        self.left_turn_speed = 0.28

        self.front_stop = 0.29
        self.front_resume = 0.37
        self.front_right_stop = 0.23

        # =====================================
        # WALL FOLLOWING
        # =====================================
        self.first_wall_distance = 0.36
        self.wall_acquire_range = 0.70

        self.inward_step = 0.20
        self.max_wall_distance = 1.56
        self.offset_ramp = 0.004

        self.wall_kp = 0.9
        self.heading_kp = 0.65

        self.max_right_steer = 0.10
        self.max_left_steer = 0.20

        self.wall_lost_timeout = 15.0

        # =====================================
        # NEW: WALL CLASSIFICATION
        # =====================================

        # Inspect the right-side LiDAR sector.
        self.wall_sector_min = -125
        self.wall_sector_max = -55

        # Reject detected straight segments shorter
        # than 35 cm.
        self.min_wall_length = 0.35

        # Maximum RMS line-fitting error.
        self.max_line_error = 0.035

        # Maximum separation between consecutive
        # points belonging to one segment.
        self.max_point_gap = 0.12

        # Minimum LiDAR points in a segment.
        self.min_wall_points = 8

        # Required consecutive compatible scans.
        self.confirmation_scans = 5

        # Line orientation limits.
        self.acquire_angle_limit = math.radians(30)
        self.follow_angle_limit = math.radians(45)

        # Maximum changes between consecutive scans
        # when confirming the same candidate.
        self.stability_distance = 0.12
        self.stability_angle = math.radians(20)

        # Detect segments within this range.
        self.max_detection_range = 2.5

        # =====================================
        # SAFETY PARAMETERS
        # =====================================
        self.scan_timeout = 0.7
        self.odom_timeout = 0.7
        self.max_avoid_seconds = 16.0

        # Small-loop watchdog: approximately
        # 25 seconds at the 10 Hz control rate.
        self.loop_window_ticks = 250
        self.loop_max_span = 1.40
        self.loop_return_radius = 0.35
        self.loop_min_travel = 1.20

        # =====================================
        # LAP DETECTION
        # =====================================
        self.lap_leave_radius = 0.90
        self.lap_return_radius = 0.40
        self.lap_heading_tolerance = math.radians(40)

        self.min_lap_distance = 3.0
        self.min_lap_seconds = 35.0
        self.min_wall_fraction = 0.65

        # =====================================
        # SENSOR DATA
        # =====================================
        self.front = None
        self.front_right = None

        self.last_scan = None
        self.last_odom = None

        self.pose = None
        self.previous_xy = None

        # Fitted wall:
        # (perpendicular_distance, angle,
        #  segment_length, fitting_error)
        self.wall_candidate = None
        self.previous_candidate = None

        self.wall_confirmations = 0
        self.confirmed_wall = None

        # =====================================
        # STATE MACHINE
        # =====================================
        self.state = 'SEARCH'
        self.resume_after_avoid = 'SEARCH'

        self.avoid_started = None
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

        # Recent trajectory for loop detection.
        self.recent_path = deque(
            maxlen=self.loop_window_ticks
        )

        # =====================================
        # ROS INTERFACES
        # =====================================
        self.create_subscription(
            LaserScan,
            '/scan',
            self.on_scan,
            qos_profile_sensor_data
        )

        self.create_subscription(
            Odometry,
            '/odom',
            self.on_odom,
            qos_profile_sensor_data
        )

        # Check the actual /cmd_vel type.
        # This version assumes TwistStamped.
        self.publisher = self.create_publisher(
            TwistStamped,
            '/cmd_vel',
            10
        )

        self.create_timer(0.1, self.control)

        self.get_logger().info(
            'Started: geometric wall validation enabled.'
        )

    # =========================================
    # BASIC LIDAR SECTOR
    # =========================================
    @staticmethod
    def sector(scan, low, high):

        valid = []
        clear = 0

        for i, d in enumerate(scan.ranges):

            angle = math.degrees(
                scan.angle_min +
                i * scan.angle_increment
            )

            angle = (angle + 180) % 360 - 180

            if not low <= angle <= high:
                continue

            if (
                math.isfinite(d)
                and scan.range_min <= d <= scan.range_max
            ):
                valid.append(d)

            elif math.isinf(d) and d > 0:
                clear += 1

        if valid:
            return min(valid)

        if clear >= 3:
            return math.inf

        return None

    # =========================================
    # NEW: FIT STRAIGHT LINE TO POINTS
    # =========================================
    @staticmethod
    def fit_line(points):

        n = len(points)

        if n < 2:
            return None

        mx = sum(p[0] for p in points) / n
        my = sum(p[1] for p in points) / n

        sxx = sum(
            (x - mx) ** 2 for x, y in points
        ) / n

        syy = sum(
            (y - my) ** 2 for x, y in points
        ) / n

        sxy = sum(
            (x - mx) * (y - my)
            for x, y in points
        ) / n

        # Principal direction of the points.
        angle = 0.5 * math.atan2(
            2 * sxy,
            sxx - syy
        )

        tx = math.cos(angle)
        ty = math.sin(angle)

        # Orient tangent toward the robot's
        # positive x direction.
        if tx < 0:
            tx = -tx
            ty = -ty

        heading = math.atan2(ty, tx)

        projections = []
        errors = []

        for x, y in points:

            dx = x - mx
            dy = y - my

            along = dx * tx + dy * ty

            perpendicular = -dx * ty + dy * tx

            projections.append(along)
            errors.append(perpendicular ** 2)

        length = (
            max(projections) - min(projections)
        )

        rms_error = math.sqrt(
            sum(errors) / n
        )

        # Distance from the robot origin to
        # the infinite fitted line.
        distance = abs(
            -mx * ty + my * tx
        )

        return (
            distance,
            heading,
            length,
            rms_error
        )

    # =========================================
    # NEW: DETECT RIGHT-WALL CANDIDATES
    # =========================================
    def detect_right_wall(self, scan):

        segments = []
        current = []
        previous = None

        measurements = []

        for i, d in enumerate(scan.ranges):

            angle = wrap_angle(
                scan.angle_min +
                i * scan.angle_increment
            )

            degrees = math.degrees(angle)

            if not (
                self.wall_sector_min
                <= degrees
                <= self.wall_sector_max
            ):
                continue

            if (
                not math.isfinite(d)
                or d < scan.range_min
                or d > min(
                    scan.range_max,
                    self.max_detection_range
                )
            ):
                measurements.append(None)
                continue

            x = d * math.cos(angle)
            y = d * math.sin(angle)

            measurements.append(
                (degrees, x, y)
            )

        # Sort by angle in case a scan crosses
        # the angular wrap boundary.
        measurements = sorted(
            measurements,
            key=lambda p: (
                p is None,
                p[0] if p is not None else 0
            )
        )

        # Use actual angular order for valid
        # points. Invalid rays separate segments.
        # Rebuild the original ordered sequence.
        measurements = []

        for i, d in enumerate(scan.ranges):

            angle = wrap_angle(
                scan.angle_min +
                i * scan.angle_increment
            )

            degrees = math.degrees(angle)

            if not (
                self.wall_sector_min
                <= degrees
                <= self.wall_sector_max
            ):
                continue

            if (
                not math.isfinite(d)
                or d < scan.range_min
                or d > min(
                    scan.range_max,
                    self.max_detection_range
                )
            ):
                measurements.append(None)
            else:
                measurements.append(
                    (
                        d * math.cos(angle),
                        d * math.sin(angle)
                    )
                )

        for point in measurements:

            if point is None:

                if current:
                    segments.append(current)

                current = []
                previous = None
                continue

            if previous is not None:

                gap = math.hypot(
                    point[0] - previous[0],
                    point[1] - previous[1]
                )

                if gap > self.max_point_gap:

                    if current:
                        segments.append(current)

                    current = []

            current.append(point)
            previous = point

        if current:
            segments.append(current)

        candidates = []

        for points in segments:

            if len(points) < self.min_wall_points:
                continue

            result = self.fit_line(points)

            if result is None:
                continue

            distance, heading, length, error = result

            if length < self.min_wall_length:
                continue

            if error > self.max_line_error:
                continue

            if abs(heading) > self.follow_angle_limit:
                continue

            if distance > self.max_detection_range:
                continue

            candidates.append(result)

        if not candidates:
            return None

        # Prefer nearer, more parallel segments.
        candidates.sort(
            key=lambda c: (
                c[0] + 0.20 * abs(c[1]),
                -c[2]
            )
        )

        return candidates[0]

    # =========================================
    # NEW: CONFIRM WALL OVER MULTIPLE SCANS
    # =========================================
    def update_wall_confirmation(self, candidate):

        self.wall_candidate = candidate

        if candidate is None:

            self.wall_confirmations = 0
            self.previous_candidate = None
            self.confirmed_wall = None
            return

        if self.previous_candidate is None:

            self.wall_confirmations = 1

        else:

            old_distance = self.previous_candidate[0]
            old_angle = self.previous_candidate[1]

            new_distance = candidate[0]
            new_angle = candidate[1]

            stable_distance = (
                abs(new_distance - old_distance)
                <= self.stability_distance
            )

            stable_angle = (
                abs(wrap_angle(new_angle - old_angle))
                <= self.stability_angle
            )

            if stable_distance and stable_angle:

                self.wall_confirmations = min(
                    self.confirmation_scans,
                    self.wall_confirmations + 1
                )

            else:
                self.wall_confirmations = 1

        self.previous_candidate = candidate

        if (
            self.wall_confirmations
            >= self.confirmation_scans
        ):
            self.confirmed_wall = candidate
        else:
            self.confirmed_wall = None

    # =========================================
    # SENSOR CALLBACKS
    # =========================================
    def on_scan(self, scan):

        self.front = self.sector(
            scan, -18, 18
        )

        self.front_right = self.sector(
            scan, -55, -20
        )

        candidate = self.detect_right_wall(scan)

        self.update_wall_confirmation(candidate)

        self.last_scan = time.monotonic()

    def on_odom(self, msg):

        p = msg.pose.pose.position
        q = msg.pose.pose.orientation

        yaw = math.atan2(
            2 * (q.w * q.z + q.x * q.y),
            1 - 2 * (q.y * q.y + q.z * q.z)
        )

        self.pose = (p.x, p.y, yaw)
        self.last_odom = time.monotonic()

        if (
            self.previous_xy is not None
            and self.lap_origin is not None
        ):

            delta = math.hypot(
                p.x - self.previous_xy[0],
                p.y - self.previous_xy[1]
            )

            if delta < 0.20:
                self.lap_distance += delta

        self.previous_xy = (p.x, p.y)

    # =========================================
    # VELOCITY COMMAND
    # =========================================
    def move(self, linear, angular):

        cmd = TwistStamped()

        cmd.header.stamp = (
            self.get_clock().now().to_msg()
        )

        cmd.header.frame_id = 'base_link'

        cmd.twist.linear.x = float(linear)
        cmd.twist.angular.z = float(angular)

        self.publisher.publish(cmd)

    # =========================================
    # LAP MANAGEMENT
    # =========================================
    def begin_lap(self, now):

        self.lap_origin = self.pose
        self.lap_distance = 0.0
        self.lap_start_time = now

        self.lap_seen_ticks = 0
        self.lap_total_ticks = 0

        self.left_lap_origin = False

        self.get_logger().info(
            f'Lap {self.lap_count + 1} started; '
            f'gap {self.target_distance:.2f} m'
        )

    def check_lap(self, now, wall_seen):

        if self.lap_origin is None:
            self.begin_lap(now)
            return

        self.lap_total_ticks += 1

        if wall_seen:
            self.lap_seen_ticks += 1

        x, y, yaw = self.pose
        x0, y0, yaw0 = self.lap_origin

        separation = math.hypot(
            x - x0,
            y - y0
        )

        if separation >= self.lap_leave_radius:
            self.left_lap_origin = True

        fraction = (
            self.lap_seen_ticks /
            max(self.lap_total_ticks, 1)
        )

        completed = (
            self.left_lap_origin
            and self.lap_distance >= self.min_lap_distance
            and now - self.lap_start_time
                >= self.min_lap_seconds
            and fraction >= self.min_wall_fraction
            and wall_seen
            and separation <= self.lap_return_radius
            and abs(wrap_angle(yaw - yaw0))
                <= self.lap_heading_tolerance
        )

        if not completed:
            return

        self.lap_count += 1

        next_target = (
            self.first_wall_distance +
            self.lap_count * self.inward_step
        )

        self.get_logger().info(
            f'Completed lap {self.lap_count}'
        )

        if (
            next_target >
            self.max_wall_distance + 1e-6
        ):

            self.state = 'DONE'
            self.move(0, 0)
            return

        self.target_distance = next_target
        self.state = 'SHIFT_INWARD'

        self.lap_origin = None

    # =========================================
    # NEW: SMALL-LOOP DETECTOR
    # =========================================
    def check_small_loop(self):

        if self.state not in (
            'FOLLOW', 'SHIFT_INWARD'
        ):
            self.recent_path.clear()
            return False

        self.recent_path.append(self.pose)

        if (
            len(self.recent_path)
            < self.loop_window_ticks
        ):
            return False

        path = list(self.recent_path)

        xs = [p[0] for p in path]
        ys = [p[1] for p in path]

        span_x = max(xs) - min(xs)
        span_y = max(ys) - min(ys)

        travel = sum(
            math.hypot(
                b[0] - a[0],
                b[1] - a[1]
            )
            for a, b in zip(path, path[1:])
        )

        start = path[0]
        end = path[-1]

        separation = math.hypot(
            end[0] - start[0],
            end[1] - start[1]
        )

        heading_error = abs(
            wrap_angle(end[2] - start[2])
        )

        small_area = (
            span_x < self.loop_max_span
            and span_y < self.loop_max_span
        )

        returned = (
            separation < self.loop_return_radius
            and heading_error < math.radians(35)
        )

        if (
            small_area
            and returned
            and travel >= self.loop_min_travel
        ):
            self.get_logger().warning(
                'Possible small obstacle loop: STOP'
            )
            return True

        return False

    # =========================================
    # AVOIDANCE HELPERS
    # =========================================
    def start_avoidance(self, now):

        self.resume_after_avoid = self.state
        self.state = 'AVOID_LEFT'

        self.avoid_started = now

        self.move(0, self.left_turn_speed)

    # =========================================
    # MAIN CONTROLLER
    # =========================================
    def control(self):

        if self.state == 'DONE':
            self.move(0, 0)
            return

        now = time.monotonic()

        # Stop when sensors are stale.
        if (
            self.last_scan is None
            or self.last_odom is None
            or self.pose is None
            or now - self.last_scan > self.scan_timeout
            or now - self.last_odom > self.odom_timeout
            or self.front is None
        ):
            self.move(0, 0)
            return

        front_close = (
            self.front < self.front_stop
            or (
                self.front_right is not None
                and self.front_right
                    < self.front_right_stop
            )
        )

        wall = self.confirmed_wall

        wall_acquirable = (
            wall is not None
            and wall[0] <= self.wall_acquire_range
            and abs(wall[1])
                <= self.acquire_angle_limit
        )

        wall_seen = (
            wall is not None
            and wall[0] <= max(
                self.wall_acquire_range,
                self.active_distance + 0.70
            )
            and abs(wall[1])
                <= self.follow_angle_limit
        )

        # =====================================
        # SEARCH: MOVE STRAIGHT
        # =====================================
        if self.state == 'SEARCH':

            if front_close:

                self.start_avoidance(now)
                return

            if wall_acquirable:

                self.state = 'FOLLOW'

                self.wall_missing_since = None

                self.target_distance = (
                    self.first_wall_distance
                )

                self.active_distance = (
                    self.first_wall_distance
                )

                self.recent_path.clear()

                self.begin_lap(now)

                self.get_logger().info(
                    'Confirmed right wall: '
                    f'distance={wall[0]:.2f} m, '
                    f'length={wall[2]:.2f} m, '
                    f'angle={math.degrees(wall[1]):.1f} deg'
                )

            else:

                self.move(
                    self.forward_speed, 0
                )
                return

        # =====================================
        # AVOID LEFT
        # =====================================
        if self.state == 'AVOID_LEFT':

            if (
                self.avoid_started is not None
                and now - self.avoid_started
                    > self.max_avoid_seconds
            ):

                self.get_logger().warning(
                    'Avoidance timeout: STOP'
                )

                self.state = 'DONE'
                self.move(0, 0)
                return

            still_blocked = (
                self.front < self.front_resume
                or (
                    self.front_right is not None
                    and self.front_right
                        < self.front_right_stop + 0.05
                )
            )

            if still_blocked:

                self.move(
                    0, self.left_turn_speed
                )
                return

            self.state = self.resume_after_avoid
            self.avoid_started = None

            self.move(
                self.cautious_speed, 0
            )
            return

        # =====================================
        # FOLLOW SAFETY
        # =====================================
        if front_close:

            self.start_avoidance(now)
            return

        if self.check_small_loop():

            self.state = 'DONE'
            self.move(0, 0)
            return

        # =====================================
        # INCREASE WALL CLEARANCE
        # =====================================
        if self.state == 'SHIFT_INWARD':

            self.active_distance = min(
                self.target_distance,
                self.active_distance +
                self.offset_ramp
            )

            if (
                self.active_distance >=
                self.target_distance - 1e-6
            ):

                self.state = 'FOLLOW'
                self.begin_lap(now)

        # =====================================
        # LOST WALL
        # =====================================
        if not wall_seen:

            if self.wall_missing_since is None:
                self.wall_missing_since = now

            if (
                now - self.wall_missing_since >
                self.wall_lost_timeout
            ):

                self.get_logger().warning(
                    'Confirmed wall lost: STOP'
                )

                self.state = 'DONE'

                self.move(0, 0)
                return

            self.move(
                self.cautious_speed, 0
            )

            if self.state == 'FOLLOW':
                self.check_lap(now, False)

            return

        self.wall_missing_since = None

        # =====================================
        # IMPROVED WALL CONTROLLER
        # =====================================
        distance = wall[0]
        wall_angle = wall[1]

        # Closer than target -> turn left.
        # Farther than target -> turn right.
        distance_error = (
            self.active_distance - distance
        )

        # Steer parallel to fitted wall.
        heading_error = wall_angle

        angular = (
            self.wall_kp * distance_error
            + self.heading_kp * heading_error
        )

        angular = clamp(
            angular,
            -self.max_right_steer,
            self.max_left_steer
        )

        if abs(angular) > 0.14:
            speed = self.cautious_speed
        else:
            speed = self.forward_speed

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
