
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


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


class WallCoverage(Node):

    def __init__(self):
        super().__init__('wall_coverage')

        # =====================================
        # ROBOT MOVEMENT
        # =====================================
        self.forward_speed = 0.075
        self.cautious_speed = 0.045
        self.left_turn_speed = 0.28

        self.front_stop = 0.29
        self.front_resume = 0.37
        self.front_right_stop = 0.23

        # =====================================
        # RIGHT WALL FOLLOWING
        # =====================================
        self.first_wall_distance = 0.36
        self.wall_acquire_range = 0.70

        self.inward_step = 0.20
        self.max_wall_distance = 1.56
        self.offset_ramp = 0.004

        self.wall_kp = 0.9
        self.heading_kp = 0.65

        # Increased right turn capability
        self.max_right_steer = 0.24
        self.max_left_steer = 0.20

        # =====================================
        # WALL DETECTION
        # =====================================
        self.wall_sector_min = -145
        self.wall_sector_max = -25

        self.min_wall_length = 0.35
        self.max_line_error = 0.035
        self.max_point_gap = 0.12
        self.min_wall_points = 8

        self.confirmation_scans = 5

        self.acquire_angle_limit = math.radians(30)
        self.follow_angle_limit = math.radians(70)

        self.stability_distance = 0.12
        self.stability_angle = math.radians(20)

        self.max_detection_range = 2.5

        # =====================================
        # RIGHT CORNER RECOVERY
        # =====================================
        self.corner_grace_time = 0.5
        self.corner_turn_speed = 0.20

        # Enough time for approximately
        # a 90-degree right turn.
        self.corner_timeout = 9.0

        self.corner_reacquire_angle = math.radians(40)

        self.corner_started = None
        self.wall_missing_since = None

        # =====================================
        # SAFETY
        # =====================================
        self.scan_timeout = 0.7
        self.odom_timeout = 0.7
        self.max_avoid_seconds = 16.0

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

        self.wall_candidate = None
        self.previous_candidate = None
        self.confirmed_wall = None

        self.wall_confirmations = 0

        # =====================================
        # STATE MACHINE
        # =====================================
        # SEARCH
        # FOLLOW
        # AVOID_LEFT
        # CORNER_RIGHT
        # SHIFT_INWARD
        # DONE

        self.state = 'SEARCH'

        self.resume_after_avoid = 'SEARCH'
        self.avoid_started = None

        self.active_distance = self.first_wall_distance
        self.target_distance = self.first_wall_distance

        # =====================================
        # LAP VARIABLES
        # =====================================
        self.lap_count = 0
        self.lap_origin = None

        self.lap_distance = 0.0
        self.lap_start_time = None

        self.lap_seen_ticks = 0
        self.lap_total_ticks = 0

        self.left_lap_origin = False

        # Store recent robot positions.
        self.recent_path = deque(
            maxlen=self.loop_window_ticks
        )

        # =====================================
        # ROS SUBSCRIPTIONS
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

        # Assumes /cmd_vel uses TwistStamped.
        self.publisher = self.create_publisher(
            TwistStamped,
            '/cmd_vel',
            10
        )

        self.create_timer(0.1, self.control)

        self.get_logger().info(
            'Wall coverage started: '
            'right-corner recovery enabled.'
        )

    # =========================================
    # READ A LIDAR SECTOR
    # =========================================
    @staticmethod
    def sector(scan, low, high):

        valid = []
        clear = 0

        for i, distance in enumerate(scan.ranges):

            angle = math.degrees(
                scan.angle_min +
                i * scan.angle_increment
            )

            angle = (angle + 180) % 360 - 180

            if not low <= angle <= high:
                continue

            if (
                math.isfinite(distance)
                and scan.range_min <= distance <= scan.range_max
            ):
                valid.append(distance)

            elif math.isinf(distance) and distance > 0:
                clear += 1

        if valid:
            return min(valid)

        if clear >= 3:
            return math.inf

        return None

    # =========================================
    # FIT A LINE TO LIDAR POINTS
    # =========================================
    @staticmethod
    def fit_line(points):

        n = len(points)

        if n < 2:
            return None

        mx = sum(x for x, y in points) / n
        my = sum(y for x, y in points) / n

        sxx = sum(
            (x - mx) ** 2
            for x, y in points
        ) / n

        syy = sum(
            (y - my) ** 2
            for x, y in points
        ) / n

        sxy = sum(
            (x - mx) * (y - my)
            for x, y in points
        ) / n

        angle = 0.5 * math.atan2(
            2 * sxy,
            sxx - syy
        )

        tx = math.cos(angle)
        ty = math.sin(angle)

        # Orient line direction forward.
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
            max(projections) -
            min(projections)
        )

        rms_error = math.sqrt(
            sum(errors) / n
        )

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
    # DETECT POSSIBLE RIGHT WALL
    # =========================================
    def detect_right_wall(self, scan):

        segments = []
        current = []
        previous = None

        measurements = []

        for i, distance in enumerate(scan.ranges):

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
                not math.isfinite(distance)
                or distance < scan.range_min
                or distance > min(
                    scan.range_max,
                    self.max_detection_range
                )
            ):
                measurements.append(None)
                continue

            x = distance * math.cos(angle)
            y = distance * math.sin(angle)

            measurements.append((x, y))

        # Divide LiDAR points into continuous
        # clusters using point-to-point gaps.
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

            # Long enough to be a wall candidate?
            if length < self.min_wall_length:
                continue

            # Straight enough?
            if error > self.max_line_error:
                continue

            # Reasonable orientation?
            if abs(heading) > self.follow_angle_limit:
                continue

            if distance > self.max_detection_range:
                continue

            candidates.append(result)

        if not candidates:
            return None

        # Prefer nearby, parallel surfaces.
        candidates.sort(
            key=lambda candidate: (
                candidate[0] +
                0.20 * abs(candidate[1]),
                -candidate[2]
            )
        )

        return candidates[0]

    # =========================================
    # CONFIRM DETECTED WALL
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
    # LIDAR CALLBACK
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

    # =========================================
    # ODOMETRY CALLBACK
    # =========================================
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
    # SEND VELOCITY
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
    # START A LAP
    # =========================================
    def begin_lap(self, now):

        self.lap_origin = self.pose
        self.lap_distance = 0.0

        self.lap_start_time = now

        self.lap_seen_ticks = 0
        self.lap_total_ticks = 0

        self.left_lap_origin = False

        self.get_logger().info(
            f'Lap {self.lap_count + 1} started. '
            f'Target gap: {self.target_distance:.2f} m'
        )

    # =========================================
    # CHECK LAP COMPLETION
    # =========================================
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

            self.get_logger().info(
                'Maximum target distance reached. STOP.'
            )

            self.state = 'DONE'
            self.move(0, 0)
            return

        self.target_distance = next_target

        self.state = 'SHIFT_INWARD'

        self.lap_origin = None

        self.get_logger().info(
            f'Increasing right-wall gap to '
            f'{next_target:.2f} m'
        )

    # =========================================
    # DETECT SMALL REPEATING LOOPS
    # =========================================
    def check_small_loop(self):

        if self.state not in (
            'FOLLOW',
            'SHIFT_INWARD'
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
                'Possible small obstacle loop. STOP.'
            )

            return True

        return False

    # =========================================
    # START LEFT AVOIDANCE
    # =========================================
    def start_avoidance(self, now):

        self.resume_after_avoid = self.state

        self.state = 'AVOID_LEFT'

        self.avoid_started = now

        self.get_logger().info(
            'Obstacle ahead: turning LEFT.'
        )

        self.move(
            0,
            self.left_turn_speed
        )

    # =========================================
    # MAIN CONTROL LOOP
    # =========================================
    def control(self):

        if self.state == 'DONE':
            self.move(0, 0)
            return

        now = time.monotonic()

        # =====================================
        # CHECK SENSOR VALIDITY
        # =====================================
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

        # =====================================
        # FRONT OBSTACLE DETECTION
        # =====================================
        front_close = (
            self.front < self.front_stop
            or (
                self.front_right is not None
                and self.front_right
                    < self.front_right_stop
            )
        )

        wall = self.confirmed_wall

        # A wall suitable for initial following.
        wall_acquirable = (
            wall is not None
            and wall[0] <= self.wall_acquire_range
            and abs(wall[1])
                <= self.acquire_angle_limit
        )

        # A wall suitable for continued tracking.
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
        # STATE 1: SEARCH
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
                    'Confirmed right wall. '
                    f'Distance={wall[0]:.2f} m, '
                    f'Length={wall[2]:.2f} m, '
                    f'Angle={math.degrees(wall[1]):.1f} deg'
                )

            else:

                # No confirmed wall:
                # continue searching straight.
                self.move(
                    self.forward_speed,
                    0
                )
                return

        # =====================================
        # STATE 2: AVOID LEFT
        # =====================================
        if self.state == 'AVOID_LEFT':

            if (
                self.avoid_started is not None
                and now - self.avoid_started
                    > self.max_avoid_seconds
            ):

                self.get_logger().warning(
                    'Avoidance timeout. STOP.'
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
                    0,
                    self.left_turn_speed
                )
                return

            # Return to the previous state.
            self.state = self.resume_after_avoid

            self.avoid_started = None

            self.move(
                self.cautious_speed,
                0
            )
            return

        # =====================================
        # STATE 3: CORNER RIGHT
        # =====================================
        if self.state == 'CORNER_RIGHT':

            # Safety priority.
            if front_close:

                self.start_avoidance(now)
                return

            # Check whether a suitable wall
            # has been recovered.
            recovered = (
                wall_seen
                and abs(wall[1])
                    <= self.corner_reacquire_angle
            )

            if recovered:

                self.state = 'FOLLOW'

                self.wall_missing_since = None
                self.corner_started = None

                self.get_logger().info(
                    'Right wall recovered after corner.'
                )

                self.move(0, 0)
                return

            # Prevent endless right rotation.
            if (
                self.corner_started is not None
                and now - self.corner_started
                    >= self.corner_timeout
            ):

                self.get_logger().warning(
                    'Right corner recovery failed. STOP.'
                )

                self.state = 'DONE'
                self.move(0, 0)
                return

            # Turn RIGHT in place.
            self.move(
                0,
                -self.corner_turn_speed
            )
            return

        # =====================================
        # FRONT SAFETY DURING FOLLOWING
        # =====================================
        if front_close:

            self.start_avoidance(now)
            return

        # =====================================
        # LOOP DETECTION
        # =====================================
        if self.check_small_loop():

            self.state = 'DONE'
            self.move(0, 0)
            return

        # =====================================
        # STATE 4: SHIFT INWARD
        # =====================================
        if self.state == 'SHIFT_INWARD':

            self.active_distance = min(
                self.target_distance,
                self.active_distance
                + self.offset_ramp
            )

            if (
                self.active_distance
                >= self.target_distance - 1e-6
            ):

                self.state = 'FOLLOW'

                self.begin_lap(now)

        # =====================================
        # WALL LOST: RIGHT CORNER RECOVERY
        # =====================================
        if not wall_seen:

            if self.wall_missing_since is None:
                self.wall_missing_since = now

            missing_duration = (
                now - self.wall_missing_since
            )

            # Brief disappearance:
            # move straight slowly.
            if (
                missing_duration
                < self.corner_grace_time
            ):

                self.move(
                    self.cautious_speed,
                    0
                )

                if self.state == 'FOLLOW':
                    self.check_lap(now, False)

                return

            # Wall has disappeared long enough.
            # Attempt a RIGHT turn.
            self.state = 'CORNER_RIGHT'

            self.corner_started = now

            self.recent_path.clear()

            self.get_logger().info(
                'Right wall lost: '
                'attempting RIGHT corner recovery.'
            )

            self.move(
                0,
                -self.corner_turn_speed
            )
            return

        self.wall_missing_since = None

        # =====================================
        # STATE 5: NORMAL WALL FOLLOWING
        # =====================================
        distance = wall[0]
        wall_angle = wall[1]

        # Right wall too close -> steer left.
        # Right wall too far -> steer right.
        distance_error = (
            self.active_distance - distance
        )

        # Align with wall direction.
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

        # Slow down for substantial turns.
        if angular < -0.07:

            speed = self.cautious_speed

        elif angular > 0.14:

            speed = self.cautious_speed

        else:

            speed = self.forward_speed

        self.move(
            speed,
            angular
        )

        if self.state == 'FOLLOW':
            self.check_lap(now, True)

    # =========================================
    # STOP ROBOT
    # =========================================
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
