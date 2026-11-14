"""TurtleBot3 Burger / ROS 2 Jazzy: cautious rolling right-corner recovery.

Heuristic only: scan line segments are not guaranteed to be outer walls.
Uses /scan, /odom and TwistStamped /cmd_vel. Check topic type before use.
"""

import math
import time
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistStamped


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


class WallCoverage(Node):
    def __init__(self):
        super().__init__('wall_coverage')

        # Speeds, in m/s and rad/s.
        self.forward_speed = 0.075
        self.slow_speed = 0.040
        self.avoid_turn_speed = 0.24
        self.corner_turn_speed = 0.20
        self.corner_creep_speed = 0.025  # forward arc only if LiDAR clearance permits
        self.avoid_creep_speed = 0.020   # shallow left arc, only in a clear corridor
        self.corner_forward_clearance = 0.40
        self.corner_front_right_clearance = 0.32
        self.corner_side_clearance = 0.24
        self.front_stop = 0.29
        self.front_resume = 0.37
        self.front_right_stop = 0.23
        self.right_rotation_stop = 0.19

        # Wall controller: TWO proportional gains (no derivative term).
        self.wall_kp = 0.50
        self.heading_kp = 0.35
        self.max_right_steer = 0.22
        self.max_left_steer = 0.18
        self.first_wall_distance = 0.30
        self.wall_acquire_range = 0.70
        self.wall_range_extra = 0.70

        # Line detection: stricter for a NEW wall, looser for tracking.
        self.min_new_wall_length = 0.35
        self.min_track_wall_length = 0.12
        self.min_line_points = 4
        self.max_point_gap = 0.12
        self.max_fit_error = 0.035
        self.max_scan_distance = 2.5
        self.new_confirmations = 5
        self.track_confirmations = 2
        self.acquisition_angle = math.radians(30)
        self.tracking_angle = math.radians(75)
        self.stability_distance = 0.15
        self.stability_angle = math.radians(30)

        # Right-hand corner and escape rules.
        self.corner_grace = 0.5
        self.corner_timeout = 25.0  # allow time to round short/interrupted walls
        self.avoid_timeout = 10.0   # keep individual obstacle-avoidance safety limit
        self.recovery_search_timeout = 15.0  # after corner timeout, search forward gently
        self.recovery_search_speed = 0.030
        self.max_recovery_failures = 2  # stop only after repeated unsuccessful recoveries
        self.recovery_failure_window = 120.0
        self.corner_reacquire_angle = math.radians(35)
        self.corner_confirm_ticks = 3
        self.settle_time = 1.5
        self.settle_max_steer = 0.10
        self.max_corner_attempts = 3  # retained for compatibility; no count-based stop
        self.corner_attempt_window = 30.0
        self.recovery_started = None  # continuous wall-recovery budget across avoidance
        self.wall_stable_since = None
        self.recovery_reset_seconds = 2.0

        # Gradually increase clearance between odometry-detected laps.
        self.inward_step = 0.20
        self.max_wall_distance = 1.56
        self.offset_ramp = 0.004
        self.lap_leave_radius = 0.90
        self.lap_return_radius = 0.40
        self.lap_heading_tolerance = math.radians(40)
        self.min_lap_distance = 3.0
        self.min_lap_seconds = 35.0
        self.min_wall_fraction = 0.65

        # Data and states.
        self.state = 'SEARCH'
        self.resume_after_avoid = 'SEARCH'
        self.front = None
        self.front_right = None
        self.right_near = None
        self.last_scan = None
        self.last_odom = None
        self.pose = None
        self.previous_xy = None
        self.candidate = None
        self.previous_candidate = None
        self.candidate_count = 0
        self.new_wall = None
        self.tracked_wall = None
        self.active_distance = self.first_wall_distance
        self.target_distance = self.first_wall_distance
        self.wall_missing_since = None
        self.corner_started = None
        self.avoid_started = None
        self.reacquire_ticks = 0
        self.last_corner_recovery = None
        self.corner_attempts = deque()  # failed recovery episodes within window
        self.recovery_search_started = None
        self.recent_path = deque(maxlen=250)
        self.lap_count = 0
        self.lap_origin = None
        self.lap_distance = 0.0
        self.lap_start = None
        self.lap_seen = 0
        self.lap_ticks = 0
        self.left_origin = False
        self.last_report = 0.0

        self.create_subscription(LaserScan, '/scan', self.on_scan,
                                 qos_profile_sensor_data)
        self.create_subscription(Odometry, '/odom', self.on_odom,
                                 qos_profile_sensor_data)
        self.cmd_pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.create_timer(0.1, self.control)
        self.get_logger().info('Wall coverage ready: rolling corners, 25s recovery + 15s cautious search.')

    @staticmethod
    def sector(scan, low, high):
        values = []
        clears = 0
        for i, d in enumerate(scan.ranges):
            a = math.degrees(wrap(scan.angle_min + i * scan.angle_increment))
            if not low <= a <= high:
                continue
            if math.isfinite(d) and scan.range_min <= d <= scan.range_max:
                values.append(d)
            elif math.isinf(d) and d > 0:
                clears += 1
        if values:
            return min(values)
        return math.inf if clears >= 3 else None

    @staticmethod
    def fit_line(points):
        n = len(points)
        mx = sum(p[0] for p in points) / n
        my = sum(p[1] for p in points) / n
        xx = sum((x - mx) ** 2 for x, y in points) / n
        yy = sum((y - my) ** 2 for x, y in points) / n
        xy = sum((x - mx) * (y - my) for x, y in points) / n
        a = 0.5 * math.atan2(2 * xy, xx - yy)
        tx, ty = math.cos(a), math.sin(a)
        if tx < 0:
            tx, ty = -tx, -ty
        angle = math.atan2(ty, tx)
        along = [(x - mx) * tx + (y - my) * ty for x, y in points]
        residuals = [-(x - mx) * ty + (y - my) * tx for x, y in points]
        length = max(along) - min(along)
        error = math.sqrt(sum(e * e for e in residuals) / n)
        distance = abs(-mx * ty + my * tx)
        return distance, angle, length, error

    def detect_wall(self, scan):
        clusters, cluster, previous = [], [], None
        for i, d in enumerate(scan.ranges):
            a = wrap(scan.angle_min + i * scan.angle_increment)
            degrees = math.degrees(a)
            if not -145 <= degrees <= -25:
                continue
            valid = (math.isfinite(d) and
                     scan.range_min <= d <= min(scan.range_max, self.max_scan_distance))
            p = (d * math.cos(a), d * math.sin(a)) if valid else None
            if p is None or (previous is not None and
                             math.dist(p, previous) > self.max_point_gap):
                if cluster:
                    clusters.append(cluster)
                cluster = []
            if p is not None:
                cluster.append(p)
            previous = p
        if cluster:
            clusters.append(cluster)

        candidates = []
        for points in clusters:
            if len(points) < self.min_line_points:
                continue
            d, a, length, error = self.fit_line(points)
            if (length >= self.min_track_wall_length and
                    error <= self.max_fit_error and
                    abs(a) <= self.tracking_angle):
                candidates.append((d, a, length, error))
        if not candidates:
            return None
        if self.state != 'SEARCH' and self.previous_candidate is not None:
            old = self.previous_candidate
            candidates.sort(key=lambda c: abs(c[0] - old[0]) +
                            0.4 * abs(wrap(c[1] - old[1])))
        else:
            candidates.sort(key=lambda c: (c[0] + 0.2 * abs(c[1]), -c[2]))
        return candidates[0]

    def on_scan(self, scan):
        self.front = self.sector(scan, -18, 18)
        self.front_right = self.sector(scan, -55, -20)
        self.right_near = self.sector(scan, -125, -55)
        found = self.detect_wall(scan)
        self.candidate = found
        if found is None:
            self.candidate_count = 0
            self.previous_candidate = None
            self.tracked_wall = None
            self.new_wall = None
        else:
            old = self.previous_candidate
            stable = (old is not None and
                      abs(found[0] - old[0]) <= self.stability_distance and
                      abs(wrap(found[1] - old[1])) <= self.stability_angle)
            self.candidate_count = self.candidate_count + 1 if stable else 1
            self.previous_candidate = found
            self.tracked_wall = found if self.candidate_count >= self.track_confirmations else None
            self.new_wall = found if (
                self.candidate_count >= self.new_confirmations and
                found[2] >= self.min_new_wall_length and
                found[0] <= self.wall_acquire_range and
                abs(found[1]) <= self.acquisition_angle
            ) else None
        self.last_scan = time.monotonic()

    def on_odom(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y),
                         1 - 2 * (q.y * q.y + q.z * q.z))
        self.pose = (p.x, p.y, yaw)
        self.last_odom = time.monotonic()
        if self.previous_xy is not None and self.lap_origin is not None:
            delta = math.hypot(p.x - self.previous_xy[0], p.y - self.previous_xy[1])
            if delta < 0.20:
                self.lap_distance += delta
        self.previous_xy = (p.x, p.y)

    def move(self, linear, angular):
        cmd = TwistStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = 'base_link'
        cmd.twist.linear.x = float(linear)
        cmd.twist.angular.z = float(angular)
        self.cmd_pub.publish(cmd)

    def finish(self, reason):
        self.state = 'DONE'
        self.get_logger().warning(reason)
        self.move(0, 0)

    def begin_lap(self, now):
        self.lap_origin = self.pose
        self.lap_distance = 0.0
        self.lap_start = now
        self.lap_seen = 0
        self.lap_ticks = 0
        self.left_origin = False
        self.get_logger().info(f'Lap {self.lap_count + 1}: target gap {self.target_distance:.2f}m')

    def check_lap(self, now, seen):
        if self.lap_origin is None:
            self.begin_lap(now)
            return
        self.lap_ticks += 1
        self.lap_seen += int(seen)
        x, y, yaw = self.pose
        ox, oy, oyaw = self.lap_origin
        separation = math.hypot(x - ox, y - oy)
        if separation >= self.lap_leave_radius:
            self.left_origin = True
        fraction = self.lap_seen / max(1, self.lap_ticks)
        if not (self.left_origin and
                self.lap_distance >= self.min_lap_distance and
                now - self.lap_start >= self.min_lap_seconds and
                fraction >= self.min_wall_fraction and seen and
                separation <= self.lap_return_radius and
                abs(wrap(yaw - oyaw)) <= self.lap_heading_tolerance):
            return
        self.lap_count += 1
        target = self.first_wall_distance + self.lap_count * self.inward_step
        if target > self.max_wall_distance + 1e-6:
            self.finish('Reached maximum clearance; coverage unverified.')
            return
        self.target_distance = target
        self.lap_origin = None
        self.state = 'SHIFT_INWARD'
        self.get_logger().info(f'Lap complete: increasing gap to {target:.2f}m')

    def small_loop(self):
        if self.state not in ('FOLLOW', 'SHIFT_INWARD'):
            self.recent_path.clear()
            return False
        self.recent_path.append(self.pose)
        if len(self.recent_path) < self.recent_path.maxlen:
            return False
        pts = list(self.recent_path)
        dx = max(p[0] for p in pts) - min(p[0] for p in pts)
        dy = max(p[1] for p in pts) - min(p[1] for p in pts)
        travel = sum(math.hypot(b[0] - a[0], b[1] - a[1])
                     for a, b in zip(pts, pts[1:]))
        return (dx < 1.40 and dy < 1.40 and travel >= 1.20 and
                math.hypot(pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1]) < 0.35 and
                abs(wrap(pts[-1][2] - pts[0][2])) < math.radians(35))

    def start_avoid(self, now):
        self.resume_after_avoid = self.state
        self.state = 'AVOID_LEFT'
        self.avoid_started = now
        self.get_logger().info('AVOID_LEFT: front obstacle detected')
        self.move(0, self.avoid_turn_speed)

    def start_corner(self, now):
        # One continuous recovery episode. Left avoidance does not reset its clock,
        # preventing repeated corner/avoid transitions from continuing forever.
        if self.recovery_started is None:
            self.recovery_started = now
        self.state = 'CORNER_RIGHT'
        self.corner_started = now
        self.reacquire_ticks = 0
        self.wall_stable_since = None
        self.recent_path.clear()
        self.get_logger().info('CORNER_RIGHT: wall lost, trying slow right arc')
        self.corner_motion()

    def begin_recovery_search(self, now):
        """Do not stop just because a corner could not be matched in time.

        Travel slowly in the current heading while looking for a confirmed
        right-side wall. If the way is blocked, AVOID_LEFT has priority.
        """
        self.state = 'RECOVER_SEARCH'
        self.recovery_search_started = now
        self.corner_started = None
        self.wall_missing_since = None
        self.reacquire_ticks = 0
        self.get_logger().warning(
            'Wall still missing: slow forward search instead of stopping')
        self.move(0, 0)

    def recovery_failed(self, now):
        """Bound repeated unsuccessful recovery rather than circling forever."""
        self.corner_attempts.append(now)
        while (self.corner_attempts and
               now - self.corner_attempts[0] > self.recovery_failure_window):
            self.corner_attempts.popleft()
        if len(self.corner_attempts) >= self.max_recovery_failures:
            self.finish('STOP: repeated wall recovery failures; check the environment')
            return
        # One bounded second chance, starting with a slow forward search.
        self.recovery_started = None
        self.begin_recovery_search(now)

    def corner_motion(self):
        # A right arc is allowed only with generous FRONT, FRONT-RIGHT and
        # RIGHT clearances. Otherwise rotate in place, or let safety stop us.
        safe_arc = (self.front >= self.corner_forward_clearance and
                    self.front_right >= self.corner_front_right_clearance and
                    self.right_near is not None and
                    self.right_near >= self.corner_side_clearance)
        if safe_arc:
            self.move(self.corner_creep_speed, -self.corner_turn_speed)
        else:
            self.move(0, -self.corner_turn_speed)

    def control(self):
        if self.state == 'DONE':
            self.move(0, 0)
            return
        now = time.monotonic()
        if (self.pose is None or self.last_scan is None or self.last_odom is None or
                now - self.last_scan > 0.7 or now - self.last_odom > 0.7 or
                self.front is None or self.front_right is None):
            self.move(0, 0)
            return

        front_close = (self.front < self.front_stop or
                       self.front_right < self.front_right_stop)
        wall = self.tracked_wall
        wall_seen = (wall is not None and wall[0] <=
                     max(self.wall_acquire_range, self.active_distance + self.wall_range_extra))

        # Recovery can span AVOID_LEFT; a timeout does not immediately stop.
        # A frontal obstacle is still handled by the avoidance state.
        if (self.recovery_started is not None and
                now - self.recovery_started >= self.corner_timeout and
                self.state not in ('RECOVER_SEARCH', 'SEARCH')):
            if self.state == 'AVOID_LEFT':
                # Never override an active collision-avoidance turn.
                pass
            else:
                self.recovery_failed(now)
                return

        if self.state == 'SEARCH':
            if front_close:
                self.start_avoid(now)
                return
            if self.new_wall is None:
                self.move(self.forward_speed, 0)
                return
            self.state = 'FOLLOW'
            self.target_distance = self.first_wall_distance
            self.active_distance = self.first_wall_distance
            self.wall_missing_since = None
            self.recovery_started = None
            self.recent_path.clear()
            self.begin_lap(now)
            self.get_logger().info('FOLLOW: initial wall confirmed')

        if self.state == 'AVOID_LEFT':
            if now - self.avoid_started >= self.avoid_timeout:
                self.finish('STOP: left avoidance exceeded 10 seconds')
                return
            blocked = (self.front < self.front_resume or
                       self.front_right < self.front_right_stop + 0.05)
            if blocked:
                # If partly clear, creep while turning LEFT. With a truly
                # close obstacle, rotate on the spot instead of driving in.
                if (self.front >= 0.35 and
                        self.front_right >= 0.32 and
                        self.right_near is not None and
                        self.right_near >= self.right_rotation_stop):
                    self.move(self.avoid_creep_speed, self.avoid_turn_speed)
                else:
                    self.move(0, self.avoid_turn_speed)
                return
            prior = self.resume_after_avoid
            # A brief front obstruction during a right turn is not a new
            # corner attempt. Resume recovery with the ORIGINAL timeout.
            self.state = prior
            self.avoid_started = None
            if prior != 'CORNER_RIGHT':
                self.wall_missing_since = now
            self.move(0, 0)
            return

        if self.state == 'RECOVER_SEARCH':
            # Search for any sufficiently confirmed right wall, including
            # short sections. Do not resume following until it is oriented
            # reasonably close to the robot's heading.
            reacquired = (wall_seen and
                          abs(wall[1]) <= self.corner_reacquire_angle)
            self.reacquire_ticks = self.reacquire_ticks + 1 if reacquired else 0
            if self.reacquire_ticks >= self.corner_confirm_ticks:
                self.state = 'FOLLOW'
                self.last_corner_recovery = now
                self.wall_missing_since = None
                self.wall_stable_since = now
                self.recovery_search_started = None
                self.reacquire_ticks = 0
                self.get_logger().info('FOLLOW: wall found during slow search')
                self.move(0, 0)
                return
            if front_close:
                self.start_avoid(now)
                return
            if (self.recovery_search_started is not None and
                    now - self.recovery_search_started >= self.recovery_search_timeout):
                self.recovery_failed(now)
                return
            # Forward only if front/front-right are sufficiently clear.
            if (self.front >= self.corner_forward_clearance and
                    self.front_right >= self.corner_front_right_clearance):
                self.move(self.recovery_search_speed, 0)
            else:
                self.move(0, 0)
            return

        if self.state == 'CORNER_RIGHT':
            if (self.recovery_started is not None and
                    now - self.recovery_started >= self.corner_timeout):
                self.recovery_failed(now)
                return
            # A frontal obstacle cannot be driven through. Turn LEFT in place.
            if front_close:
                self.start_avoid(now)
                return
            # Clockwise rotation can sweep the robot into a close right-side
            # object. Stop instead of moving blindly into that object.
            if (self.front_right < self.right_rotation_stop or
                    (self.right_near is not None and
                     self.right_near < self.right_rotation_stop)):
                self.finish('STOP: unsafe right-side clearance during recovery')
                return
            recovered = (wall_seen and
                         abs(wall[1]) <= self.corner_reacquire_angle)
            self.reacquire_ticks = self.reacquire_ticks + 1 if recovered else 0
            if self.reacquire_ticks >= self.corner_confirm_ticks:
                self.state = 'FOLLOW'
                self.wall_missing_since = None
                self.last_corner_recovery = now
                self.wall_stable_since = now
                self.corner_started = None
                self.reacquire_ticks = 0
                self.get_logger().info('FOLLOW: right corner recovered; steering stabilized')
                self.move(0, 0)
                return
            self.corner_motion()
            return

        if front_close:
            self.start_avoid(now)
            return
        if self.small_loop():
            self.finish('STOP: repeating small loop')
            return

        if self.state == 'SHIFT_INWARD':
            self.active_distance = min(self.target_distance,
                                       self.active_distance + self.offset_ramp)
            if self.active_distance >= self.target_distance - 1e-6:
                self.state = 'FOLLOW'
                self.begin_lap(now)

        if not wall_seen:
            self.wall_stable_since = None
            if self.wall_missing_since is None:
                self.wall_missing_since = now
            if now - self.wall_missing_since < self.corner_grace:
                self.move(self.slow_speed, 0)
                if self.state == 'FOLLOW':
                    self.check_lap(now, False)
                return
            self.start_corner(now)
            return
        self.wall_missing_since = None
        if self.recovery_started is not None:
            if self.wall_stable_since is None:
                self.wall_stable_since = now
            elif now - self.wall_stable_since >= self.recovery_reset_seconds:
                self.recovery_started = None
                self.wall_stable_since = None
                self.get_logger().info('Wall stable for 2s; recovery timer reset')

        # Controller: signed line heading + distance correction.
        distance_error = self.active_distance - wall[0]
        heading_error = wall[1]
        angular = clamp(self.wall_kp * distance_error +
                        self.heading_kp * heading_error,
                        -self.max_right_steer, self.max_left_steer)
        settling = (self.last_corner_recovery is not None and
                    now - self.last_corner_recovery < self.settle_time)
        if settling:
            angular = clamp(angular, -self.settle_max_steer, self.settle_max_steer)
        speed = self.slow_speed if settling or angular < -0.07 or angular > 0.14 else self.forward_speed
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
