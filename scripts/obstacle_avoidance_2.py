#!/usr/bin/env python3
"""TurtleBot3 Burger / ROS 2: bounded reactive avoidance, NOT map exploration.

Assumes a horizontal 360-degree LaserScan aligned with the robot (+X forward)
and a scanner close to the robot centre. Test in Gazebo first. The conservative
0.18 m circular envelope includes the body and a margin; adjust for attachments.
Only this node should command cmd_vel. A base-controller command timeout is
still required if this process dies. Positive infinity follows REP-117 by
default; set inf_is_clear:=false for a driver without that convention.
"""
import math
import time
from collections import deque
from dataclasses import dataclass


@dataclass
class View:
    front: float
    rear: float
    nearest: float
    left: float
    right: float


def scan_view(msg, inf_is_clear=True):
    """Conservative 360-degree scan validation and swept-strip clearance.

    Every 5-degree bin must be observed; invalid rays cause a stop. This is
    deliberately strict: unknown space is never silently converted to free.
    Clearance distances are measured from the scanner, not the bumper.
    """
    if (not all(math.isfinite(v) for v in
                (msg.angle_min, msg.angle_increment, msg.range_min, msg.range_max))
            or msg.angle_increment == 0 or msg.range_min < 0
            or msg.range_max <= max(msg.range_min, 0.60)):
        return None
    bins = set()
    front = rear = nearest = left = right = msg.range_max
    for i, value in enumerate(msg.ranges):
        angle = (msg.angle_min + i * msg.angle_increment + math.pi) % (2 * math.pi) - math.pi
        bins.add(int(math.degrees(angle) + 180) // 5)
        if value == -math.inf or (math.isfinite(value) and 0 <= value < msg.range_min):
            distance = 0.0  # too close is danger, not missing data
        elif value == math.inf and inf_is_clear:
            distance = msg.range_max
        elif math.isfinite(value) and msg.range_min <= value <= msg.range_max:
            distance = value
        else:
            return None
        nearest = min(nearest, distance)
        degrees = math.degrees(angle)
        if 40 <= degrees <= 100:
            left = min(left, distance)
        if -100 <= degrees <= -40:
            right = min(right, distance)
        x, y = distance * math.cos(angle), distance * math.sin(angle)
        # Obstacles alongside the travel corridor do not block it merely
        # because they lie inside a broad angular sector.
        if abs(y) <= 0.18:
            if x >= 0:
                front = min(front, x)
            if x <= 0:
                rear = min(rear, -x)
    if len(bins) < 72:
        return None
    return View(front, rear, nearest, left, right)


class Avoidance:
    """Pure controller so recovery logic can be tested without ROS."""
    def __init__(self):
        self.state = 'DRIVE'
        self.reason = 'Starting'
        self.direction = None
        self.anchor = None
        self.history = deque()
        self.started = 0.0
        self.last_yaw = None
        self.rotation = 0.0
        self.clear_since = None
        self.back_start = None
        self.attempts = 0

    def enter(self, state, now, pose, reason):
        self.state, self.started, self.reason = state, now, reason
        self.last_yaw = pose[2]
        self.rotation = 0.0
        self.clear_since = None
        if state == 'BACKUP':
            self.back_start = pose[:2]

    def recover(self, now, pose, view, reason):
        self.attempts += 1
        if self.attempts > 3:
            self.enter('BLOCKED', now, pose, 'Recovery exhausted; inspect and restart')
        elif view.nearest < 0.18:
            self.enter('BLOCKED', now, pose, 'Obstacle inside rotation envelope')
        elif view.rear > 0.40:
            self.enter('BACKUP', now, pose, reason)
        else:
            self.enter('ESCAPE', now, pose, reason + '; rear blocked, rotating only')
        self.history.clear()
        return 0.0, 0.0

    def step(self, now, pose, view):
        if self.state == 'BLOCKED':
            return 0.0, 0.0
        if view.nearest < 0.18:
            self.enter('BLOCKED', now, pose, 'Obstacle too close for safe rotation')
            return 0.0, 0.0
        if self.anchor is None:
            self.anchor = pose[:2]
        # Unlock direction only after real displacement away from the episode.
        if self.state == 'DRIVE' and math.dist(pose[:2], self.anchor) >= 0.50:
            self.anchor = pose[:2]
            self.direction = None
            self.attempts = 0
            self.history.clear()
        self.history.append((now, pose[0], pose[1]))
        while self.history and now - self.history[0][0] > 15.1:
            self.history.popleft()
        if self.state == 'DRIVE' and self.history and now - self.history[0][0] >= 15:
            displacement = max(math.hypot(x-pose[0], y-pose[1]) for _, x, y in self.history)
            if displacement < 0.15:
                return self.recover(now, pose, view, 'No positional progress for 15 seconds')
        if self.state in ('TURN', 'ESCAPE'):
            delta = (pose[2] - self.last_yaw + math.pi) % (2 * math.pi) - math.pi
            self.rotation += abs(delta)
            self.last_yaw = pose[2]
            # Require stable clearance before leaving a turn.
            if view.front >= 0.60:
                if self.clear_since is None:
                    self.clear_since = now
                if now - self.clear_since >= 0.3:
                    self.enter('DRIVE', now, pose, 'Forward corridor clear')
                    return 0.0, 0.0
            else:
                self.clear_since = None
            if now - self.started >= 18 or self.rotation >= 2 * math.pi:
                return self.recover(now, pose, view, 'Turn limit reached')
            return 0.0, 0.4 * (self.direction or 1.0)
        if self.state == 'BACKUP':
            distance = math.dist(pose[:2], self.back_start)
            if distance >= 0.15 or now - self.started >= 4 or view.rear <= 0.30:
                self.enter('ESCAPE', now, pose, 'Backup finished; searching in same direction')
                return 0.0, 0.0
            return -0.04, 0.0
        if view.front < 0.45:
            if self.direction is None:
                # Small differences do not outweigh the default left preference.
                self.direction = -1.0 if view.right > view.left + 0.15 else 1.0
            self.enter('TURN', now, pose, 'Obstacle ahead')
            return 0.0, 0.4 * self.direction
        return 0.06, 0.0


def main(args=None):
    import rclpy
    from rclpy.node import Node
    from rclpy.clock import Clock, ClockType
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.executors import ExternalShutdownException
    from sensor_msgs.msg import LaserScan
    from nav_msgs.msg import Odometry
    from geometry_msgs.msg import TwistStamped

    class ObstacleAvoidance(Node):
        def __init__(self):
            super().__init__('obstacle_avoidance')
            self.declare_parameter('inf_is_clear', True)
            self.core = Avoidance()
            self.view = self.pose = None
            self.scan_time = self.odom_time = None
            self.scan_stamp = self.odom_stamp = None
            self.status = None
            self.interrupted = False
            self.publisher = self.create_publisher(TwistStamped, '/cmd_vel', 10)
            self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, qos_profile_sensor_data)
            self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, qos_profile_sensor_data)
            # Watchdog must run even when Gazebo's simulation clock is paused.
            self.timer = self.create_timer(0.1, self.control_loop,
                                          clock=Clock(clock_type=ClockType.STEADY_TIME))

        @staticmethod
        def stamp(message):
            return message.header.stamp.sec + message.header.stamp.nanosec * 1e-9

        def scan_callback(self, msg):
            self.view = scan_view(msg, self.get_parameter('inf_is_clear').value)
            self.scan_time, self.scan_stamp = time.monotonic(), self.stamp(msg)

        def odom_callback(self, msg):
            p, q = msg.pose.pose.position, msg.pose.pose.orientation
            values = (p.x, p.y, q.x, q.y, q.z, q.w)
            norm = sum(v*v for v in (q.x, q.y, q.z, q.w))
            if not all(math.isfinite(v) for v in values) or abs(norm - 1) > 0.05:
                self.pose = None
                return
            yaw = math.atan2(2*(q.w*q.z + q.x*q.y), 1-2*(q.y*q.y + q.z*q.z))
            self.pose = (p.x, p.y, yaw)
            self.odom_time, self.odom_stamp = time.monotonic(), self.stamp(msg)

        def move(self, linear, angular):
            command = TwistStamped()
            command.header.stamp = self.get_clock().now().to_msg()
            command.header.frame_id = 'base_link'
            command.twist.linear.x = float(linear)
            command.twist.angular.z = float(angular)
            self.publisher.publish(command)

        def report(self, text):
            if text != self.status:
                self.get_logger().info(text)
                self.status = text

        def control_loop(self):
            now = time.monotonic()
            ros_now = self.get_clock().now().nanoseconds * 1e-9
            valid = self.view is not None and self.pose is not None
            for arrival, stamp in ((self.scan_time, self.scan_stamp), (self.odom_time, self.odom_stamp)):
                valid = valid and arrival is not None and stamp is not None
                if arrival is not None and stamp is not None:
                    valid = valid and now-arrival <= 0.5 and -0.1 <= ros_now-stamp <= 0.5
            if not valid:
                self.move(0, 0)
                self.interrupted = True
                self.report('WAIT: missing, invalid or stale scan/odom; check use_sim_time and 360-degree scan')
                return
            if self.interrupted:
                # Never resume an old reverse manoeuvre after a sensor outage.
                if self.core.state != 'BLOCKED':
                    self.core.enter('DRIVE', now, self.pose, 'Fresh sensors restored')
                    self.core.history.clear()
                self.interrupted = False
            linear, angular = self.core.step(now, self.pose, self.view)
            self.move(linear, angular)
            self.report(self.core.state + ': ' + self.core.reason)

    rclpy.init(args=args)
    node = ObstacleAvoidance()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            node.move(0, 0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()