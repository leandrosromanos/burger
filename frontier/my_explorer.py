#!/usr/bin/env python3
"""Frontier exploration with Python A*, waypoint control and LiDAR stop checks.

ROS 2 Jazzy / TurtleBot3 Burger simulation. Publishes TwistStamped directly.
Use only one cmd_vel controller. A base-controller command timeout is required
if this process or its host stops. No Nav2 dependency. This is a conservative
baseline, not a guarantee that every unknown region is physically observable.
"""
import heapq
import math
import signal
import time
from collections import deque

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import OccupancyGrid, Path
from sensor_msgs.msg import LaserScan
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from rclpy.time import Time
from scipy.ndimage import distance_transform_edt, label
from tf2_ros import Buffer, TransformException, TransformListener


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_of(q):
    return math.atan2(2 * (q.w*q.z + q.x*q.y), 1 - 2 * (q.y*q.y + q.z*q.z))


def stamp_seconds(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def cell_to_world(cell, info):
    x, y = (cell[0] + .5) * info.resolution, (cell[1] + .5) * info.resolution
    yaw = yaw_of(info.origin.orientation)
    c, s = math.cos(yaw), math.sin(yaw)
    return info.origin.position.x + c*x - s*y, info.origin.position.y + s*x + c*y


def world_to_cell(point, info):
    dx, dy = point[0] - info.origin.position.x, point[1] - info.origin.position.y
    yaw = yaw_of(info.origin.orientation)
    c, s = math.cos(yaw), math.sin(yaw)
    return (math.floor((c*dx + s*dy) / info.resolution),
            math.floor((-s*dx + c*dy) / info.resolution))


def clearance_grid(free, resolution):
    padded = np.pad(free, 1, constant_values=False)
    distances = distance_transform_edt(padded, sampling=resolution)[1:-1, 1:-1]
    return np.maximum(distances - resolution * math.sqrt(2) / 2, 0)


def frontier_groups(grid, threshold):
    free = (grid >= 0) & (grid <= threshold)
    unknown = grid == -1
    touches = np.zeros_like(free)
    touches[1:, :] |= unknown[:-1, :]
    touches[:-1, :] |= unknown[1:, :]
    touches[:, 1:] |= unknown[:, :-1]
    touches[:, :-1] |= unknown[:, 1:]
    labels, count = label(free & touches, structure=np.ones((3, 3)))
    return [list(zip(*np.where(labels == i)[::-1])) for i in range(1, count + 1)]


def astar(walkable, start, goal, clearance=None, required_clearance=.17):
    """Four-neighbor path; optionally penalize travel near obstacle boundaries."""
    height, width = walkable.shape
    for x, y in (start, goal):
        if not (0 <= x < width and 0 <= y < height) or not walkable[y, x]:
            return []
    def h(cell):
        return abs(cell[0]-goal[0]) + abs(cell[1]-goal[1])
    queue = [(h(start), 0, start)]
    costs, parent = {start: 0}, {}
    while queue:
        _, cost, current = heapq.heappop(queue)
        if cost != costs[current]:
            continue
        if current == goal:
            path = [goal]
            while path[-1] != start:
                path.append(parent[path[-1]])
            return path[::-1]
        x, y = current
        for nxt in ((x-1, y), (x+1, y), (x, y-1), (x, y+1)):
            nx, ny = nxt
            if not (0 <= nx < width and 0 <= ny < height) or not walkable[ny, nx]:
                continue
            # Prefer open corridors rather than hugging the minimum-clearance edge.
            penalty = (0.0 if clearance is None else
                       .05 / max(.01, float(clearance[ny, nx]) - required_clearance))
            new = cost + 1 + penalty
            if new < costs.get(nxt, math.inf):
                costs[nxt], parent[nxt] = new, current
                heapq.heappush(queue, (new + h(nxt), new, nxt))
    return []


def corner_cells(cells):
    """Remove only collinear points; never shortcut across an obstacle corner."""
    if len(cells) < 3:
        return cells[:]
    result = [cells[0]]
    for a, b, c in zip(cells, cells[1:], cells[2:]):
        if (b[0]-a[0], b[1]-a[1]) != (c[0]-b[0], c[1]-b[1]):
            result.append(b)
    return result + [cells[-1]]


def track_waypoint(pose, target, max_speed, max_turn, tolerance):
    """Stop to turn when misaligned; otherwise approach the next waypoint."""
    dx, dy = target[0]-pose[0], target[1]-pose[1]
    distance = math.hypot(dx, dy)
    if distance <= tolerance:
        return 0.0, 0.0, True
    error = wrap(math.atan2(dy, dx) - pose[2])
    angular = max(-max_turn, min(max_turn, 1.5 * error))
    linear = 0.0 if abs(error) > .15 else min(max_speed, .8 * distance)
    return linear, angular, False


def laser_points(scan, extrinsic, inf_is_clear=True):
    """Return finite obstacle endpoints in the robot frame, or None if invalid.

    Require a horizontal, essentially full 360-degree scan. Positive infinity
    means no return only when inf_is_clear is enabled. Sensor offsets and yaw
    are handled by extrinsic=(x,y,yaw) from TF.
    """
    n = len(scan.ranges)
    values = (scan.angle_min, scan.angle_increment, scan.range_min, scan.range_max)
    if (n < 72 or not all(math.isfinite(v) for v in values)
            or scan.range_min < 0 or scan.range_max <= max(.60, scan.range_min)
            or abs(scan.angle_increment) < 1e-9
            or abs(scan.angle_increment) * n < 2*math.pi - .10):
        return None
    r = np.asarray(scan.ranges, dtype=float)
    finite = np.isfinite(r)
    valid = finite & (r >= scan.range_min) & (r <= scan.range_max)
    if inf_is_clear:
        valid |= np.isposinf(r)
    if not np.all(valid):
        return None
    angles = scan.angle_min + np.arange(n)[finite] * scan.angle_increment + extrinsic[2]
    return np.column_stack((extrinsic[0] + r[finite]*np.cos(angles),
                            extrinsic[1] + r[finite]*np.sin(angles)))


def lidar_allows(points, linear, angular, radius, margin):
    """Conservative circular rotation envelope and forward stopping corridor."""
    if points is None:
        return False
    if not len(points):
        return True
    if np.any(np.hypot(points[:, 0], points[:, 1]) <= radius + margin):
        return False
    if linear > 0:
        # Allow for sensing/control delay plus braking at 0.1 m/s^2.
        stop_x = radius + margin + linear*.5 + linear*linear/(2*.1)
        in_corridor = (points[:, 0] >= 0) & (np.abs(points[:, 1]) <= radius + margin)
        if np.any(in_corridor & (points[:, 0] <= stop_x)):
            return False
    return True


def limit_forward_speed(points, desired, radius, margin):
    """Match speed to the same delay/braking model used by lidar_allows."""
    if desired <= 0 or points is None or not len(points):
        return desired
    corridor = (points[:, 0] >= 0) & (np.abs(points[:, 1]) <= radius + margin)
    if not np.any(corridor):
        return desired
    gap = float(np.min(points[corridor, 0])) - radius - margin - .005
    if gap <= 0:
        return 0.0
    # Solve 5*v**2 + 0.5*v <= gap; never exceed the requested speed.
    return min(desired, (math.sqrt(.25 + 20*gap) - .5) / 10)


class Explorer(Node):
    def __init__(self):
        super().__init__('my_explorer')
        defaults = {
            'robot_frame': 'base_footprint', 'free_threshold': 40,
            'robot_radius': .12, 'safety_margin': .05,
            'forward_speed': .06, 'turn_speed': .4,
            'approach_search_distance': .60, 'min_goal_distance': .30,
            'scan_timeout': .5, 'map_timeout': 5.0, 'tf_timeout': 1.0,
            'retry_delay': 60.0, 'inf_is_clear': True,
            'enable_motion': True,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
            setattr(self, name, self.get_parameter(name).value)
        if (not all(math.isfinite(v) for v in (self.robot_radius, self.safety_margin,
                                               self.forward_speed, self.turn_speed))
                or not 0 <= self.free_threshold < 100 or self.robot_radius <= 0
                or self.safety_margin < .03 or not 0 < self.forward_speed <= .12
                or not 0 < self.turn_speed <= .8):
            raise ValueError('Invalid footprint/threshold/speed parameters.')
        for name in ('scan_timeout', 'map_timeout', 'tf_timeout', 'retry_delay',
                     'approach_search_distance', 'min_goal_distance'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be finite and positive.')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.cmd_pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.path_pub = self.create_publisher(Path, '/explorer/path', 10)
        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback,
                                                  qos_profile_sensor_data)
        self.grid = self.info = None
        self.frame = ''
        self.map_arrival = self.scan_arrival = -math.inf
        self.scan_stamp = -math.inf
        self.points = None
        self.map_version = 0
        self.planned_version = -1
        self.waypoints = []
        self.waypoint_index = 0
        self.goal = None
        self.recent_goals = []
        self.next_plan = 0.0
        self.blocked_since = None
        self.progress_time = time.monotonic()
        self.best_distance = math.inf
        self.last_pose = None
        self.stopping = False
        self.status = None
        self.paused = False
        self.clear_since = None
        self.envelope_hold = False
        self.invalid_path_since = None
        self.episode_anchor = None
        self.episode_started = time.monotonic()
        self.replan_count = 0
        self.stationary_hold = False
        self.timer = self.create_timer(.1, self.control_loop,
                                       clock=Clock(clock_type=ClockType.STEADY_TIME))
        self.get_logger().info('Controller revision 2.1: start-cell diagnostics; motion limits unchanged.')
        self.report('WAIT: map, scan and robot transform required.')

    def report(self, text):
        if text != self.status:
            self.get_logger().info(text)
            self.status = text

    def command(self, linear=0.0, angular=0.0):
        if self.stopping or not self.enable_motion:
            linear = angular = 0.0
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.robot_frame
        msg.twist.linear.x, msg.twist.angular.z = float(linear), float(angular)
        self.cmd_pub.publish(msg)

    def map_callback(self, msg):
        resolution = msg.info.resolution
        q, p = msg.info.origin.orientation, msg.info.origin.position
        valid = (msg.info.width > 0 and msg.info.height > 0
                 and len(msg.data) == msg.info.width * msg.info.height
                 and math.isfinite(resolution) and resolution > 0
                 and bool(msg.header.frame_id)
                 and all(math.isfinite(v) for v in (p.x, p.y, q.x, q.y, q.z, q.w))
                 and abs(q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w-1) < .01
                 and abs(q.x) < .01 and abs(q.y) < .01)
        if not valid:
            self.grid = None
            self.command()
            self.report('WAIT: invalid map.')
            return
        grid = np.asarray(msg.data, dtype=np.int16).reshape(msg.info.height, msg.info.width)
        if np.any((grid < -1) | (grid > 100)):
            self.grid = None
            self.command()
            return
        if self.frame and self.frame != msg.header.frame_id:
            self.abandon('Map frame changed.', remember=False)
            self.recent_goals.clear()
        self.grid, self.info, self.frame = grid, msg.info, msg.header.frame_id
        self.map_arrival = time.monotonic()
        self.map_version += 1

    def scan_callback(self, msg):
        self.points = None
        self.scan_arrival = time.monotonic()
        self.scan_stamp = stamp_seconds(msg.header.stamp)
        try:
            if msg.header.frame_id == self.robot_frame:
                extrinsic = (0., 0., 0.)
            else:
                tf = self.tf_buffer.lookup_transform(self.robot_frame, msg.header.frame_id,
                                                      Time.from_msg(msg.header.stamp))
                q, t = tf.transform.rotation, tf.transform.translation
                if abs(q.x) > .02 or abs(q.y) > .02:
                    return
                extrinsic = (t.x, t.y, yaw_of(q))
            if all(math.isfinite(v) for v in extrinsic):
                self.points = laser_points(msg, extrinsic, self.inf_is_clear)
        except TransformException:
            pass

    def robot_pose(self):
        try:
            tf = self.tf_buffer.lookup_transform(self.frame, self.robot_frame, Time())
            age = self.get_clock().now().nanoseconds * 1e-9 - stamp_seconds(tf.header.stamp)
            if not -.1 <= age <= self.tf_timeout:
                return None
            t, q = tf.transform.translation, tf.transform.rotation
            pose = (t.x, t.y, yaw_of(q))
            return pose if all(math.isfinite(v) for v in pose) else None
        except TransformException:
            return None

    def publish_path(self, points):
        msg = Path()
        msg.header.frame_id = self.frame
        msg.header.stamp = self.get_clock().now().to_msg()
        for i, point in enumerate(points):
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x, pose.pose.position.y = map(float, point)
            a, b = (points[i], points[i+1]) if i+1 < len(points) else (points[max(0, i-1)], point)
            yaw = math.atan2(b[1]-a[1], b[0]-a[0])
            pose.pose.orientation.z, pose.pose.orientation.w = math.sin(yaw/2), math.cos(yaw/2)
            msg.poses.append(pose)
        self.path_pub.publish(msg)

    def abandon(self, reason, remember=True):
        self.command()
        if remember and self.goal is not None:
            self.recent_goals.append((self.goal, time.monotonic() + self.retry_delay))
        self.goal = None
        self.waypoints = []
        self.blocked_since = None
        self.clear_since = None
        self.invalid_path_since = None
        self.best_distance = math.inf
        self.publish_path([])
        self.next_plan = time.monotonic() + 2.0
        self.report(reason)

    def cooling(self, point):
        return any(math.dist(point, goal) < .25 for goal, _ in self.recent_goals)

    def planning_grid(self, pose):
        free = (self.grid >= 0) & (self.grid <= self.free_threshold)
        # Overlay fresh LiDAR endpoints so A* can route around newly seen obstacles.
        # This temporary overlay never modifies the SLAM occupancy grid.
        if self.points is not None and len(self.points):
            c, s = math.cos(pose[2]), math.sin(pose[2])
            for x, y in self.points:
                cell = world_to_cell((pose[0]+c*x-s*y, pose[1]+s*x+c*y), self.info)
                cx, cy = cell
                if 0 <= cy < free.shape[0] and 0 <= cx < free.shape[1]:
                    free[cy, cx] = False
        distances = clearance_grid(free, self.info.resolution)
        return free, distances, free & (distances > self.robot_radius+self.safety_margin)

    def segment_clear(self, a, b, distances):
        # Account for the actual point's offset from the sampled cell center.
        # Include half the sample spacing so gaps between samples are covered.
        length = math.dist(a, b)
        steps = max(1, math.ceil(length / (self.info.resolution / 4)))
        sampling_margin = length / steps / 2
        required = self.robot_radius + .015 + sampling_margin
        for k in range(steps+1):
            f = k / steps
            point = (a[0]+f*(b[0]-a[0]), a[1]+f*(b[1]-a[1]))
            x, y = world_to_cell(point, self.info)
            if not (0 <= y < distances.shape[0] and 0 <= x < distances.shape[1]):
                return False
            offset = math.dist(point, cell_to_world((x, y), self.info))
            if distances[y, x] - offset <= required:
                return False
        return True

    def find_approach(self, group, free, reachable, robot):
        limit = math.floor(self.approach_search_distance / self.info.resolution + 1e-9)
        queue = deque((int(x), int(y), 0) for x, y in group)
        seen = set(group)
        while queue:
            x, y, depth = queue.popleft()
            if reachable[y, x]:
                point = cell_to_world((x, y), self.info)
                if math.dist(point, robot[:2]) >= self.min_goal_distance and not self.cooling(point):
                    return x, y
            if depth >= limit:
                continue
            for nx, ny in ((x-1,y), (x+1,y), (x,y-1), (x,y+1)):
                if (0 <= ny < free.shape[0] and 0 <= nx < free.shape[1]
                        and free[ny,nx] and (nx,ny) not in seen):
                    seen.add((nx,ny))
                    queue.append((nx,ny,depth+1))
        return None

    def plan(self, pose):
        # Stop before synchronous planning; do not leave a nonzero command active.
        self.command()
        now = time.monotonic()
        self.recent_goals = [(p,t) for p,t in self.recent_goals if t > now]
        self.next_plan = now + 2.0
        free, distances, walkable = self.planning_grid(pose)
        start = world_to_cell(pose[:2], self.info)
        x, y = start
        if not (0 <= y < free.shape[0] and 0 <= x < free.shape[1]) or not walkable[y,x]:
            measured = (float(distances[y,x]) if
                        0 <= y < free.shape[0] and 0 <= x < free.shape[1] else 0.0)
            self.abandon(f'WAIT: robot cell clearance={measured:.3f} m; '
                         f'required={self.robot_radius+self.safety_margin:.3f} m. No automatic relocation.',
                         remember=False)
            self.get_logger().info(
                self.describe_start(pose, start, free, distances),
                throttle_duration_sec=5.0
            )
            return
        labels, _ = label(walkable, structure=np.array([[0,1,0],[1,1,1],[0,1,0]]))
        reachable = labels == labels[y,x]
        groups = frontier_groups(self.grid, self.free_threshold)
        if not groups:
            self.abandon('IDLE: no frontiers at the current threshold. Waiting for map updates.', False)
            return
        candidates = set()
        # Retain an active destination when it still lies near a frontier.
        if self.goal is not None:
            previous = world_to_cell(self.goal, self.info)
            gx, gy = previous
            near_frontier = any(
                abs(gx-int(cx)) + abs(gy-int(cy)) <= self.approach_search_distance/self.info.resolution
                for group in groups for cx,cy in group
            )
            if (near_frontier and 0 <= gy < free.shape[0] and 0 <= gx < free.shape[1]
                    and reachable[gy,gx]):
                candidates.add(previous)
        if not candidates:
            for group in groups:
                cell = self.find_approach(group, free, reachable, pose)
                if cell is not None:
                    candidates.add(cell)
        # Simple initial ranking by straight-line distance; A* supplies the route.
        ranked = sorted(candidates, key=lambda cell: math.dist(cell_to_world(cell,self.info),pose[:2]))
        for goal_cell in ranked:
            cells = astar(walkable, start, goal_cell, distances,
                          self.robot_radius + self.safety_margin)
            if not cells:
                continue
            points = [cell_to_world(cell,self.info) for cell in corner_cells(cells)]
            # Validate the short connection from actual pose to the grid path too.
            segments = list(zip([pose[:2]] + points, points))
            if not all(self.segment_clear(a,b,distances) for a,b in segments):
                continue
            self.goal = cell_to_world(goal_cell,self.info)
            self.waypoints, self.waypoint_index = points, 0
            self.planned_version = self.map_version
            self.progress_time, self.best_distance = time.monotonic(), math.inf
            # Replanning must not erase the duration of a blocked episode.
            self.replan_count += 1
            self.invalid_path_since = None
            self.publish_path([pose[:2]] + points)
            self.report(f'PLAN: {len(cells)} cells, {(len(cells)-1)*self.info.resolution:.2f} m; '
                        f'goal=({self.goal[0]:.2f}, {self.goal[1]:.2f}); groups={len(groups)}')
            return
        self.goal = None
        self.waypoints = []
        self.publish_path([])
        if self.recent_goals:
            self.report('WAIT: no eligible goal now; recently attempted areas are cooling down.')
        else:
            self.report('IDLE: frontiers remain but no usable approach path. Mapping is not proven complete.')

    def describe_start(self, pose, start, free, distances):
        x, y = start
        height, width = self.grid.shape
        prefix = (f'START CHECK: pose=({pose[0]:.3f}, {pose[1]:.3f}), '
                  f'cell={start}, map={width}x{height}, '
                  f'resolution={self.info.resolution:.3f}, threshold={self.free_threshold}')
        if not (0 <= x < width and 0 <= y < height):
            return prefix + ' | reason=OUTSIDE_MAP'
        value = int(self.grid[y,x])
        raw_free = (self.grid >= 0) & (self.grid <= self.free_threshold)
        raw_clearance = float(clearance_grid(raw_free,self.info.resolution)[y,x])
        if value == -1:
            reason = 'UNKNOWN_ROBOT_CELL'
        elif not raw_free[y,x]:
            reason = 'ROBOT_CELL_ABOVE_FREE_THRESHOLD'
        elif not free[y,x]:
            reason = 'LIDAR_ENDPOINT_IN_ROBOT_CELL'
        else:
            reason = 'NEIGHBOR_CLEARANCE_TOO_SMALL'
        nearest = (float(np.min(np.hypot(self.points[:,0],self.points[:,1])))
                   if self.points is not None and len(self.points) else math.inf)
        local = self.grid[max(0,y-2):min(height,y+3),max(0,x-2):min(width,x+3)].tolist()
        return (prefix + f' | reason={reason}, occupancy={value}, '
                f'map_only_clearance={raw_clearance:.3f}, '
                f'with_lidar_clearance={float(distances[y,x]):.3f}, '
                f'nearest_lidar={nearest:.3f} | local_grid={local}')

    def control_loop(self):
        if self.stopping:
            self.command()
            return
        now = time.monotonic()
        ros_now = self.get_clock().now().nanoseconds * 1e-9
        fresh = (self.grid is not None and now-self.map_arrival <= self.map_timeout
                 and self.points is not None and now-self.scan_arrival <= self.scan_timeout
                 and -.1 <= ros_now-self.scan_stamp <= self.scan_timeout)
        pose = self.robot_pose() if fresh else None
        if pose is None:
            self.command()
            self.paused = True
            self.report('WAIT: missing/stale map, scan or TF. Motors stopped.')
            return
        if self.paused:
            self.paused = False
            self.abandon('Fresh inputs restored; replanning.', remember=False)
            return
        self.last_pose = pose
        # A new path is not physical progress. Bound retries from the same spot.
        if self.episode_anchor is None or math.dist(pose[:2], self.episode_anchor) >= .10:
            self.episode_anchor = pose[:2]
            self.episode_started = now
            self.replan_count = 0
            self.stationary_hold = False
        if (self.replan_count >= 4 and now-self.episode_started > 30.0):
            self.stationary_hold = True
        if self.stationary_hold:
            self.command()
            self.publish_path([])
            self.report('STOPPED: repeated replans without 0.10 m of progress. Inspect the surroundings before restarting.')
            return

        # Inside the configured circular envelope, changing the goal cannot help.
        # Hold until clearance is stable; do not repeatedly replace the path.
        nearest = (float(np.min(np.hypot(self.points[:,0], self.points[:,1])))
                   if len(self.points) else math.inf)
        envelope = self.robot_radius + self.safety_margin
        if nearest <= envelope:
            if not self.envelope_hold:
                self.abandon(f'HOLD: nearest LiDAR point={nearest:.3f} m; required envelope={envelope:.3f} m.', False)
            self.envelope_hold = True
            self.clear_since = None
        if self.envelope_hold:
            self.command()
            if nearest > envelope + .02:
                if self.clear_since is None:
                    self.clear_since = now
                if now-self.clear_since >= .5:
                    self.envelope_hold = False
                    self.clear_since = None
                    self.next_plan = now
                    self.report('Envelope clear for 0.5 seconds; replanning.')
            else:
                self.clear_since = None
            return
        if not self.waypoints:
            self.command()
            if now >= self.next_plan:
                self.plan(pose)
            return
        # Revalidate remaining path on new maps, using fresh LiDAR endpoints too.
        if self.map_version != self.planned_version:
            self.command()
            _, distances, _ = self.planning_grid(pose)
            remaining = self.waypoints[self.waypoint_index:]
            valid = all(self.segment_clear(a,b,distances)
                        for a,b in zip([pose[:2]]+remaining, remaining))
            if not valid:
                if self.invalid_path_since is None:
                    self.invalid_path_since = now
                if now-self.invalid_path_since >= .5 and now >= self.next_plan:
                    self.plan(pose)
                return
            self.invalid_path_since = None
            self.planned_version = self.map_version
        target = self.waypoints[self.waypoint_index]
        # Small tolerance prevents cutting corners of a cell-center path.
        tolerance = min(.015, self.info.resolution*.25)
        linear, angular, reached = track_waypoint(pose,target,self.forward_speed,self.turn_speed,tolerance)
        if reached:
            self.command()
            self.waypoint_index += 1
            self.progress_time, self.best_distance = now, math.inf
            if self.waypoint_index >= len(self.waypoints):
                self.abandon('GOAL REACHED: pausing for map updates, then choosing another frontier.')
            return
        wanted_linear = linear
        linear = limit_forward_speed(self.points,linear,self.robot_radius,self.safety_margin)
        blocked = ((wanted_linear > 0 and linear < .008)
                   or not lidar_allows(self.points,linear,angular,self.robot_radius,self.safety_margin))
        if blocked:
            self.command()
            if self.blocked_since is None:
                self.blocked_since = now
            self.clear_since = None
            self.report(f'BLOCKED: forward corridor; nearest point={nearest:.3f} m. Waiting for stable clearance.')
            if now-self.blocked_since >= 3.0:
                self.abandon('Obstacle persists; temporarily excluding this goal and replanning.')
            return
        if self.blocked_since is not None:
            # A single clear scan must not restart the robot.
            self.command()
            if self.clear_since is None:
                self.clear_since = now
            if now-self.clear_since < .5:
                if now-self.blocked_since >= 3.0:
                    self.abandon('Intermittent blockage persists; trying another approach.')
                return
            self.blocked_since = None
            self.clear_since = None
        # Progress includes heading alignment, so a necessary turn is not mistaken for a stall.
        angle = abs(wrap(math.atan2(target[1]-pose[1],target[0]-pose[0])-pose[2]))
        metric = math.dist(pose[:2],target) + .10*angle
        if metric < self.best_distance - .01:
            self.best_distance, self.progress_time = metric, now
        elif now-self.progress_time > 20.0:
            self.abandon('No progress for 20 seconds; choosing another frontier.')
            return
        self.report('FOLLOWING: Python waypoint controller.' if self.enable_motion
                    else 'PREVIEW: enable_motion is false; all velocity commands are zero.')
        self.command(linear,angular)


def main(args=None):
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = Explorer()
    def request_stop(signum, frame):
        node.stopping = True
    old_int = signal.signal(signal.SIGINT,request_stop)
    old_term = signal.signal(signal.SIGTERM,request_stop)
    try:
        while rclpy.ok() and not node.stopping:
            rclpy.spin_once(node,timeout_sec=.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.stopping = True
        if rclpy.ok():
            node.command()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        signal.signal(signal.SIGINT,old_int)
        signal.signal(signal.SIGTERM,old_term)


if __name__ == '__main__':
    main()
