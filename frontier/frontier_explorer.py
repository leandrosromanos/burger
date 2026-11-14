#!/usr/bin/env python3
"""Frontier exploration supervisor for Cartographer + Nav2 (ROS 2 Jazzy).
Never publishes cmd_vel. Run without teleop or the reactive avoidance node.
The grid model is conservative; Nav2 validates paths and handles local obstacles.
"""
import math
import time
from collections import deque
from dataclasses import dataclass
import numpy as np

NEIGHBOURS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def dilate(mask, radius, outside=False):
    """Circular dilation including half-cell discretization uncertainty."""
    n = math.ceil(radius)
    padded = np.pad(mask, n, constant_values=outside)
    h, w = mask.shape
    result = np.zeros_like(mask, dtype=bool)
    for dy in range(-n, n+1):
        for dx in range(-n, n+1):
            if dx*dx + dy*dy <= radius*radius:
                result |= padded[n+dy:n+dy+h, n+dx:n+dx+w]
    return result


def flood(mask, start):
    h, w = mask.shape
    distances = np.full((h, w), -1, dtype=np.int32)
    y, x = start
    if not (0 <= y < h and 0 <= x < w and mask[y, x]):
        return distances
    queue = deque([(y, x)])
    distances[y, x] = 0
    while queue:
        y, x = queue.popleft()
        for dy, dx in NEIGHBOURS:
            yy, xx = y+dy, x+dx
            if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and distances[yy, xx] < 0:
                distances[yy, xx] = distances[y, x]+1
                queue.append((yy, xx))
    return distances


def components(mask):
    pending = set(zip(*np.where(mask)))
    groups = []
    while pending:
        seed = min(pending)
        pending.remove(seed)
        group, queue = [seed], deque([seed])
        while queue:
            y, x = queue.popleft()
            for dy, dx in NEIGHBOURS:
                cell = (y+dy, x+dx)
                if cell in pending:
                    pending.remove(cell)
                    queue.append(cell)
                    group.append(cell)
        groups.append(group)
    return groups


@dataclass
class Grid:
    cells: np.ndarray
    resolution: float
    origin_x: float = 0.0
    origin_y: float = 0.0
    origin_yaw: float = 0.0

    def cell(self, x, y):
        dx, dy = x-self.origin_x, y-self.origin_y
        c, s = math.cos(self.origin_yaw), math.sin(self.origin_yaw)
        return (math.floor((-s*dx+c*dy)/self.resolution),
                math.floor((c*dx+s*dy)/self.resolution))

    def world(self, y, x):
        xx, yy = (x+.5)*self.resolution, (y+.5)*self.resolution
        c, s = math.cos(self.origin_yaw), math.sin(self.origin_yaw)
        return self.origin_x+c*xx-s*yy, self.origin_y+s*xx+c*yy


def plan_frontiers(grid, robot_xy, clearance=.18, standoff=.35):
    """Return reachable observation poses and counts, never unknown-space goals.

    Paths are 4-connected in free space eroded by clearance. Frontier components
    use the occupied-inflated map; a short known-space BFS connects their cells
    to safe observation positions. Small frontiers are retained, not silently
    treated as completed. Goal and path inflation treat unknown as blocked.
    """
    a, r = grid.cells, grid.resolution
    free, unknown = (a >= 0) & (a <= 25), a < 0
    adjacent = np.zeros(a.shape, dtype=bool)
    adjacent[1:] |= unknown[:-1]
    adjacent[:-1] |= unknown[1:]
    adjacent[:, 1:] |= unknown[:, :-1]
    adjacent[:, :-1] |= unknown[:, 1:]
    adjacent[0] = adjacent[-1] = True
    adjacent[:, 0] = adjacent[:, -1] = True
    frontier = free & adjacent
    # A half diagonal accounts conservatively for occupied cell area.
    inflation = clearance/r + math.sqrt(.5)
    loose = free & ~dilate((a > 25), inflation, outside=True)
    safe = free & ~dilate(~free, inflation, outside=True)
    start = grid.cell(*robot_xy)
    distance = flood(safe, start)
    if not np.any(distance >= 0):
        return [], {'reason': 'ROBOT_NOT_IN_SAFE_FREE_SPACE', 'reachable': 0,
                    'unresolved': int(frontier.sum())}
    loose_dist = flood(loose, start)
    reachable_frontier = frontier & (loose_dist >= 0)
    groups = components(reachable_frontier)
    goal_mask = (distance >= 0) & ~dilate(unknown, standoff/r+math.sqrt(.5), outside=True)
    goals, unresolved = [], 0
    h, w = a.shape
    for group in groups:
        found = {}
        # Sample up to 24 locations along each connected frontier, giving
        # alternatives when one viewing position repeatedly fails.
        samples = sorted(group)
        for index in np.linspace(0, len(samples)-1, min(24, len(samples)), dtype=int):
            seed = samples[index]
            queue, seen = deque([(seed, 0)]), {seed}
            while queue:
                (y, x), steps = queue.popleft()
                # A known-space path alone can bend around an occluding corner;
                # also require an unobstructed sight line to this frontier.
                sy, sx = seed
                samples_count = 2*max(abs(sy-y), abs(sx-x))+1
                line_y = np.rint(np.linspace(y, sy, samples_count)).astype(int)
                line_x = np.rint(np.linspace(x, sx, samples_count)).astype(int)
                if goal_mask[y, x] and np.all(free[line_y, line_x]):
                    found[(y, x)] = seed
                    break
                if steps*r >= standoff + .45:
                    continue
                for dy, dx in NEIGHBOURS:
                    cell = yy, xx = y+dy, x+dx
                    if (0 <= yy < h and 0 <= xx < w and loose[yy, xx]
                            and cell not in seen):
                        seen.add(cell)
                        queue.append((cell, steps+1))
        if not found:
            unresolved += 1
        for (y, x), (fy, fx) in found.items():
            gx, gy = grid.world(y, x)
            tx, ty = grid.world(fy, fx)
            yaw = math.atan2(ty-gy, tx-gx)
            score = distance[y, x]*r - .3*math.log1p(len(group)*r)
            goals.append((score, gx, gy, yaw))
    goals.sort()
    return goals, {'reason': 'OK', 'reachable': len(groups),
                   'unresolved': unresolved,
                   'disconnected_cells': int((frontier & ~reachable_frontier).sum())}


class GoalHistory:
    """Spatial cooldown and finite retries; exhausted goals are not completion."""
    def __init__(self):
        self.records = []

    def record(self, xy, now, success):
        previous = next((v for v in self.records if math.dist(v['xy'], xy) < .35), None)
        if previous is None:
            previous = {'xy': xy, 'tries': 0, 'until': 0}
            self.records.append(previous)
        previous['tries'] += 1
        previous['until'] = now + (20 if success else 60)

    def allowed(self, xy, now):
        return all(math.dist(v['xy'], xy) >= .35 or
                   (v['tries'] < 3 and now >= v['until']) for v in self.records)


def main(args=None):
    import rclpy
    from rclpy.node import Node
    from rclpy.action import ActionClient
    from rclpy.clock import Clock, ClockType
    from rclpy.time import Time
    from rclpy.signals import SignalHandlerOptions
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from nav_msgs.msg import OccupancyGrid
    from nav2_msgs.action import NavigateToPose
    from geometry_msgs.msg import PoseStamped
    from action_msgs.msg import GoalStatus
    from std_msgs.msg import String
    from std_srvs.srv import SetBool
    from tf2_ros import Buffer, TransformListener, TransformException

    class Explorer(Node):
        def __init__(self):
            super().__init__('frontier_explorer')
            for name, value in {'base_frame': 'base_footprint', 'map_topic': '/map',
                                'clearance': .18, 'standoff': .35,
                                'goal_timeout': 240., 'progress_timeout': 40.,
                                'map_timeout': 8.}.items():
                self.declare_parameter(name, value)
            if not (.14 <= self.param('clearance') <= .5 and
                    self.param('standoff') >= self.param('clearance')+.1):
                raise ValueError('Need clearance >= .14 and standoff >= clearance + .1')
            self.grid = None
            self.map_frame = None
            self.map_time = 0.
            self.map_seq = 0
            self.tf = Buffer()
            self.listener = TransformListener(self.tf, self)
            # Volatile subscriber also matches transient-local publishers and
            # receives their NEXT update; avoids requiring a particular mapper QoS.
            qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.VOLATILE)
            self.sub = self.create_subscription(OccupancyGrid, self.param('map_topic'), self.on_map, qos)
            self.nav = ActionClient(self, NavigateToPose, '/navigate_to_pose')
            self.status_pub = self.create_publisher(String, '~/status', 10)
            self.goal_pub = self.create_publisher(PoseStamped, '~/goal', 10)
            self.pause_service = self.create_service(SetBool, '~/pause', self.pause)
            self.history = GoalHistory()
            self.handle = self.pending = self.result_future = None
            self.target = None
            self.cancel_reason = None
            self.cancel_sent = False
            self.fault = False
            self.paused = False
            self.stopping = False
            self.start_time = self.last_progress = 0.
            self.progress_xy = None
            self.next_plan = 0.
            self.empty_since = None
            self.empty_sequences = set()
            self.last_status = ''
            self.last_status_time = 0.
            self.timer = self.create_timer(.5, self.tick,
                                          clock=Clock(clock_type=ClockType.STEADY_TIME))

        def param(self, name):
            return self.get_parameter(name).value

        def status(self, text):
            now = time.monotonic()
            if text != self.last_status or now-self.last_status_time >= 10:
                self.get_logger().info(text)
                self.status_pub.publish(String(data=text))
                self.last_status, self.last_status_time = text, now

        def on_map(self, msg):
            info = msg.info
            q = info.origin.orientation
            vals = (info.resolution, info.origin.position.x, info.origin.position.y,
                    q.x, q.y, q.z, q.w)
            valid = (all(math.isfinite(v) for v in vals) and
                     info.resolution > 0 and info.width > 0 and info.height > 0 and
                     len(msg.data) == info.width*info.height and bool(msg.header.frame_id) and
                     abs(q.x) < .001 and abs(q.y) < .001 and
                     abs(q.z*q.z+q.w*q.w-1) < .01)
            if not valid:
                self.grid = None
                return
            cells = np.asarray(msg.data, dtype=np.int16).reshape(info.height, info.width)
            if np.any((cells < -1) | (cells > 100)):
                self.grid = None
                return
            yaw = math.atan2(2*q.w*q.z, 1-2*q.z*q.z)
            if self.map_frame is not None and self.map_frame != msg.header.frame_id:
                self.fault = True
                self.cancel('Map frame changed; restart explorer')
            self.map_frame = msg.header.frame_id
            self.grid = Grid(cells, info.resolution, info.origin.position.x,
                             info.origin.position.y, yaw)
            self.map_time = time.monotonic()
            self.map_seq += 1

        def pose(self):
            try:
                transform = self.tf.lookup_transform(self.map_frame, self.param('base_frame'), Time())
                stamp = transform.header.stamp
                age = self.get_clock().now().nanoseconds*1e-9 - (stamp.sec+stamp.nanosec*1e-9)
                if not -.2 <= age <= 1.:
                    return None
                p = transform.transform.translation
                if not math.isfinite(p.x) or not math.isfinite(p.y):
                    return None
                return p.x, p.y
            except TransformException:
                return None

        def pause(self, request, response):
            self.paused = request.data
            if self.paused:
                self.cancel('Paused by user')
            response.success = True
            response.message = 'Pause requested; waiting for cancellation' if self.paused else 'Exploration resumed'
            return response

        def busy(self):
            return self.pending is not None or self.handle is not None

        def cancel(self, reason):
            if not self.busy():
                return
            if self.cancel_reason is None:
                self.cancel_reason = reason
            if self.handle is not None and not self.cancel_sent:
                self.cancel_sent = True
                self.handle.cancel_goal_async().add_done_callback(self.cancel_ack)
                self.status('CANCELING: ' + self.cancel_reason)

        def cancel_ack(self, future):
            try:
                reply = future.result()
                if not reply.goals_canceling:
                    self.status('WAIT_CANCEL: goal may already have ended; waiting for result')
            except Exception as exc:
                self.fault = True
                self.status('FAULT: cancellation failed; stop Nav2: ' + str(exc))

        def goal_response(self, future):
            self.pending = None
            try:
                handle = future.result()
            except Exception as exc:
                self.fault = True
                self.status('FAULT: goal acceptance unknown; stop Nav2: ' + str(exc))
                return
            if not handle.accepted:
                self.history.record(self.target, time.monotonic(), False)
                self.target = None
                self.cancel_reason = None
                self.next_plan = time.monotonic()+3
                self.status('REJECTED: will select another goal')
                return
            self.handle = handle
            self.result_future = handle.get_result_async()
            self.result_future.add_done_callback(self.goal_result)
            if self.cancel_reason is not None or self.stopping or self.paused:
                self.cancel(self.cancel_reason or 'Stopping')

        def goal_result(self, future):
            try:
                result = future.result()
                success = result.status == GoalStatus.STATUS_SUCCEEDED
                # User/data interruption is not a failure of this frontier.
                if self.cancel_reason is None or self.cancel_reason in ('Goal timeout', 'No positional progress'):
                    self.history.record(self.target, time.monotonic(), success)
                code = getattr(result.result, 'error_code', 0)
                self.status(('REACHED' if success else 'GOAL_ENDED') + f': status={result.status}, error={code}')
            except Exception as exc:
                self.fault = True
                self.status('FAULT: result unknown; stop Nav2: ' + str(exc))
            self.handle = self.result_future = self.target = None
            self.cancel_reason = None
            self.cancel_sent = False
            self.next_plan = time.monotonic()+3
            self.empty_since = None
            self.empty_sequences.clear()

        def send(self, goal, xy, now):
            _, gx, gy, yaw = goal
            message = NavigateToPose.Goal()
            message.pose.header.frame_id = self.map_frame
            message.pose.header.stamp = self.get_clock().now().to_msg()
            message.pose.pose.position.x = float(gx)
            message.pose.pose.position.y = float(gy)
            message.pose.pose.orientation.z = math.sin(yaw/2)
            message.pose.pose.orientation.w = math.cos(yaw/2)
            self.target = (gx, gy)
            self.start_time = self.last_progress = now
            self.progress_xy = xy
            self.cancel_reason = None
            self.cancel_sent = False
            self.goal_pub.publish(message.pose)
            self.pending = self.nav.send_goal_async(message)
            self.pending.add_done_callback(self.goal_response)
            self.status(f'NAVIGATING: x={gx:.2f}, y={gy:.2f}')

        def tick(self):
            now = time.monotonic()
            if self.stopping or self.paused or self.fault:
                self.cancel('Stopping or paused')
                return
            if self.grid is None or now-self.map_time > self.param('map_timeout'):
                self.cancel('Map unavailable or stale')
                self.empty_since = None
                self.empty_sequences.clear()
                self.status('WAIT_MAP: need a live /map from SLAM')
                return
            xy = self.pose()
            if xy is None:
                self.cancel('TF unavailable or stale')
                self.empty_since = None
                self.empty_sequences.clear()
                self.status('WAIT_TF: need fresh map -> base_footprint transform')
                return
            if self.busy():
                if self.pending is not None and now-self.start_time > 10:
                    self.cancel('Goal acceptance delayed')
                    self.status('WAIT_SERVER: no new goal until previous acceptance is resolved')
                if self.handle is not None:
                    if math.dist(xy, self.progress_xy) >= .12:
                        self.progress_xy, self.last_progress = xy, now
                    if now-self.start_time > self.param('goal_timeout'):
                        self.cancel('Goal timeout')
                    elif now-self.last_progress > self.param('progress_timeout'):
                        self.cancel('No positional progress')
                return
            if now < self.next_plan:
                return
            self.next_plan = now+3
            if not self.nav.server_is_ready():
                self.status('WAIT_NAV2: /navigate_to_pose action server is not ready')
                return
            goals, stats = plan_frontiers(self.grid, xy, self.param('clearance'), self.param('standoff'))
            if stats['reason'] != 'OK':
                self.empty_since = None
                self.empty_sequences.clear()
                self.status('BLOCKED: ' + stats['reason'] + '; check map, pose and clearance')
                return
            if stats['reachable'] == 0:
                if self.empty_since is None:
                    self.empty_since = now
                self.empty_sequences.add(self.map_seq)
                if now-self.empty_since >= 15 and len(self.empty_sequences) >= 3:
                    self.status('COMPLETE_REACHABLE: no reachable frontiers under current clearance; '
                                f"disconnected frontier cells={stats['disconnected_cells']}. Save and inspect map.")
                else:
                    self.status('VERIFYING: waiting for fresh maps before declaring completion')
                # Keep watching: newly appearing frontiers automatically resume work.
                return
            self.empty_since = None
            self.empty_sequences.clear()
            for goal in goals:
                if self.history.allowed(goal[1:3], now):
                    self.send(goal, xy, now)
                    return
            self.status('UNRESOLVED: frontiers remain, but viewpoints are unsafe, cooling down '
                        'or exhausted after 3 visits. Not complete; inspect RViz.')

    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = Explorer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stopping = True
        node.cancel('User stopped explorer')
        deadline = time.monotonic()+5
        try:
            while node.busy() and time.monotonic() < deadline and rclpy.ok():
                rclpy.spin_once(node, timeout_sec=.1)
        except KeyboardInterrupt:
            pass
        if node.busy() or node.fault:
            node.get_logger().error('Cancellation not confirmed: STOP NAV2 before leaving the robot.')
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
