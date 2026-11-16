#!/usr/bin/env python3
"""Frontier exploration, revision 2.2: diagonal routes and checked shortcuts.

ROS 2 Jazzy / TurtleBot3 Burger simulation; no Nav2 dependency.
Publishes TwistStamped. Run only one cmd_vel controller and configure a base
command timeout. Exploration covers eligible reachable frontiers, not a proof
that every unknown region can be observed.
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
    return math.atan2(2*(q.w*q.z + q.x*q.y), 1-2*(q.y*q.y + q.z*q.z))


def stamp_seconds(stamp):
    return stamp.sec + stamp.nanosec*1e-9


def cell_to_world(cell, info):
    x, y = (cell[0]+.5)*info.resolution, (cell[1]+.5)*info.resolution
    c, s = math.cos(yaw_of(info.origin.orientation)), math.sin(yaw_of(info.origin.orientation))
    return info.origin.position.x+c*x-s*y, info.origin.position.y+s*x+c*y


def world_to_cell(point, info):
    dx, dy = point[0]-info.origin.position.x, point[1]-info.origin.position.y
    c, s = math.cos(yaw_of(info.origin.orientation)), math.sin(yaw_of(info.origin.orientation))
    return math.floor((c*dx+s*dy)/info.resolution), math.floor((-s*dx+c*dy)/info.resolution)


def clearance_grid(free, resolution):
    distances = distance_transform_edt(np.pad(free, 1, constant_values=False),
                                       sampling=resolution)[1:-1, 1:-1]
    return np.maximum(distances-resolution*math.sqrt(2)/2, 0)


def frontier_groups(grid, threshold):
    free, unknown = (grid >= 0) & (grid <= threshold), grid == -1
    touches = np.zeros_like(free)
    touches[1:, :] |= unknown[:-1, :]
    touches[:-1, :] |= unknown[1:, :]
    touches[:, 1:] |= unknown[:, :-1]
    touches[:, :-1] |= unknown[:, 1:]
    labels, count = label(free & touches, structure=np.ones((3, 3)))
    return [list(zip(*np.where(labels == i)[::-1])) for i in range(1, count+1)]


def route_to_goals(walkable, start, goals, clearance=None, required_clearance=.17,
                   clearance_weight=.25, edge_ok=None):
    """Shortest weighted 8-neighbor route to ANY candidate.

    Single goal: A* with octile heuristic. Multiple goals: Dijkstra (A* with
    h=0), comparing actual accumulated route costs in one search.
    Diagonals cost sqrt(2); both adjacent cardinal cells must be walkable.
    Clearance surcharge is bounded to 25% by default, unlike the old
    unbounded inverse-clearance penalty. edge_ok checks continuous segments.
    """
    height, width = walkable.shape
    def inside(cell):
        x, y = cell
        return 0 <= x < width and 0 <= y < height and walkable[y, x]
    if not inside(start):
        return [], math.inf
    goals = {g for g in goals if inside(g)}
    if not goals:
        return [], math.inf
    single = next(iter(goals)) if len(goals) == 1 else None
    def heuristic(cell):
        if single is None:
            return 0.0
        dx, dy = abs(cell[0]-single[0]), abs(cell[1]-single[1])
        return max(dx, dy)+(math.sqrt(2)-1)*min(dx, dy)
    def penalty(cell):
        if clearance is None:
            return 0.0
        spare = max(0., float(clearance[cell[1], cell[0]])-required_clearance)
        return clearance_weight*max(0., 1-spare/.20)
    queue = [(heuristic(start), 0., start)]
    costs, parent = {start: 0.}, {}
    moves = ((-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1))
    while queue:
        _, cost, current = heapq.heappop(queue)
        if cost != costs[current]:
            continue
        if current in goals:
            path = [current]
            while path[-1] != start:
                path.append(parent[path[-1]])
            return path[::-1], cost
        x, y = current
        for dx, dy in moves:
            nxt = nx, ny = x+dx, y+dy
            if not inside(nxt):
                continue
            if dx and dy and (not walkable[y, nx] or not walkable[ny, x]):
                continue
            if edge_ok is not None and not edge_ok(current, nxt):
                continue
            step = math.sqrt(2) if dx and dy else 1.
            new = cost + step*(1+(penalty(current)+penalty(nxt))/2)
            if new < costs.get(nxt, math.inf):
                costs[nxt], parent[nxt] = new, current
                heapq.heappush(queue, (new+heuristic(nxt), new, nxt))
    return [], math.inf


def astar(walkable, start, goal, clearance=None, required_clearance=.17):
    return route_to_goals(walkable, start, {goal}, clearance, required_clearance)[0]


def simplify_path(points, clear):
    """Greedily use the farthest visible waypoint; never skip a failed check."""
    if len(points) < 3:
        return points[:]
    result, anchor = [points[0]], 0
    while anchor < len(points)-1:
        candidate = len(points)-1
        while candidate > anchor+1 and not clear(points[anchor], points[candidate]):
            candidate -= 1
        if not clear(points[anchor], points[candidate]):
            return []
        result.append(points[candidate])
        anchor = candidate
    return result


def track_waypoint(pose, target, max_speed, max_turn, tolerance):
    dx, dy = target[0]-pose[0], target[1]-pose[1]
    distance = math.hypot(dx, dy)
    if distance <= tolerance:
        return 0., 0., True
    error = wrap(math.atan2(dy, dx)-pose[2])
    angular = max(-max_turn, min(max_turn, 1.5*error))
    linear = 0. if abs(error) > .15 else min(max_speed, .8*distance)
    return linear, angular, False


def laser_points(scan, extrinsic, inf_is_clear=True):
    n = len(scan.ranges)
    values = (scan.angle_min, scan.angle_increment, scan.range_min, scan.range_max)
    if (n < 72 or not all(math.isfinite(v) for v in values)
            or scan.range_min < 0 or scan.range_max <= max(.60, scan.range_min)
            or abs(scan.angle_increment) < 1e-9
            or abs(scan.angle_increment)*n < 2*math.pi-.10):
        return None
    r = np.asarray(scan.ranges, dtype=float)
    finite = np.isfinite(r)
    valid = finite & (r >= scan.range_min) & (r <= scan.range_max)
    if inf_is_clear:
        valid |= np.isposinf(r)
    if not np.all(valid):
        return None
    angles = scan.angle_min+np.arange(n)[finite]*scan.angle_increment+extrinsic[2]
    return np.column_stack((extrinsic[0]+r[finite]*np.cos(angles),
                            extrinsic[1]+r[finite]*np.sin(angles)))


def lidar_allows(points, linear, angular, radius, margin):
    if points is None:
        return False
    if not len(points):
        return True
    if np.any(np.hypot(points[:,0], points[:,1]) <= radius+margin):
        return False
    if linear > 0:
        stop_x = radius+margin+linear*.5+linear*linear/(2*.1)
        corridor = (points[:,0] >= 0) & (np.abs(points[:,1]) <= radius+margin)
        if np.any(corridor & (points[:,0] <= stop_x)):
            return False
    return True


def limit_forward_speed(points, desired, radius, margin):
    if desired <= 0 or points is None or not len(points):
        return desired
    corridor = (points[:,0] >= 0) & (np.abs(points[:,1]) <= radius+margin)
    if not np.any(corridor):
        return desired
    gap = float(np.min(points[corridor,0]))-radius-margin-.005
    if gap <= 0:
        return 0.
    return min(desired, (math.sqrt(.25+20*gap)-.5)/10)


class Explorer(Node):
    def __init__(self):
        super().__init__('my_explorer')
        defaults = {
            'robot_frame': 'base_footprint', 'free_threshold': 40,
            'robot_radius': .12, 'safety_margin': .05,
            'forward_speed': .09, 'turn_speed': .4,
            'approach_search_distance': .60, 'min_goal_distance': .30,
            'scan_timeout': .5, 'map_timeout': 5., 'tf_timeout': 1.,
            'retry_delay': 60., 'inf_is_clear': True, 'enable_motion': True,
            'clearance_weight': .25,
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
        if not math.isfinite(self.clearance_weight) or not 0 <= self.clearance_weight <= 1:
            raise ValueError('clearance_weight must be between 0 and 1.')
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
        self.map_arrival = self.scan_arrival = self.scan_stamp = -math.inf
        self.points = None
        self.map_version, self.planned_version = 0, -1
        self.waypoints, self.waypoint_index = [], 0
        self.goal, self.recent_goals = None, []
        self.next_plan = 0.
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
        self.get_logger().info('Controller revision 2.2: diagonal routes, checked shortcuts, '
                               f'route-cost goal selection; max speed={self.forward_speed:.2f} m/s.')
        self.report('WAIT: map, scan and robot transform required.')

    def report(self, text):
        if text != self.status:
            self.get_logger().info(text)
            self.status = text

    def command(self, linear=0., angular=0.):
        if self.stopping or not self.enable_motion:
            linear = angular = 0.
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.robot_frame
        msg.twist.linear.x, msg.twist.angular.z = float(linear), float(angular)
        self.cmd_pub.publish(msg)

    def map_callback(self, msg):
        resolution = msg.info.resolution
        q, p = msg.info.origin.orientation, msg.info.origin.position
        valid = (msg.info.width > 0 and msg.info.height > 0
                 and len(msg.data) == msg.info.width*msg.info.height
                 and math.isfinite(resolution) and resolution > 0
                 and bool(msg.header.frame_id)
                 and all(math.isfinite(v) for v in (p.x,p.y,q.x,q.y,q.z,q.w))
                 and abs(q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w-1) < .01
                 and abs(q.x) < .01 and abs(q.y) < .01)
        if not valid:
            self.grid = None
            self.command()
            self.report('WAIT: invalid map.')
            return
        grid = np.asarray(msg.data, dtype=np.int16).reshape(msg.info.height,msg.info.width)
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
                extrinsic = (0.,0.,0.)
            else:
                tf = self.tf_buffer.lookup_transform(self.robot_frame,msg.header.frame_id,
                                                      Time.from_msg(msg.header.stamp))
                q, t = tf.transform.rotation, tf.transform.translation
                if abs(q.x) > .02 or abs(q.y) > .02:
                    return
                extrinsic = (t.x,t.y,yaw_of(q))
            if all(math.isfinite(v) for v in extrinsic):
                self.points = laser_points(msg,extrinsic,self.inf_is_clear)
        except TransformException:
            pass

    def robot_pose(self):
        try:
            tf = self.tf_buffer.lookup_transform(self.frame,self.robot_frame,Time())
            age = self.get_clock().now().nanoseconds*1e-9-stamp_seconds(tf.header.stamp)
            if not -.1 <= age <= self.tf_timeout:
                return None
            t, q = tf.transform.translation, tf.transform.rotation
            pose = (t.x,t.y,yaw_of(q))
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
            pose.pose.position.x, pose.pose.position.y = map(float,point)
            a, b = (points[i],points[i+1]) if i+1 < len(points) else (points[max(0,i-1)],point)
            yaw = math.atan2(b[1]-a[1],b[0]-a[0])
            pose.pose.orientation.z, pose.pose.orientation.w = math.sin(yaw/2), math.cos(yaw/2)
            msg.poses.append(pose)
        self.path_pub.publish(msg)

    def abandon(self, reason, remember=True):
        self.command()
        if remember and self.goal is not None:
            self.recent_goals.append((self.goal,time.monotonic()+self.retry_delay))
        self.goal, self.waypoints = None, []
        self.blocked_since = self.clear_since = self.invalid_path_since = None
        self.best_distance = math.inf
        self.publish_path([])
        self.next_plan = time.monotonic()+2.
        self.report(reason)

    def cooling(self, point):
        return any(math.dist(point,goal) < .25 for goal,_ in self.recent_goals)

    def planning_grid(self, pose):
        free = (self.grid >= 0) & (self.grid <= self.free_threshold)
        if self.points is not None and len(self.points):
            c, s = math.cos(pose[2]), math.sin(pose[2])
            for x,y in self.points:
                cx,cy = world_to_cell((pose[0]+c*x-s*y,pose[1]+s*x+c*y),self.info)
                if 0 <= cy < free.shape[0] and 0 <= cx < free.shape[1]:
                    free[cy,cx] = False
        distances = clearance_grid(free,self.info.resolution)
        return free, distances, free & (distances > self.robot_radius+self.safety_margin)

    def segment_clear(self, a, b, distances):
        """Continuous clearance bound, including cell offset and sampling gaps.

        Use the FULL configured margin for shortcuts and path revalidation.
        The old checker used a fixed .015 m instead of safety_margin.
        """
        length = math.dist(a,b)
        endpoint_bounds = []
        for point in (a,b):
            x,y = world_to_cell(point,self.info)
            if not (0 <= y < distances.shape[0] and 0 <= x < distances.shape[1]):
                return False
            endpoint_bounds.append(distances[y,x]-math.dist(point,cell_to_world((x,y),self.info)))
        # Distance to an obstacle is 1-Lipschitz. Every point on the
        # segment is within length/2 of at least one endpoint.
        if min(endpoint_bounds)-length/2 > self.robot_radius+self.safety_margin:
            return True
        steps = max(1,math.ceil(length/(self.info.resolution/4)))
        required = self.robot_radius+self.safety_margin+length/steps/2
        # Vectorized samples keep full-route and graph-edge checks inexpensive.
        f = np.linspace(0.,1.,steps+1)
        px, py = a[0]+f*(b[0]-a[0]), a[1]+f*(b[1]-a[1])
        yaw = yaw_of(self.info.origin.orientation)
        c,s = math.cos(yaw),math.sin(yaw)
        dx,dy = px-self.info.origin.position.x,py-self.info.origin.position.y
        lx,ly = c*dx+s*dy,-s*dx+c*dy
        xs,ys = np.floor(lx/self.info.resolution).astype(int),np.floor(ly/self.info.resolution).astype(int)
        if np.any((xs < 0)|(ys < 0)|(xs >= distances.shape[1])|(ys >= distances.shape[0])):
            return False
        offset = np.hypot(lx-(xs+.5)*self.info.resolution,ly-(ys+.5)*self.info.resolution)
        return bool(np.all(distances[ys,xs]-offset > required))

    def find_approaches(self, group, free, reachable, robot):
        """Keep ALL eligible cells at the first usable BFS depth, not scan-order first."""
        limit = math.floor(self.approach_search_distance/self.info.resolution+1e-9)
        queue = deque((int(x),int(y),0) for x,y in group)
        seen, found, found_depth = set(group), set(), None
        while queue:
            x,y,depth = queue.popleft()
            if found_depth is not None and depth > found_depth:
                break
            if reachable[y,x]:
                point = cell_to_world((x,y),self.info)
                if math.dist(point,robot[:2]) >= self.min_goal_distance and not self.cooling(point):
                    found.add((x,y))
                    found_depth = depth
            if depth >= limit or found_depth is not None:
                continue
            for nx,ny in ((x-1,y),(x+1,y),(x,y-1),(x,y+1)):
                if (0 <= ny < free.shape[0] and 0 <= nx < free.shape[1]
                        and free[ny,nx] and (nx,ny) not in seen):
                    seen.add((nx,ny))
                    queue.append((nx,ny,depth+1))
        return found

    def plan(self, pose):
        self.command()
        now = time.monotonic()
        self.recent_goals = [(p,t) for p,t in self.recent_goals if t > now]
        self.next_plan = now+2.
        free,distances,walkable = self.planning_grid(pose)
        start = x,y = world_to_cell(pose[:2],self.info)
        if not (0 <= y < free.shape[0] and 0 <= x < free.shape[1]) or not walkable[y,x]:
            measured = float(distances[y,x]) if 0 <= y < free.shape[0] and 0 <= x < free.shape[1] else 0.
            self.abandon(f'WAIT: robot cell clearance={measured:.3f} m; '
                         f'required={self.robot_radius+self.safety_margin:.3f} m. No automatic relocation.',False)
            self.get_logger().info(self.describe_start(pose,start,free,distances),throttle_duration_sec=5.)
            return
        start_point = cell_to_world(start,self.info)
        if not self.segment_clear(pose[:2],start_point,distances):
            self.abandon('WAIT: actual pose to start-cell center lacks configured clearance.',False)
            return
        # Four-connected components are also valid for the no-corner-cut graph:
        # every permitted diagonal has two walkable cardinal alternatives.
        labels,_ = label(walkable,structure=np.array([[0,1,0],[1,1,1],[0,1,0]]))
        reachable = labels == labels[y,x]
        groups = frontier_groups(self.grid,self.free_threshold)
        if not groups:
            self.abandon('IDLE: no frontiers at the current threshold. Waiting for map updates.',False)
            return
        candidates = set()
        if self.goal is not None:
            previous = gx,gy = world_to_cell(self.goal,self.info)
            near = any(abs(gx-int(cx))+abs(gy-int(cy)) <= self.approach_search_distance/self.info.resolution
                       for group in groups for cx,cy in group)
            if near and 0 <= gy < free.shape[0] and 0 <= gx < free.shape[1] and reachable[gy,gx]:
                candidates.add(previous)
        if not candidates:
            for group in groups:
                candidates.update(self.find_approaches(group,free,reachable,pose))
        # Cache checked undirected edges for this map snapshot. A cheap distance
        # bound proves edges in open space; sample only near boundaries.
        edge_cache = {}
        def edge_ok(a,b):
            key = tuple(sorted((a,b)))
            if key not in edge_cache:
                length = math.dist(a,b)*self.info.resolution
                lower = min(distances[a[1],a[0]],distances[b[1],b[0]])-length/2
                edge_cache[key] = (lower > self.robot_radius+self.safety_margin or
                                  self.segment_clear(cell_to_world(a,self.info),cell_to_world(b,self.info),distances))
            return edge_cache[key]
        cells,cost = route_to_goals(walkable,start,candidates,distances,
                                    self.robot_radius+self.safety_margin,self.clearance_weight,edge_ok)
        if cells:
            raw = [pose[:2]]+[cell_to_world(cell,self.info) for cell in cells]
            points = simplify_path(raw,lambda a,b: self.segment_clear(a,b,distances))
            if points and all(self.segment_clear(a,b,distances) for a,b in zip(points,points[1:])):
                self.goal = cell_to_world(cells[-1],self.info)
                self.waypoints, self.waypoint_index = points[1:],0
                self.planned_version = self.map_version
                self.progress_time,self.best_distance = time.monotonic(),math.inf
                self.replan_count += 1
                self.invalid_path_since = None
                self.publish_path(points)
                length = sum(math.dist(a,b) for a,b in zip(points,points[1:]))
                self.report(f'PLAN: {len(cells)} grid cells -> {len(self.waypoints)} waypoints; '
                            f'path={length:.2f} m; weighted_grid_cost={cost*self.info.resolution:.2f}; '
                            f'goal=({self.goal[0]:.2f}, {self.goal[1]:.2f}); groups={len(groups)}')
                return
        self.goal,self.waypoints = None,[]
        self.publish_path([])
        self.report('WAIT: no eligible goal now; recently attempted areas are cooling down.' if self.recent_goals else
                    'IDLE: frontiers remain but no usable approach path. Mapping is not proven complete.')

    def describe_start(self, pose, start, free, distances):
        x,y = start
        height,width = self.grid.shape
        prefix = (f'START CHECK: pose=({pose[0]:.3f}, {pose[1]:.3f}), cell={start}, '
                  f'map={width}x{height}, resolution={self.info.resolution:.3f}, threshold={self.free_threshold}')
        if not (0 <= x < width and 0 <= y < height):
            return prefix+' | reason=OUTSIDE_MAP'
        value = int(self.grid[y,x])
        raw_free = (self.grid >= 0)&(self.grid <= self.free_threshold)
        raw_clearance = float(clearance_grid(raw_free,self.info.resolution)[y,x])
        reason = ('UNKNOWN_ROBOT_CELL' if value == -1 else
                  'ROBOT_CELL_ABOVE_FREE_THRESHOLD' if not raw_free[y,x] else
                  'LIDAR_ENDPOINT_IN_ROBOT_CELL' if not free[y,x] else 'NEIGHBOR_CLEARANCE_TOO_SMALL')
        nearest = float(np.min(np.hypot(self.points[:,0],self.points[:,1]))) if self.points is not None and len(self.points) else math.inf
        local = self.grid[max(0,y-2):min(height,y+3),max(0,x-2):min(width,x+3)].tolist()
        return (prefix+f' | reason={reason}, occupancy={value}, map_only_clearance={raw_clearance:.3f}, '
                f'with_lidar_clearance={float(distances[y,x]):.3f}, nearest_lidar={nearest:.3f} | local_grid={local}')

    def control_loop(self):
        if self.stopping:
            self.command()
            return
        now = time.monotonic()
        ros_now = self.get_clock().now().nanoseconds*1e-9
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
            self.abandon('Fresh inputs restored; replanning.',False)
            return
        self.last_pose = pose
        if self.episode_anchor is None or math.dist(pose[:2],self.episode_anchor) >= .10:
            self.episode_anchor = pose[:2]
            self.episode_started = now
            self.replan_count = 0
            self.stationary_hold = False
        if self.replan_count >= 4 and now-self.episode_started > 30.:
            self.stationary_hold = True
        if self.stationary_hold:
            self.command()
            self.publish_path([])
            self.report('STOPPED: repeated replans without 0.10 m of progress. Inspect the surroundings before restarting.')
            return
        nearest = float(np.min(np.hypot(self.points[:,0],self.points[:,1]))) if len(self.points) else math.inf
        envelope = self.robot_radius+self.safety_margin
        if nearest <= envelope:
            if not self.envelope_hold:
                self.abandon(f'HOLD: nearest LiDAR point={nearest:.3f} m; required envelope={envelope:.3f} m.',False)
            self.envelope_hold = True
            self.clear_since = None
        if self.envelope_hold:
            self.command()
            if nearest > envelope+.02:
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
        if self.map_version != self.planned_version:
            self.command()
            _,distances,_ = self.planning_grid(pose)
            remaining = self.waypoints[self.waypoint_index:]
            valid = all(self.segment_clear(a,b,distances) for a,b in zip([pose[:2]]+remaining,remaining))
            if not valid:
                if self.invalid_path_since is None:
                    self.invalid_path_since = now
                if now-self.invalid_path_since >= .5 and now >= self.next_plan:
                    self.plan(pose)
                return
            self.invalid_path_since = None
            self.planned_version = self.map_version
        target = self.waypoints[self.waypoint_index]
        tolerance = min(.015,self.info.resolution*.25)
        linear,angular,reached = track_waypoint(pose,target,self.forward_speed,self.turn_speed,tolerance)
        if reached:
            self.command()
            self.waypoint_index += 1
            self.progress_time,self.best_distance = now,math.inf
            if self.waypoint_index >= len(self.waypoints):
                self.abandon('GOAL REACHED: pausing for map updates, then choosing another frontier.')
            return
        wanted_linear = linear
        linear = limit_forward_speed(self.points,linear,self.robot_radius,self.safety_margin)
        blocked = ((wanted_linear > 0 and linear < .008) or
                   not lidar_allows(self.points,linear,angular,self.robot_radius,self.safety_margin))
        if blocked:
            self.command()
            if self.blocked_since is None:
                self.blocked_since = now
            self.clear_since = None
            self.report(f'BLOCKED: forward corridor; nearest point={nearest:.3f} m. Waiting for stable clearance.')
            if now-self.blocked_since >= 3.:
                self.abandon('Obstacle persists; temporarily excluding this goal and replanning.')
            return
        if self.blocked_since is not None:
            self.command()
            if self.clear_since is None:
                self.clear_since = now
            if now-self.clear_since < .5:
                if now-self.blocked_since >= 3.:
                    self.abandon('Intermittent blockage persists; trying another approach.')
                return
            self.blocked_since = None
            self.clear_since = None
        angle = abs(wrap(math.atan2(target[1]-pose[1],target[0]-pose[0])-pose[2]))
        metric = math.dist(pose[:2],target)+.10*angle
        if metric < self.best_distance-.01:
            self.best_distance,self.progress_time = metric,now
        elif now-self.progress_time > 20.:
            self.abandon('No progress for 20 seconds; choosing another frontier.')
            return
        self.report('FOLLOWING: Python waypoint controller.' if self.enable_motion else
                    'PREVIEW: enable_motion is false; all velocity commands are zero.')
        self.command(linear,angular)


def main(args=None):
    rclpy.init(args=args,signal_handler_options=SignalHandlerOptions.NO)
    node = Explorer()
    def request_stop(signum,frame):
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
