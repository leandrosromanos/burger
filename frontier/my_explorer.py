import numpy as np
import rclpy

import math

from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener

from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from scipy.ndimage import label


class Explorer(Node):

    def __init__(self):
        super().__init__('my_explorer')

        # Robot frame whose position we want in the map frame.
        # This can be overridden through a ROS parameter.
        self.declare_parameter('robot_frame', 'base_footprint')
        self.robot_frame = self.get_parameter('robot_frame').value

        # Store transforms received from the ROS TF system.
        self.tf_buffer = Buffer()

        # Subscribe to TF and populate the buffer.
        self.tf_listener = TransformListener(
            self.tf_buffer,
            self
        )

        # Connec of all 8 neighbors
        self.connectivity = np.ones((3, 3), dtype=np.uint8)

        # Receive map messages.
        # Run map_callback whenever a new message arrives.
        self.subscription = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10
        )

        self.get_logger().info('Waiting for map...')
    
    def get_robot_position(self, map_frame):
        try:
            # Return the robot frame's pose expressed in map_frame.
            # Time() requests the latest available transform.
            transform = self.tf_buffer.lookup_transform(
                map_frame,
                self.robot_frame,
                Time()
            )

        except TransformException as error:
            # A missing transform does not mean position (0, 0).
            # Return None so the caller can skip this update.
            self.get_logger().warning(
                f'Robot position unavailable: {error}',
                throttle_duration_sec=5.0
            )
            return None

        robot_x = transform.transform.translation.x
        robot_y = transform.transform.translation.y

        return robot_x, robot_y
    
    def cell_to_map(self, cell_x, cell_y, map_info):
        resolution = map_info.resolution
        origin = map_info.origin

        # Find the cell CENTER in the grid's local coordinates.
        # The origin describes the corner of cell (0, 0).
        local_x = (cell_x + 0.5) * resolution
        local_y = (cell_y + 0.5) * resolution

        # Extract the grid's planar rotation from its quaternion.
        q = origin.orientation

        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )

        # Rotate the local coordinates, then add the origin.
        map_x = (
            origin.position.x
            + math.cos(yaw) * local_x
            - math.sin(yaw) * local_y
        )

        map_y = (
            origin.position.y
            + math.sin(yaw) * local_x
            + math.cos(yaw) * local_y
        )

        return map_x, map_y
    
    def map_callback(self, msg):
        width = msg.info.width
        height = msg.info.height

        # Reject empty maps or inconsistent data.
        if width == 0 or height == 0:
            self.get_logger().warning('Received an empty map.')
            return

        if len(msg.data) != width * height:
            self.get_logger().warning('Invalid map data size.')
            return

        # Convert the flat data into a 2D array.
        # Access a cell using grid[y, x].
        # Signed integers preserve the unknown value (-1).
        grid = np.asarray(
            msg.data,
            dtype=np.int16
        ).reshape((height, width))

        # Mark free cells that touch at least one unknown cell.
        frontier_mask = self.find_frontiers(grid)

        # Group connected frontier cells.
        frontier_groups = self.group_frontiers(frontier_mask)

        # Display the number of cells in each group.
        sizes = [len(group) for group in frontier_groups]

        self.get_logger().info(
            f'Frontier groups: {len(frontier_groups)} | '
            f'Sizes: {sizes}'
        )
        
        # Use the reference frame declared by the map message.
        map_frame = msg.header.frame_id

        if not map_frame:
            self.get_logger().warning('Map frame is empty.')
            return

        robot_position = self.get_robot_position(map_frame)

        if robot_position is None:
            return

        robot_x, robot_y = robot_position

        self.get_logger().info(
            f'Robot in {map_frame}: '
            f'x={robot_x:.2f} m, y={robot_y:.2f} m',
            throttle_duration_sec=2.0
        )

        # Demonstrate coordinate conversion using one frontier cell.
        # This is NOT goal selection or a safety check.
        if frontier_groups:
            cell_x, cell_y = frontier_groups[0][0]

            map_x, map_y = self.cell_to_map(
                cell_x,
                cell_y,
                msg.info
            )

            self.get_logger().info(
                f'Example frontier cell: ({cell_x}, {cell_y}) | '
                f'Center in {map_frame}: '
                f'x={map_x:.2f} m, y={map_y:.2f} m',
                throttle_duration_sec=2.0
            )

    def find_frontiers(self, grid):
        height, width = grid.shape

        # This is a separate boolean array.
        # The original occupancy grid is not modified.
        frontier_mask = np.zeros(
            (height, width),
            dtype=bool
        )

        for y in range(height):
            for x in range(width):

                # For now, only value 0 is accepted as free.
                if grid[y, x] != 0:
                    continue

                neighbors = [
                    (x - 1, y),
                    (x + 1, y),
                    (x, y - 1),
                    (x, y + 1),
                ]

                for nx, ny in neighbors:

                    # Ignore neighbors outside the map.
                    if not (0 <= nx < width and 0 <= ny < height):
                        continue

                    if grid[ny, nx] == -1:
                        # Mark the FREE cell as a frontier.
                        # Do not mark the unknown neighbor.
                        frontier_mask[y, x] = True
                        # One unknown neighbor is enough.
                        break

        return frontier_mask

    def group_frontiers(self, frontier_mask):
        # SciPy assigns an ID to each connected group.
        # 0 = background, 1..group_count = group IDs.
        labels, group_count = label(
            frontier_mask,
            structure=self.connectivity
        )

        groups = []

        for group_id in range(1, group_count + 1):

            # NumPy returns row indices first, then columns.
            rows, columns = np.where(labels == group_id)

            # Store coordinates as (x, y) cell indices.
            group = [
                (int(x), int(y))
                for y, x in zip(rows, columns)
            ]

            groups.append(group)

        return groups


def main(args=None):
    rclpy.init(args=args)

    node = Explorer()

    try:
        # Keep processing incoming ROS messages.
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()