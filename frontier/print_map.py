import numpy as np
import rclpy

from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node


class MapPrinter(Node):

    def __init__(self):
        super().__init__('map_printer')

        self.printed = False

        self.subscription = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10
        )

        self.get_logger().info('Waiting for map...')

    def map_callback(self, msg):
        width = msg.info.width
        height = msg.info.height

        # Wait for a non-empty map with consistent dimensions.
        if width == 0 or height == 0:
            return

        if len(msg.data) != width * height:
            self.get_logger().warning('Invalid map data size.')
            return

        # Convert the flat map data into rows and columns.
        # Access an individual cell with grid[y, x].
        grid = np.asarray(
            msg.data,
            dtype=np.int16
        ).reshape((height, width))

        print(
            f'\nMap: {width} columns x {height} rows\n'
            f'Resolution: {msg.info.resolution} m/cell\n'
            f'Frame: {msg.header.frame_id}\n'
            'Rows: highest y first, y=0 last\n'
            'Columns: x=0 first, increasing to the right\n',
            flush=True
        )

        # Display every value, without replacing cells with "...".
        # Reverse the displayed rows so y=0 appears at the bottom.
        # This does not modify the original grid.
        with np.printoptions(threshold=np.inf, linewidth=200):
            print(grid[::-1], flush=True)

        self.printed = True


def main(args=None):
    rclpy.init(args=args)
    node = MapPrinter()

    try:
        # Process messages until one valid map has been printed.
        while rclpy.ok() and not node.printed:
            rclpy.spin_once(node, timeout_sec=0.5)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()