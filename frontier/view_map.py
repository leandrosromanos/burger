import numpy as np
import matplotlib.pyplot as plt
import rclpy

from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node


class MapViewer(Node):

    def __init__(self):
        super().__init__('map_viewer')

        self.grid = None
        self.resolution = None

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

        # Ignore empty or incorrectly sized maps.
        if width == 0 or height == 0:
            return

        if len(msg.data) != width * height:
            self.get_logger().warning('Invalid map data size.')
            return

        # Copy the map into a 2D array.
        # Access cells using grid[y, x].
        self.grid = np.array(
            msg.data,
            dtype=np.int16
        ).reshape((height, width))

        self.resolution = msg.info.resolution

        self.get_logger().info('Map received.')


def show_map(grid, resolution):
    # Mask unknown cells so they receive a separate color.
    displayed_grid = np.ma.masked_where(grid == -1, grid)

    # Occupancy 0 is white; occupancy 100 is black.
    # Intermediate values appear as shades of gray.
    color_map = plt.get_cmap('gray_r').copy()
    color_map.set_bad(color='red')

    fig, ax = plt.subplots(figsize=(10, 8))

    image = ax.imshow(
        displayed_grid,
        cmap=color_map,
        vmin=0,
        vmax=100,
        origin='lower',
        interpolation='nearest',
        aspect='equal'
    )

    # Axes use cell indices, not coordinates in meters.
    ax.set_xlabel('Cell x')
    ax.set_ylabel('Cell y')
    ax.set_title(
        f'Occupancy map | Resolution: {resolution:.3f} m/cell\n'
        'Unknown: gray | Free: white | Occupied: black'
    )

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label('Occupancy value (known cells)')

    # Show the exact cell value when hovering over the map.
    def format_coordinates(x, y):
        cell_x = int(np.floor(x + 0.5))
        cell_y = int(np.floor(y + 0.5))

        height, width = grid.shape

        if 0 <= cell_x < width and 0 <= cell_y < height:
            value = int(grid[cell_y, cell_x])
            description = 'unknown' if value == -1 else str(value)

            return (
                f'Cell x={cell_x}, y={cell_y} | '
                f'Value={description}'
            )

        return ''

    ax.format_coord = format_coordinates

    fig.tight_layout()
    plt.show()


def main(args=None):
    rclpy.init(args=args)
    node = MapViewer()

    grid = None
    resolution = None

    try:
        # Wait for one valid map.
        while rclpy.ok() and node.grid is None:
            rclpy.spin_once(node, timeout_sec=0.5)

        grid = node.grid
        resolution = node.resolution

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()

    # Open the plot after closing the ROS node.
    if grid is not None:
        show_map(grid, resolution)


if __name__ == '__main__':
    main()