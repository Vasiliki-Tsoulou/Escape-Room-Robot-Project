import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, Twist
import math
import numpy as np
from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException
from myagv.a_star_planner import a_star_search

class AStarPlannerNode(Node):
    def __init__(self):
        super().__init__('a_star_planner_node')
        
        # Subscriptions
        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)
        self.goal_sub = self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)
        
        # Publisher
        self.path_pub = self.create_publisher(Path, '/planned_path', 10)
        
        # State variables
        self.current_grid = None
        self.grid_info = None

        # Ο buffer αποθηκεύει το ιστορικό των θέσεων
        self.tf_buffer = Buffer()
        # γεμισμα buffer
        self.tf_listener = TransformListener(self.tf_buffer, self)


    def map_callback(self, msg):
        """Converts the 1D ROS map array into a 2D numpy grid for A*."""
        self.grid_info = msg.info
        width = msg.info.width
        height = msg.info.height
        
        # Μετατροπή σε NumPy array και Reshape (Y, X)
        grid_array = np.array(msg.data, dtype=np.int8).reshape((height, width))
        transposed_grid = grid_array.T      # για το RViz
        
        # Εντοπισμος Εμποδιων
        self.current_grid = np.where((transposed_grid > 50) | (transposed_grid == -1), 1, 0) 
        
        self.get_logger().info(f"Map ready! Shape: {self.current_grid.shape}")

    def get_robot_position_in_grid(self):
        """Διαβάζει το TF2 για να βρει το ρομπότ και επιστρέφει συντεταγμένες πλέγματος (X, Y)."""
        try:
            # σχέση μεταξύ του χάρτη - βάσης ρομπότ
            trans = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
            
            # Πραγματικές συντεταγμένες (μέτρα)
            robot_x = trans.transform.translation.x
            robot_y = trans.transform.translation.y
            
            # Μετατροπή σε συντεταγμένες grid (κελιά)
            res = self.grid_info.resolution
            origin_x = self.grid_info.origin.position.x
            origin_y = self.grid_info.origin.position.y
            
            start_x_idx = int((robot_x - origin_x) / res)
            start_y_idx = int((robot_y - origin_y) / res)
            
            return (start_x_idx, start_y_idx)
            
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warn(f"Could not find robot transform: {e}")
            return None

    def goal_callback(self, msg):
        """Triggered when you click '2D Goal Pose' in RViz."""
        if self.current_grid is None:
            self.get_logger().warn("Cannot plan path: No map received yet.")
            return
        
        # αφετερία
        start_coords = self.get_robot_position_in_grid()
        if start_coords is None:
            self.get_logger().warn("Aborting planning: Don't know where the robot is!")
            return
        
        # στόχος
        res = self.grid_info.resolution
        origin_x = self.grid_info.origin.position.x
        origin_y = self.grid_info.origin.position.y
        
        goal_x_idx = int((msg.pose.position.x - origin_x) / res)
        goal_y_idx = int((msg.pose.position.y - origin_y) / res)
        goal_coords = (goal_x_idx, goal_y_idx)
        
        self.get_logger().info(f"Planning from {start_coords} to {goal_coords}...")
        
        # Run A*
        path_indices = a_star_search(self.current_grid, start_coords, goal_coords, weight=1.5)
        
        if not path_indices:
            self.get_logger().warn("A* failed to find a valid path!")
            return
        
        self.publish_path(path_indices)

    def publish_path(self, path_indices):
        """Converts A* grid indices back to real-world meters for RViz."""
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = "map"
        
        res = self.grid_info.resolution
        origin_x = self.grid_info.origin.position.x
        origin_y = self.grid_info.origin.position.y
        
        for (gx, gy) in path_indices:
            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.header.stamp = path_msg.header.stamp
            
            # Convert grid back to world coordinates
            pose.pose.position.x = (gx * res) + origin_x + (res / 2.0)
            pose.pose.position.y = (gy * res) + origin_y + (res / 2.0)
            
            path_msg.poses.append(pose)
            
        self.path_pub.publish(path_msg)
        self.get_logger().info("Path published to RViz!")


def main(args=None):
    rclpy.init(args=args)
    node = AStarPlannerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()