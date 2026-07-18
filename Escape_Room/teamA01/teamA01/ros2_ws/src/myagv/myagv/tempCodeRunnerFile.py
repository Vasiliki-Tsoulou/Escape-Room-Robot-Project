import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
import math

class MockLidarNode(Node):
    def __init__(self):
        super().__init__('mock_lidar_node')
        self.scan_pub = self.create_publisher(LaserScan, '/scan', 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # Τρέχει 10 φορές το δευτερόλεπτο
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.time_elapsed = 0.0
        self.get_logger().info("Mock Lidar Started! The robot is moving in circles...")

    def timer_callback(self):
        self.time_elapsed += 0.1
        
        # 1. Αυτόματη Κίνηση: Το ρομπότ κάνει κύκλους
        radius = 2.0
        speed = 0.5  # ταχύτητα περιστροφής (rad/s)
        robot_x = radius * math.cos(speed * self.time_elapsed)
        robot_y = radius * math.sin(speed * self.time_elapsed)
        robot_theta = speed * self.time_elapsed + (math.pi / 2) # Κοιτάει μπροστά

        # 2. Αποστολή TF (Οδομετρία)
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_footprint'
        t.transform.translation.x = robot_x
        t.transform.translation.y = robot_y
        t.transform.translation.z = 0.0
        
        t.transform.rotation.z = math.sin(robot_theta / 2.0)
        t.transform.rotation.w = math.cos(robot_theta / 2.0)
        self.tf_broadcaster.sendTransform(t)

        # 3. Προσομοίωση Lidar σε τετράγωνο δωμάτιο (10x10)
        scan = LaserScan()
        scan.header.stamp = t.header.stamp
        scan.header.frame_id = 'base_footprint'
        scan.angle_min = 0.0
        scan.angle_max = 2 * math.pi
        scan.angle_increment = math.pi / 180.0  # 1 ακτίνα ανά μοίρα (360 συνολικά)
        scan.range_min = 0.1
        scan.range_max = 12.0
        
        ranges = []
        for i in range(360):
            angle = robot_theta + scan.angle_min + i * scan.angle_increment
            
            # Μαθηματικά για να βρούμε πότε η ακτίνα χτυπάει τους 4 τοίχους (στα ±5 μέτρα)
            dist_x1 = (5.0 - robot_x) / math.cos(angle) if math.cos(angle) > 0.001 else 999.0
            dist_x2 = (-5.0 - robot_x) / math.cos(angle) if math.cos(angle) < -0.001 else 999.0
            dist_y1 = (5.0 - robot_y) / math.sin(angle) if math.sin(angle) > 0.001 else 999.0
            dist_y2 = (-5.0 - robot_y) / math.sin(angle) if math.sin(angle) < -0.001 else 999.0
            
            # Κρατάμε την κοντινότερη θετική απόσταση
            valid_dists = [d for d in [dist_x1, dist_x2, dist_y1, dist_y2] if d > 0]
            ranges.append(min(valid_dists) if valid_dists else 12.0)

        scan.ranges = ranges
        self.scan_pub.publish(scan)

def main(args=None):
    rclpy.init(args=args)
    node = MockLidarNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()