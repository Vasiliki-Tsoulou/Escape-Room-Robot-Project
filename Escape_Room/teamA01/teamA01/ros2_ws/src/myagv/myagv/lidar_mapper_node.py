import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import TransformStamped
from tf2_ros import Buffer, TransformListener, TransformBroadcaster
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException
from myagv.raycasting import update_occupancy_grid
import numpy as np
import math
from scipy.ndimage import maximum_filter
from myagv.icp import laser_scan_to_points, icp_2d, transform_points
from std_msgs.msg import Int32
from std_msgs.msg import Float32MultiArray

def disk(radius):
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    return x*x + y*y <= radius*radius


def get_yaw_from_quaternion(q):
    """Βοηθητική συνάρτηση για μετατροπή Quaternion σε γωνία"""
    siny_cosp = 2 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)

def get_quaternion_from_yaw(yaw):
    """
    Μετατρέπει μια γωνία (Yaw) σε 3D Quaternion (x, y, z, w)"""
    x = 0.0
    y = 0.0
    z = math.sin(yaw / 2.0)
    w = math.cos(yaw / 2.0)
    return x, y, z, w

class LidarMapperNode(Node):
    def __init__(self):
        super().__init__('lidar_mapper_node')

        self.map_resolution = 0.05  # 5 εκατοστά ανά κελί
        self.map_width = 200        # 10m
        self.map_height = 200       # 10m
        
        # Τοποθετούμε το (0,0) του πραγματικού κόσμου στο κέντρο του πλέγματος
        self.origin_x = -(self.map_width * self.map_resolution) / 2.0
        self.origin_y = -(self.map_height * self.map_resolution) / 2.0
        
        # Δημιουργία κενού χάρτη με -1 (Άγνωστος Χώρος)
        self.grid = np.full((self.map_height, self.map_width), -1, dtype=np.int8)


        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, qos_profile_sensor_data)
        self.map_pub = self.create_publisher(OccupancyGrid, '/map', 10)
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # Τυπώνουμε τον χάρτη στο RViz κάθε 0.5 δευτερόλεπτα
        self.timer = self.create_timer(0.5, self.publish_map)

        self.previous_points = None

        self.prev_odom_x = None
        self.prev_odom_y = None
        self.prev_odom_theta = None

        self.icp_robot_x = 0.0
        self.icp_robot_y = 0.0
        self.icp_robot_theta = 0.0

        # αντιστοίχηση κλειδιών και πορτών 0-100 κλειδια,100+ πορτες
        self.key_to_door = {0: 10, 1: 11, 2: 12}
        
        # Μνήμη Ρομπότ
        self.found_keys = set()       # Κρατάει τα IDs των κλειδιών που βρήκαμε
        self.known_doors = {}         # Μορφή: {door_id: [(x1,y1), (x2,y2),..]}
        self.unlocked_doors = set()   # Κρατάει τα IDs των πορτών που έχουν ανοίξει
        
        # Subscriber
        self.vision_sub = self.create_subscription(
            Float32MultiArray,
            '/vision/marker_info',
            self.marker_callback,
            qos_profile_sensor_data
        )

    def marker_callback(self, msg):
        # Διάβασμα δεδομένων από την Κάμερα
        marker_id = int(msg.data[0])
        distance = msg.data[1]  # Απόσταση σε μέτρα
        angle = msg.data[2]     # Γωνία σε σχέση με την κάμερα
        
        # Για Marker στον χαγκόσμιο χάρτη υπολογίζουμε την απόλυτη γωνία προσθέτοντας τη γωνία του ρομπότ με τη γωνία που το είδε η κάμερα
        global_angle = self.icp_robot_theta + angle
        
        marker_global_x = self.icp_robot_x + distance * math.cos(global_angle)
        marker_global_y = self.icp_robot_y + distance * math.sin(global_angle)
        
        # Μετατροπή των μέτρων σε συντεταγμένες κελιών του Grid
        grid_x = int((marker_global_x - self.origin_x) / self.map_resolution)
        grid_y = int((marker_global_y - self.origin_y) / self.map_resolution)
        
        # Βεβαιωνόμαστε ότι το marker είναι μέσα στα όρια του χάρτη
        if not (0 <= grid_x < self.map_width and 0 <= grid_y < self.map_height):
            return

        # Κλειδί (0-99) ή Πόρτα (100+)
        if marker_id < 100:
            # κλειδί
            if marker_id not in self.found_keys:
                self.found_keys.add(marker_id)
                self.get_logger().info(f" Βρέθηκε Κλειδί {marker_id}!")
                
                # Ελέγχουμε αν έχουμε ήδη δει την πόρτα του
                target_door = self.key_to_door.get(marker_id)
                if target_door in self.known_doors and target_door not in self.unlocked_doors:
                    self.unlock_door(target_door)
                    
        else:
            # πόρτα
            if marker_id not in self.known_doors:
                self.get_logger().info(f"Εντοπίστηκε Πόρτα {marker_id} στον χάρτη!")
                
                # Υποθέτουμε ότι η πόρτα πιάνει έναν χώρο 5x5 κελιών γύρω από το marker
                door_cells = []
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        door_cells.append((grid_x + dx, grid_y + dy))
                self.known_doors[marker_id] = door_cells
                
                # Ελέγχουμε αν έχουμε το κλειδί
                # Βρίσκουμε ποιο κλειδί ανοίγει αυτή την πόρτα
                required_key = next((k for k, v in self.key_to_door.items() if v == marker_id), None)
                if required_key in self.found_keys:
                    self.unlock_door(marker_id)

    def unlock_door(self, door_id):
        """Βοηθητική συνάρτηση που ρίχνει τα εμπόδια από τον χάρτη."""
        self.get_logger().info(f" Ξεκλείδωμα Πόρτας {door_id}!")
        self.unlocked_doors.add(door_id)
        for (cx, cy) in self.known_doors[door_id]:
            if 0 <= cx < self.map_width and 0 <= cy < self.map_height:
                self.grid[cy, cx] = 0  # 0 = Ελεύθερο

    def scan_callback(self, msg):
        try:
            # Διαβάζουμε τους Τροχούς (TF) αγνοώντας το clock skew της γέφυρας
            trans = self.tf_buffer.lookup_transform('odom', 'base_footprint', rclpy.time.Time())
            odom_x = trans.transform.translation.x
            odom_y = trans.transform.translation.y
            odom_theta = get_yaw_from_quaternion(trans.transform.rotation)

            current_points = laser_scan_to_points(
                msg.ranges, msg.angle_min, msg.angle_increment, 
                msg.range_min, msg.range_max
            )
            corrected_angle_min = msg.angle_min + math.pi
            
            # Ωμά δεδομένα για το raycasting, απαραίτητα για να μην σβήνει ο χάρτης στο κενό/στα τυφλά σημεία
            raw_ranges = list(msg.ranges)

            # Πρώτο σκανάρισμα
            if self.previous_points is None or self.prev_odom_x is None:
                # Φιλτράρισμα και στο πρώτο frame για το ICP
                filtered_ranges = [r if (msg.range_min < r < 5.0 and not math.isinf(r) and not math.isnan(r)) else 0.0 for r in msg.ranges]
                self.previous_points = laser_scan_to_points(
                    filtered_ranges, corrected_angle_min, msg.angle_increment, 
                    msg.range_min, msg.range_max
                )
                self.previous_points = current_points
                self.prev_odom_x = odom_x
                self.prev_odom_y = odom_y
                self.prev_odom_theta = odom_theta
                d_odom_theta = 0.0
                # Θέλουμε να τρέξει το raycasting από την πρώτη στιγμή
                filtered_ranges_for_map = filtered_ranges
            else:
                # Υπολογίζουμε πόσο μετακινήθηκαν οι τροχοί από το προηγούμενο scan
                d_odom_x = odom_x - self.prev_odom_x
                d_odom_y = odom_y - self.prev_odom_y
                d_odom_theta = odom_theta - self.prev_odom_theta

                # --- LIDAR FILTERING (Καθαρισμός Φαντασμάτων - 3.5 μέτρα) ---
                MAX_RELIABLE_DISTANCE = 3.5

                filtered_ranges = [
                    r if (msg.range_min < r < MAX_RELIABLE_DISTANCE and not math.isinf(r) and not math.isnan(r)) else 0.0 
                    for r in msg.ranges
                ]
                filtered_ranges_for_map = filtered_ranges

                if abs(d_odom_x) < 0.01 and abs(d_odom_y) < 0.01 and abs(d_odom_theta) < 0.02:
                    self.publish_transform(odom_x, odom_y, odom_theta)

                else:
                    # Το ρομπότ κινήθηκε. Τρέχουμε το ICP κανονικά.
                    current_points = laser_scan_to_points(
                        filtered_ranges, corrected_angle_min, msg.angle_increment,
                        msg.range_min, msg.range_max
                    )

                    # Μετατροπή της παγκόσμιας κίνησης στο τοπικό σύστημα του ρομπότ
                    cos_t = math.cos(self.prev_odom_theta)
                    sin_t = math.sin(self.prev_odom_theta)
                    guess_dx = d_odom_x * cos_t + d_odom_y * sin_t
                    guess_dy = -d_odom_x * sin_t + d_odom_y * cos_t
                    guess_dtheta = d_odom_theta

                    # Δίνουμε τη μαντεψιά στο ICP.
                    dx, dy, dtheta = icp_2d(current_points, self.previous_points, initial_guess=(guess_dx, guess_dy, guess_dtheta))

                    # Μετατροπή της τελικής (διορθωμένης) κίνησης πίσω στο παγκόσμιο σύστημα
                    global_dx = dx * math.cos(self.icp_robot_theta) - dy * math.sin(self.icp_robot_theta)
                    global_dy = dx * math.sin(self.icp_robot_theta) + dy * math.cos(self.icp_robot_theta)
                    
                    # Ενημερώνουμε την τέλεια θέση του ρομπότ
                    self.icp_robot_x += global_dx
                    self.icp_robot_y += global_dy
                    self.icp_robot_theta += dtheta

                    self.publish_transform(odom_x, odom_y, odom_theta)

                    # Αποθηκεύουμε τα τωρινά δεδομένα για την επόμενη φορά
                    self.previous_points = current_points
                    self.prev_odom_x = odom_x
                    self.prev_odom_y = odom_y
                    self.prev_odom_theta = odom_theta

            # Ζωγραφίζουμε τον χάρτη!
            # Αν η στροφή είναι μικρότερη από 0.01 rad ανά frame, ζωγράφισε.
            # Αλλιώς, κράτα τον χάρτη "παγωμένο" όπως ήταν.
            is_rotating = abs(d_odom_theta) > 0.01

            if not is_rotating:
                unlocked_cells_set = set()
                for door_id in self.unlocked_doors:
                    if door_id in self.known_doors:
                        for cell in self.known_doors[door_id]:
                            unlocked_cells_set.add(cell)

                # Περνάμε τα raw_ranges
                self.grid = update_occupancy_grid(
                    self.grid, 
                    self.map_resolution, 
                    self.origin_x, 
                    self.origin_y,
                    self.icp_robot_x, 
                    self.icp_robot_y, 
                    self.icp_robot_theta,
                    raw_ranges, 
                    corrected_angle_min, 
                    msg.angle_increment, 
                    msg.range_min, 
                    msg.range_max,
                    unlocked_cells_set
                )

        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warn(f"TF Error: {e}")
            return


    def publish_map(self):
        robot_radius_cells = 3
        inflated_grid = maximum_filter(self.grid, footprint=disk(robot_radius_cells))

        # --- Robot Footprint Clearing ---
        # Βρίσκουμε πού είναι το ρομπότ στον χάρτη (σε κελιά)
        rx_grid = int((self.icp_robot_x - self.origin_x) / self.map_resolution)
        ry_grid = int((self.icp_robot_y - self.origin_y) / self.map_resolution)

        # θετουμε μια κυκλική περιοχή γύρω από το ρομπότ για να μπορεί ο A* να ξεκινήσει
        for dx in range(-4, 5):
            for dy in range(-4, 5):
                # Κυκλικός καθαρισμός ακτίνας 4 κελιών
                if dx*dx + dy*dy <= 16: 
                    cx = rx_grid + dx
                    cy = ry_grid + dy
                    if 0 <= cx < self.map_width and 0 <= cy < self.map_height:
                        inflated_grid[cy, cx] = 0

        
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map' 
        
        msg.info.resolution = self.map_resolution
        msg.info.width = self.map_width
        msg.info.height = self.map_height
        
        msg.info.origin.position.x = self.origin_x
        msg.info.origin.position.y = self.origin_y
        
        # Μετατροπή του 2D Numpy πίνακα πίσω σε 1D Λίστα
        msg.data = inflated_grid.flatten().tolist()
        
        self.map_pub.publish(msg)

    def publish_transform(self, odom_x, odom_y, odom_theta):
        """Εκπέμπει συνεχώς το TF map -> odom για να μην κρασάρει το RViz"""
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'map'
        t.child_frame_id = 'odom'

        # Η μετάφραση είναι η διαφορά μεταξύ του τέλειου ICP και των τροχών
        t.transform.translation.x = self.icp_robot_x - odom_x
        t.transform.translation.y = self.icp_robot_y - odom_y

        # Η διαφορά της γωνίας
        theta_diff = self.icp_robot_theta - odom_theta
        qx, qy, qz, qw = get_quaternion_from_yaw(theta_diff)

        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw

        self.tf_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = LidarMapperNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
