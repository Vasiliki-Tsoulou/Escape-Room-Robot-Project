import rclpy
from rclpy.node import Node
import math
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray
from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException
import numpy as np
from nav_msgs.msg import OccupancyGrid

class RobotState:
    EXPLORE_FOR_MARKERS = 0
    NAVIGATE_TO_DOOR = 1
    EXPLORE_DOOR = 2
    RETURN_HOME = 3
    OVERRIDE = 4

class TaskManagerNode(Node):
    def __init__(self):
        super().__init__('task_manager_node')

        # --- State Configuration ---
        self.current_state = RobotState.EXPLORE_FOR_MARKERS
        self.saved_state = None
        self.home_pose = (0.0, 0.0, 0.0)
        
        # --- Tracking Variables ---
        self.found_keys = set()
        self.known_doors = {}        # door_id: (x, y)
        self.explored_doors = set()  
        self.active_target_door = None

        # --- Goal commit / arrival tracking ---
        self.active_goal = None        # (x, y) στόχος που εκτελούμε τώρα
        self.goal_start_time = None    # για timeout
        self.goal_timeout = 5.0       # δευτ. πριν εγκαταλείψουμε έναν στόχο
        self.blacklisted_goals = []
        self.arrival_tol = 0.2         # μέτρα: πότε θεωρούμε ότι φτάσαμε
        self.key_to_door = {
            0: 10,
            1: 11,
            2: 12
        }

        self.grid = None
        self.map_resolution = 0.05
        self.origin_x = 0.0
        self.origin_y = 0.0
        self.map_width = 0
        self.map_height = 0

        # Ο Subscriber που ακούει τον Mapper
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10
        )
        # --- Override Specific Targets ---
        self.override_x = None
        self.override_y = None
        self.override_theta = None

        # TF & Navigation Setup
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)

        # Vision subscriber
        self.vision_sub = self.create_subscription(
            Float32MultiArray,
            '/vision/marker_info',
            self.marker_callback,
            10
        )

        # Manual override subscriber topic for Goal setup using RViz
        self.override_sub = self.create_subscription(
            PoseStamped,
            '/sys_override/goal_pose',
            self.override_callback,
            10
        )

        # timer to check state
        self.decision_timer = self.create_timer(1, self.execute_state_machine)
        self.get_logger().info(" Task-Level Decision Maker Online. State: EXPLORING FOR MARKERS.")

    def map_callback(self, msg):
        """Saves the map data from the LidarMapperNode for frontier extraction"""
        # Save map resolution and origins
        self.map_resolution = msg.info.resolution
        self.origin_x = msg.info.origin.position.x
        self.origin_y = msg.info.origin.position.y
        self.map_width = msg.info.width
        self.map_height = msg.info.height
        
        # Convert 1D list back to 2D Numpy array matching the mapper
        self.grid = np.array(msg.data, dtype=np.int8).reshape((self.map_height, self.map_width))

    def find_closest_frontier(self, center=None, radius=None):
        """Βρίσκει το πλησιέστερο σύνορο free/unknown."""
        if self.grid is None:
            return None, None
        rx, ry, _ = self.get_robot_pose()
        if rx is None:
            return None, None

        free = (self.grid == 0)
        unknown = (self.grid == -1)

        # Βρίσκουμε όλα τα υποψήφια frontiers
        frontier = np.zeros_like(free)
        frontier[1:-1, 1:-1] = free[1:-1, 1:-1] & (
            unknown[2:, 1:-1] | unknown[:-2, 1:-1] |
            unknown[1:-1, 2:] | unknown[1:-1, :-2]
        )

        # =====================================================================
        # FLOOD-FILL REACHABILITY CHECK
        # =====================================================================
        # Υπολογισμός της θέσης του ρομπότ σε συντεταγμένες πλέγματος (grid indices)
        start_x = int((rx - self.origin_x) / self.map_resolution)
        start_y = int((ry - self.origin_y) / self.map_resolution)
        
        reachable = np.zeros_like(self.grid, dtype=bool)
        
        # Ελέγχουμε αν το ρομπότ είναι εντός του χάρτη
        if 0 <= start_x < self.map_width and 0 <= start_y < self.map_height:
            from collections import deque
            q = deque([(start_x, start_y)])
            reachable[start_y, start_x] = True
            
            # Ψάχνουμε οριζόντια, κάθετα και διαγώνια (8-Way Connectivity)
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]
            
            # Απλώνουμε το BFS μόνο πάνω στα ελεύθερα κελιά
            while q:
                cx, cy = q.popleft()
                for dx, dy in directions:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < self.map_width and 0 <= ny < self.map_height:
                        if not reachable[ny, nx] and self.grid[ny, nx] == 0:
                            reachable[ny, nx] = True
                            q.append((nx, ny))

        # Κρατάμε τα frontiers που ανήκουν στα προσβάσιμα κελιά
        valid_frontiers = frontier & reachable
        # =====================================================================

        # Συνεχίζουμε τη λογική σου χρησιμοποιώντας πλέον μόνο τα valid_frontiers
        ys, xs = np.where(valid_frontiers)   
        if len(xs) == 0:
            return None, None

        # Κέντρο κελιού (συνεπές με τον A* publish_path)
        fx = self.origin_x + xs * self.map_resolution + self.map_resolution / 2.0
        fy = self.origin_y + ys * self.map_resolution + self.map_resolution / 2.0

        if center is not None and radius is not None:
            mask = np.hypot(fx - center[0], fy - center[1]) <= radius
            fx, fy = fx[mask], fy[mask]
            if len(fx) == 0:
                return None, None

        dists = np.hypot(fx - rx, fy - ry)

        min_allowable_dist = self.arrival_tol + 0.15
        valid_mask = dists > min_allowable_dist

        if not np.any(valid_mask):
            return None, None

        fx = fx[valid_mask]
        fy = fy[valid_mask]
        dists = dists[valid_mask]

        # =====================================================================
        # ΦΙΛΤΡΑΡΙΣΜΑ BLACKLIST
        # Πετάμε τα frontiers που είναι σε ακτίνα 30cm από αποτυχημένο στόχο
        # =====================================================================
        if self.blacklisted_goals:
            not_blacklisted_mask = np.ones(len(fx), dtype=bool)
            for bx, by in self.blacklisted_goals:
                dist_to_bad = np.hypot(fx - bx, fy - by)
                not_blacklisted_mask &= (dist_to_bad > 0.30) # Ακτίνα 30 cm
            
            fx = fx[not_blacklisted_mask]
            fy = fy[not_blacklisted_mask]
            dists = dists[not_blacklisted_mask]

            if len(fx) == 0:
                return None, None
        # =====================================================================

        i = int(np.argmin(dists))
        return float(fx[i]), float(fy[i])
    
    def get_robot_pose(self):
        try:
            trans = self.tf_buffer.lookup_transform('map', 'base_footprint', rclpy.time.Time())
            # Extract yaw alongside positions
            q = trans.transform.rotation
            siny_cosp = 2 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
            yaw = math.atan2(siny_cosp, cosy_cosp)
            return trans.transform.translation.x, trans.transform.translation.y, yaw
        except (LookupException, ConnectivityException, ExtrapolationException):
            return None, None, None

    def override_callback(self, msg):
        """Triggers immediately when a target is received on the override topic"""
        # If we are already in override, don't clobber the original saved state
        if self.current_state != RobotState.OVERRIDE:
            self.saved_state = self.current_state
            
        self.override_x = msg.pose.position.x
        self.override_y = msg.pose.position.y
        
        # Pull yaw orientation
        q = msg.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.override_theta = math.atan2(siny_cosp, cosy_cosp)

        self.current_state = RobotState.OVERRIDE
        self.get_logger().warn(f"OVERRIDE ACTIVATED! Rerouting instantly to: ({self.override_x:.2f}, {self.override_y:.2f})")
        
        # Actively push the new command down to the velocity controller immediately
        self.send_navigation_goal(self.override_x, self.override_y, self.override_theta)

    def marker_callback(self, msg):
        marker_id = int(msg.data[0])
        if marker_id < 100:
            if marker_id not in self.found_keys:
                self.found_keys.add(marker_id)
        else:
            if marker_id not in self.known_doors:
                rx, ry, _ = self.get_robot_pose()
                if rx is not None:
                    self.known_doors[marker_id] = (rx, ry)###apothhkeuei portees me tis syntetagmenes tou

    def execute_state_machine(self):
        rx, ry, rtheta = self.get_robot_pose()
        if rx is None:
            return 

        # --- STATE 4: OVERRIDE HANDLING ---
        if self.current_state == RobotState.OVERRIDE:
            self.commit_goal(self.override_x, self.override_y, self.override_theta)
            if self.reached(self.override_x, self.override_y):
                self.get_logger().info(f"Override reached. Resuming state: {self.saved_state}")
                self.clear_goal()
                self.current_state = self.saved_state if self.saved_state is not None else RobotState.EXPLORE_FOR_MARKERS
                self.saved_state = None
            elif self.goal_timed_out():
                self.get_logger().warn("Override unreachable (timeout). Resuming previous task.")
                self.clear_goal()
                self.current_state = self.saved_state if self.saved_state is not None else RobotState.EXPLORE_FOR_MARKERS
                self.saved_state = None
            return

        # --- STATE 0: ΑΡΧΙΚΗ ΑΥΤΟΝΟΜΗ ΕΞΕΡΕΥΝΗΣΗ (FRONTIER) ---
        if self.current_state == RobotState.EXPLORE_FOR_MARKERS:
            if self.grid is None:
                self.get_logger().info("Waiting for map topic to publish data...", throttle_duration_sec=2.0)
                return

            # Έχουμε ήδη στόχο: περίμενε άφιξη ή timeout
            if self.active_goal is not None:
                gx, gy = self.active_goal
                dist_remaining = math.hypot(gx - rx, gy - ry)
                if self.reached(gx, gy) or self.goal_timed_out():
                    status = "TIMED OUT" if self.goal_timed_out() else "REACHED"
                    self.get_logger().info(f"Goal {self.active_goal} {status}. Finding next frontier...")
                    self.clear_goal()   # στον επόμενο κύκλο διαλέγουμε νέο frontier
                else:
                     # διαγνωστικό log
                    self.get_logger().info(
                        f"Moving to goal {self.active_goal}. Remaining: {dist_remaining:.2f}m (Target Tol: {self.arrival_tol}m)", 
                        throttle_duration_sec=2.0
                    )
                    return

            # Δεν έχουμε στόχο, διάλεξε νέο frontier
            frontier_x, frontier_y = self.find_closest_frontier()
            if frontier_x is not None:
                angle = math.atan2(frontier_y - ry, frontier_x - rx)
                self.commit_goal(frontier_x, frontier_y, angle)
                self.get_logger().info("State 0: Exploring map using frontier exploration...", throttle_duration_sec=5.0)
            else:
                self.get_logger().info(
                    f"Initial exploration complete. Found {len(self.found_keys)} keys and {len(self.known_doors)} doors."
                )
                self.clear_goal()
                self.current_state = RobotState.NAVIGATE_TO_DOOR
        # --- STATE 1: NAVIGATE TO THE NEAREST UNLOCKED DOOR ---
        elif self.current_state == RobotState.NAVIGATE_TO_DOOR:

            # Ήδη πηγαίνουμε σε πόρτα -> περίμενε άφιξη
            if self.active_target_door is not None and self.active_goal is not None:
                gx, gy = self.active_goal
                if self.reached(gx, gy):
                    self.get_logger().info(f"Arrived at Door {self.active_target_door}. Exploring behind it.")
                    self.clear_goal()
                    self.current_state = RobotState.EXPLORE_DOOR
                elif self.goal_timed_out():
                    self.get_logger().warn(f"Could not reach Door {self.active_target_door}. Skipping it.")
                    self.explored_doors.add(self.active_target_door)
                    self.active_target_door = None
                    self.clear_goal()
                return

            # Διάλεξε νέα προσβάσιμη πόρτα
            accessible_doors = {}
            for key_id, door_id in self.key_to_door.items():
                if key_id in self.found_keys and door_id in self.known_doors and door_id not in self.explored_doors:
                    accessible_doors[door_id] = self.known_doors[door_id]

            if not accessible_doors:
                self.get_logger().info("No unexplored doors with available keys remain. Returning home...")
                self.clear_goal()
                self.active_target_door = None
                self.current_state = RobotState.RETURN_HOME
                return

            closest_door_id = None
            min_distance = float('inf')
            for door_id, (ddx, ddy) in accessible_doors.items():
                dist = math.hypot(ddx - rx, ddy - ry)
                if dist < min_distance:
                    min_distance = dist
                    closest_door_id = door_id

            self.active_target_door = closest_door_id
            ddx, ddy = self.known_doors[closest_door_id]
            self.commit_goal(ddx, ddy, 0.0)
            self.get_logger().info(f"Navigating to Door {closest_door_id} (Key Available)")

        # --- STATE 2: ΕΞΕΡΕΥΝΗΣΗ ΛΑΒΥΡΙΝΘΟΥ ΠΟΡΤΑΣ (FRONTIER) ---
        elif self.current_state == RobotState.EXPLORE_DOOR:
            door_center = self.known_doors.get(self.active_target_door)
            if door_center is None:
                self.clear_goal()
                self.current_state = RobotState.NAVIGATE_TO_DOOR
                return

            # Έχουμε στόχο -> περίμενε άφιξη/timeout
            if self.active_goal is not None:
                gx, gy = self.active_goal
                if self.reached(gx, gy) or self.goal_timed_out():
                    self.clear_goal()
                return

            # Frontier κοντά στην πόρτα (ακτίνα 2.0 m)
            frontier_x, frontier_y = self.find_closest_frontier(center=door_center, radius=2.0)
            if frontier_x is not None:
                angle = math.atan2(frontier_y - ry, frontier_x - rx)
                self.commit_goal(frontier_x, frontier_y, angle)
                self.get_logger().info(f"State 2: Mapping door {self.active_target_door}...", throttle_duration_sec=5.0)
            else:
                self.get_logger().info(f"Mapping behind door {self.active_target_door} is complete!")
                self.explored_doors.add(self.active_target_door)
                self.active_target_door = None
                self.clear_goal()
                self.current_state = RobotState.NAVIGATE_TO_DOOR
        # --- STATE 3: ΕΠΙΣΤΡΟΦΗ ΣΤΟ ΣΠΙΤΙ ---
        elif self.current_state == RobotState.RETURN_HOME:
            hx, hy, htheta = self.home_pose
            dist_to_home = math.hypot(hx - rx, hy - ry)
            
            self.commit_goal(hx, hy, htheta)
            if dist_to_home < 0.15:
                self.get_logger().info("Our robot's back in base! Mission Complete.")
                self.destroy_timer(self.decision_timer)
    def reached(self, gx, gy):
        rx, ry, _ = self.get_robot_pose()
        if rx is None:
            return False
        return math.hypot(gx - rx, gy - ry) < self.arrival_tol

    def commit_goal(self, x, y, theta):
        """Στέλνει στόχο ΜΟΝΟ αν είναι νέος -> δεν ξαναπλανάρει κάθε δευτερόλεπτο."""
        new_goal = (round(x, 2), round(y, 2))
        if self.active_goal != new_goal:
            self.active_goal = new_goal
            self.goal_start_time = self.get_clock().now().nanoseconds / 1e9
            self.send_navigation_goal(x, y, theta)

    def goal_timed_out(self):
        if self.goal_start_time is None:
            return False
        now = self.get_clock().now().nanoseconds / 1e9
        return (now - self.goal_start_time) > self.goal_timeout

    def clear_goal(self):
        self.active_goal = None
        self.goal_start_time = None
    def send_navigation_goal(self, x, y, theta):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.orientation.z = math.sin(theta / 2.0)
        msg.pose.orientation.w = math.cos(theta / 2.0)
        self.goal_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = TaskManagerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
