import rclpy
from rclpy.node import Node
import math
import numpy as np
from geometry_msgs.msg import Twist
from nav_msgs.msg import Path
from std_msgs.msg import Bool
from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

class HolonomicPathFollowerNode(Node):
    def __init__(self):
        super().__init__('holonomic_path_follower_node')

        # --- PI-Controller Gains ---
        self.kp_linear = 0.6
        self.ki_linear = 0.02   
        
        self.kp_angular = 1.2
        self.ki_angular = 0.05   

        # --- Limits & Tolerances ---
        self.max_linear_vel = 0.09
        self.max_angular_vel = 0.12
        self.xy_tolerance = 0.2    # Tolerance to consider the final destination reached
        self.yaw_tolerance = 0.2
        
        # --- Pure Pursuit / Lookahead Parameters ---
        self.lookahead_distance = 0.15 # How far ahead on the path the robot looks (meters)
        
        # --- Anti-Windup Limits ---
        self.max_integral_linear = 0.3 
        self.max_integral_angular = 0.3

        # --- Path Tracking State Variables ---
        self.current_path = []
        self.has_path = False

        # --- Recovery State Machine Variables ---
        self.state = "NORMAL" # Can be: "NORMAL", "BACKING_UP", "SPINNING"
        self.recovery_start_time = None
        self.last_time = None

        # --- Integral Accumulators ---
        self.reset_integrals()

        # --- TF Setup ---
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # --- Publishers & Subscribers ---
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # subscribing to the A* Path topic instead of a single goal pose
        self.path_sub = self.create_subscription(Path, '/planned_path', self.path_callback, 10)
        self.stuck_sub = self.create_subscription(Bool, '/stuck_alert', self.stuck_callback, 10)

        # Control Loop at 20Hz
        self.control_timer = self.create_timer(0.05, self.control_loop)
        
        self.get_logger().info("🚀 Holonomic A* Path Follower Node Ready!")

    def reset_integrals(self):
        """Resets PI integrals to prevent windup spikes."""
        self.integral_x = 0.0
        self.integral_y = 0.0
        self.integral_theta = 0.0

    def path_callback(self, msg):
        """Triggered when A* node publishes a new path."""
        if len(msg.poses) == 0:
            self.get_logger().warn("Received an empty path!")
            return

        self.current_path = msg.poses
        self.has_path = True
        self.reset_integrals()
        self.last_time = self.get_clock().now()
        self.get_logger().info(f"🛣️ Received new path containing {len(self.current_path)} waypoints.")

    def stuck_callback(self, msg):
        self.get_logger().info(f"stuck_alert received: {msg.data}")
        if msg.data and self.has_path and self.state == "NORMAL":
            self.get_logger().warn("⚠️ Stuck Alert received! Starting recovery...")
            self.state = "BACKING_UP"
            self.recovery_start_time = self.get_clock().now()

    def get_yaw_from_quaternion(self, q):
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def control_loop(self):
        if not self.has_path or not self.current_path:
            return

        current_time = self.get_clock().now()
        
        # Calculate dt for the Integral term
        if self.last_time is None:
            self.last_time = current_time
            return
        dt = (current_time - self.last_time).nanoseconds / 1e9
        self.last_time = current_time
        if dt > 0.5: dt = 0.05 # Cap dt spikes if simulation lags

        # =========================================================================
        # RECOVERY STATES
        # =========================================================================
        if self.state == "BACKING_UP":
            self.get_logger().info("STATE: BACKING_UP", throttle_duration_sec=0.3)
            duration = (current_time - self.recovery_start_time).nanoseconds / 1e9
            if duration < 1.5:
                cmd_msg = Twist()
                cmd_msg.linear.x = -0.07 
                self.cmd_vel_pub.publish(cmd_msg)
                return
            else:
                self.state = "SPINNING"
                self.recovery_start_time = current_time

        elif self.state == "SPINNING":
            self.get_logger().info("STATE: SPINNING", throttle_duration_sec=0.3)
            duration = (current_time - self.recovery_start_time).nanoseconds / 1e9
            if duration < 1.5:
                cmd_msg = Twist()
                cmd_msg.angular.z = 0.15 
                self.cmd_vel_pub.publish(cmd_msg)
                return
            else:
                self.get_logger().info("✅ Recovery complete. Returning to path tracking.")
                self.state = "NORMAL"
                self.reset_integrals()
                return

        # =========================================================================
        # NORMAL PATH TRACKING STATE (Lookahead + PI)
        # =========================================================================
        elif self.state == "NORMAL":
            self.get_logger().info("entering NORMAL", throttle_duration_sec=0.3)
            try:
                trans = self.tf_buffer.lookup_transform('map', 'base_footprint', rclpy.time.Time())
                current_x = trans.transform.translation.x
                current_y = trans.transform.translation.y
                current_theta = self.get_yaw_from_quaternion(trans.transform.rotation)
            except (LookupException, ConnectivityException, ExtrapolationException) as e:
                self.get_logger().warn(f"TF Error: {e}")
                return

            # Check if we reached the absolute final destination of the path
            final_pose = self.current_path[-1].pose.position
            dist_to_destination = math.hypot(final_pose.x - current_x, final_pose.y - current_y)
            self.get_logger().info(f"dist to goal: {dist_to_destination:.3f}", throttle_duration_sec=0.3)
            
            if dist_to_destination < self.xy_tolerance:
                self.stop_robot()
                self.has_path = False
                self.current_path = []
                self.get_logger().info("🏁 Destination reached successfully!")
                return

            # Find the closest waypoint on the path to avoid tracking old data
            closest_idx = 0
            min_dist = float('inf')
            for i in range(len(self.current_path)):
                wp = self.current_path[i]
                dist = math.hypot(wp.pose.position.x - current_x, wp.pose.position.y - current_y)
                if dist < min_dist:
                    min_dist = dist
                    closest_idx = i

            # Remove old points
            if closest_idx > 0:
                self.current_path = self.current_path[closest_idx:]
                closest_idx = 0

            # Look ahead along the path to find our "carrot" target point
            target_x = final_pose.x
            target_y = final_pose.y
            

            if dist_to_destination < self.lookahead_distance:
                target_x = final_pose.x
                target_y = final_pose.y
                self.reset_integrals()
            else:
                for i in range(closest_idx, len(self.current_path)):
                    wp_x = self.current_path[i].pose.position.x
                    wp_y = self.current_path[i].pose.position.y
                    dist_to_wp = math.hypot(wp_x - current_x, wp_y - current_y)
                    
                    if dist_to_wp >= self.lookahead_distance:
                        target_x = wp_x
                        target_y = wp_y
                        break

            # Compute Dynamic Orientation (Face toward the lookahead point)
            dist_to_target = math.hypot(target_x - current_x, target_y - current_y)
            if dist_to_target > 0.02:
                target_theta = math.atan2(target_y - current_y, target_x - current_x)
            else:
                target_theta = current_theta # Keep heading if practically on top of target

            # Global Error Calculations
            error_x_global = target_x - current_x
            error_y_global = target_y - current_y
            
            error_theta = target_theta - current_theta
            error_theta = math.atan2(math.sin(error_theta), math.cos(error_theta)) # Normalize angle [-pi, pi]

            # Transform errors to local frame (Robot reference frame)
            cos_t = math.cos(current_theta)
            sin_t = math.sin(current_theta)
            error_x_local = error_x_global * cos_t + error_y_global * sin_t
            error_y_local = -error_x_global * sin_t + error_y_global * cos_t

            # Integral Terms + Anti-Windup Clamping
            self.integral_x = np.clip(self.integral_x + (error_x_local * dt), -self.max_integral_linear, self.max_integral_linear)
            self.integral_y = np.clip(self.integral_y + (error_y_local * dt), -self.max_integral_linear, self.max_integral_linear)
            self.integral_theta = np.clip(self.integral_theta + (error_theta * dt), -self.max_integral_angular, self.max_integral_angular)

            # PI Control Equations
            vx = (self.kp_linear * error_x_local) + (self.ki_linear * self.integral_x)
            # vy = (self.kp_linear * error_y_local) + (self.ki_linear * self.integral_y)
            wz = (self.kp_angular * error_theta) + (self.ki_angular * self.integral_theta)

            # --- Differential Drive Logic ---
            # Αν το ρομπότ πρέπει να κάνει μεγάλη στροφή,
            # μειώνουμε τη γραμμική ταχύτητα για να προλάβει να στρίψει πριν φύγει από το path!
            directional_multiplier = max(0.1, math.cos(error_theta))
            vx = vx * directional_multiplier

            # --- Smart Differential Drive Logic ---
            # Αν το σφάλμα γωνίας είναι πάνω από 0.2 rad, 
            # μηδενίζουμε εντελώς την ταχύτητα X. Το ρομπότ στρίβει επιτόπου.
            if abs(error_theta) > 0.2:
                vx = 0.0
            else:
                # Αν έχει ευθυγραμμιστεί αρκετά, προχωράει ομαλά
                directional_multiplier = max(0.2, math.cos(error_theta))
                vx = vx * directional_multiplier

            # Velocity Limits Constraints
            #linear_speed = math.hypot(vx, vy)
            #if linear_speed > self.max_linear_vel:
            #    scale = self.max_linear_vel / linear_speed
            #    vx *= scale
            #    vy *= scale
            #wz = np.clip(wz, -self.max_angular_vel, self.max_angular_vel)

            if vx > self.max_linear_vel:
                vx = self.max_linear_vel
            elif vx < -self.max_linear_vel:
                vx = -self.max_linear_vel
                
            wz = np.clip(wz, -self.max_angular_vel, self.max_angular_vel)

            # Constraint to avoid jitter behavior
            if abs(wz) < 0.01:
                wz = 0.0

            # Publish Control Commands
            cmd_msg = Twist()
            cmd_msg.linear.x = float(vx)
            cmd_msg.linear.y = 0.0
            cmd_msg.angular.z = float(wz)
            self.cmd_vel_pub.publish(cmd_msg)

    def stop_robot(self):
        self.cmd_vel_pub.publish(Twist())

def main(args=None):
    rclpy.init(args=args)
    node = HolonomicPathFollowerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
