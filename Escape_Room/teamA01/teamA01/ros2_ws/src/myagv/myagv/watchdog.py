import rclpy
from rclpy.node import Node
import math
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException


class StuckRecoveryNode(Node):

    def __init__(self):
        super().__init__('stuck_recovery_node')

        self.cmd_sub = self.create_subscription(
            Twist, '/cmd_vel', self.cmd_callback, 10
        )

        self.alert_pub = self.create_publisher(Bool, '/stuck_alert', 10)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # state
        self.position_history = []  # (time, x, y)

        self.cmd_linear = 0.0
        self.cmd_angular = 0.0
        self.last_alert_time = 0.0
        self.alert_cooldown = 5.0   # after firing, wait before checking again

        # timer
        self.create_timer(1.0, self.check_stuck)

        self.get_logger().info("StuckRecoveryNode started")

    # cmd velocity input
    def cmd_callback(self, msg: Twist):
        self.cmd_linear = math.hypot(msg.linear.x, msg.linear.y)
        self.cmd_angular = abs(msg.angular.z)

    # get robot position
    def get_position(self):
        try:
            trans = self.tf_buffer.lookup_transform(
                'map',
                'base_footprint',
                rclpy.time.Time()
            )
            return (
                trans.transform.translation.x,
                trans.transform.translation.y
            )
        except (LookupException, ConnectivityException, ExtrapolationException):
            return None

    # stuck detection
    def check_stuck(self):
        alert = Bool()
        alert.data = False

        now = self.get_clock().now().nanoseconds / 1e9

        # after firing, give the follower time to recover
        if now - self.last_alert_time < self.alert_cooldown:
            self.alert_pub.publish(alert)
            return

        # Only judge stuck if actually told to drive
        if self.cmd_linear < 0.05:
            self.alert_pub.publish(alert)
            return

        # and not simply rotating in place (#2)
        if self.cmd_angular > 0.1:
            self.alert_pub.publish(alert)
            return

        pos = self.get_position()
        if pos is None:
            self.alert_pub.publish(alert)
            return

        x, y = pos
        self.position_history.append((now, x, y))

        # keep last 4s
        self.position_history = [p for p in self.position_history if now - p[0] <= 4.0]

        # require a real 3s window before judging
        if now - self.position_history[0][0] < 3.0:
            self.alert_pub.publish(alert)
            return

        old_x, old_y = self.position_history[0][1], self.position_history[0][2]
        distance = math.hypot(x - old_x, y - old_y)

        if distance < 0.05:
            self.get_logger().error(f"🚨 STUCK: only {distance:.3f} m moved in 3s")
            alert.data = True
            self.alert_pub.publish(alert)
            self.position_history.clear()
            self.last_alert_time = now
            return

        self.alert_pub.publish(alert)

def main():
    rclpy.init()
    node = StuckRecoveryNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()