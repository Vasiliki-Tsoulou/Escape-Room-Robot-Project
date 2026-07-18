import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import math
import numpy as np
from std_msgs.msg import Float32MultiArray

class VisionNode(Node):
    def __init__(self):
        super().__init__('vision_node')
        
        self.subscription = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.publisher_ = self.create_publisher(Image, '/camera/image_annotated', 10)

        # Publisher που στέλνει Array: [ID, Απόσταση, Γωνία]
        self.vision_pub = self.create_publisher(Float32MultiArray, '/vision/marker_info', 10)
        self.found_keys = set()
        
        self.bridge = CvBridge()
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        
        # --- Παράμετροι Κάμερας ---
        self.marker_real_size = 0.10  # Μέγεθος φυσικού ArUco σε μέτρα
        self.focal_length = 500.0     # Προσεγγιστικό Focal Length σε pixels για κάμερες 640x480
        self.image_center_x = 320.0   # Το κέντρο της οθόνης
        
        self.get_logger().info("Το Vision Node ξεκίνησε! Περιμένω εικόνα...")

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            return

        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
        
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(cv_image, corners, ids)
            
            for i in range(len(ids)):
                target_id = int(ids[i][0])
                marker_corners = corners[i][0]
                
                center_x = int(np.mean(marker_corners[:, 0]))
                center_y = int(np.mean(marker_corners[:, 1]))
                
                # Απόσταση Βρίσκουμε το πλάτος του marker σε pixels
                pixel_width = math.sqrt((marker_corners[0][0] - marker_corners[1][0])**2 + 
                                        (marker_corners[0][1] - marker_corners[1][1])**2)
                
                # Τύπος Pinhole: Distance = (Real_Size * Focal_Length) / Pixel_Width
                distance = (self.marker_real_size * self.focal_length) / pixel_width
                 
                # (Αρνητική γωνία = δεξιά, Θετική = αριστερά, σε ακτίνια)
                angle = (self.image_center_x - center_x) * 0.0015 
                
                # Οπτικοποίηση
                cv2.circle(cv_image, (center_x, center_y), 6, (0, 0, 255), -1)
                text = f"ID: {target_id} | Dist: {distance:.2f}m"
                cv2.putText(cv_image, text, (center_x - 30, center_y - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Στέλνουμε το μήνυμα στον Mapper
                # Ακόμα κι αν είναι κλειδί που έχουμε ξαναδεί, ο Mapper κάνει μόνος του τον έλεγχο.
                msg_out = Float32MultiArray()
                msg_out.data = [float(target_id), float(distance), float(angle)]
                self.vision_pub.publish(msg_out)
                
        try:
            annotated_msg = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
            self.publisher_.publish(annotated_msg)
        except Exception as e:
            pass

def main(args=None):
    rclpy.init(args=args)
    node = VisionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()