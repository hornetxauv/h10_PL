import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from cv_bridge import CvBridge
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import CompressedImage
from scipy.stats import mode

# Detect Floor tile lines and calculate angles
# Continuously output yaw angles as a graph and publish detected lines

class PoolLinesDetectorNode(Node):
    def __init__(self):
        super().__init__('pool_lines_detector_node')

        self.tile_angles = self.create_publisher(Float32MultiArray, '/perc/pool_lines', 10)
        self.lines_pub = self.create_publisher(CompressedImage, '/perc/debug_pool_lines_img', 10)
        self.bottom_image_feed = self.create_subscription(
            CompressedImage,
            #"/left/compressed", #for feed from session3 rosbag
            "/left/image_raw/compressed", #for live feed from v4l2
            self.image_feed_callback,
            10)

        # self.pub_gate_detection = self.create_publisher(
        #     GateDetection,
        #     "/perc/quali_gate", 10)
        self.bridge = CvBridge()
        self.angle_history = []

    def image_feed_callback(self, msg):
        img = self.bridge.compressed_imgmsg_to_cv2(msg)
        # img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        print("\033c")  # Clear terminal
        
        # Crop image to avoid detecting frame edges
        h, w = img.shape[:2]
        crop_margin = int(0.15 * w)  # Crop 15% from each side
        img_cropped = img[:, crop_margin:w - crop_margin]  # Crop left and right edges
        
        self.calculate_angle(img_cropped)
        self.plot_yaw_angles()

    def calculate_angle(self, img):
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(12, 12))
        cl1 = clahe.apply(imgGray)

        # Apply Otsu's thresholding
        _, thresholded_img = cv2.threshold(cl1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Perform Canny edge detection
        edges = cv2.Canny(thresholded_img, 50, 200, None, 3)
        cdstP = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # Perform Probabilistic Hough line transformation
        linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, None, 50, 10)

        if linesP is not None:
            all_thetas = []
            thetas = []
            h, w = img.shape[:2]
            edge_margin = int(0.10 * w)  # Ignore lines close to 10% of width from edges
            
            for line in linesP:
                line = line[0]
                x1, y1, x2, y2 = line
                
                # Ignore lines near the extreme left and right edges
                if x1 <= edge_margin or x2 >= w - edge_margin:
                    continue
                
                angle = np.arctan2(y2 - y1, x2 - x1)
                angle_deg = np.degrees(angle)
                angle_deg = angle_deg + (90 if angle_deg < 0 else -90)

                if -20 <= angle_deg <= 20:
                    cv2.line(cdstP, (x1, y1), (x2, y2), (0, 0, 255), 3, cv2.LINE_AA)
                    cv2.putText(cdstP, str(angle_deg.round(2)), (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3, cv2.LINE_AA)
                thetas.append(angle_deg)

            print(f"all thetas: {[i.round(2) for i in thetas]}")

            if not thetas:
                print("No valid lines detected")
                self.lines_pub.publish(self.bridge.cv2_to_compressed_imgmsg(img))
                return
            
            dominant_angle = np.median(thetas)
            print(f"Yaw Angle: {dominant_angle:.2f}")
            
            self.angle_history.append(dominant_angle)
            if len(self.angle_history) > 100:
                self.angle_history.pop(0)

            angles_msg = Float32MultiArray()
            angles_msg.data = [dominant_angle]
            self.tile_angles.publish(angles_msg)

            # Publish detected lines image
            #img_msg = self.bridge.cv2_to_compressed_imgmsg(cdstP)
            # img_msg = self.bridge.cv2_to_imgmsg(cdstP, encoding="bgr8")
            img_msg = self.bridge.cv2_to_compressed_imgmsg(cdstP)
            self.lines_pub.publish(img_msg)
        else:
            print("No lines detected")

    def plot_yaw_angles(self):
        plt.clf()
        plt.plot(self.angle_history, label='Yaw Angle')
        plt.xlabel('Frame')
        plt.ylabel('Angle (degrees)')
        plt.title('Yaw Angle Over Time')
        plt.legend()
        plt.pause(0.01)

def main(args=None):
    rclpy.init(args=args)
    floor_tiles_node = PoolLinesDetectorNode()
    plt.ion()
    rclpy.spin(floor_tiles_node)
    floor_tiles_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main() 
