from __future__ import print_function
import threading
import cv2
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage

# for foxglove params
from rclpy.executors import MultiThreadedExecutor
from ament_index_python.packages import get_package_share_directory
from controls_movement.param_helper import read_pid_yaml_and_generate_parameters

class QualiGateDetector(Node):

    def __init__(self):
        super().__init__("quali_gate_detector_node")

        self.sub_image_feed = self.create_subscription(
            CompressedImage,
            #"/left/compressed", #for feed from session3 rosbag
            "/left/image_raw/compressed", #for live feed from v4l2
            self.process_image,
            10)
        self.bridge = CvBridge()

        self.get_logger().info("init")

        input_thread = threading.Thread(target=self.init_foxglove_params)
        input_thread.daemon = True
        input_thread.start()

    def init_foxglove_params(self):
        package_directory = get_package_share_directory('quali_gate_detector')
        self.declare_parameter('config_location', rclpy.Parameter.Type.STRING)
        config_location = package_directory + self.get_parameter('config_location').get_parameter_value().string_value
        self.declare_parameters(namespace='', parameters=read_pid_yaml_and_generate_parameters('quali_gate_detector_node', config_location))
    
    def get_value(self, param_name: str):
        return int(self.get_parameter(param_name).get_parameter_value().double_value)

    def process_image(self, msg):
        self.get_logger().info("callbackk")

        cv_img = self.bridge.compressed_imgmsg_to_cv2(msg) 
        
        # no error if both lines below are commented out
        hsv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_img) # with only this commented out: Exception when getting parameters: Timed out waiting for 70 parameter(s) from node '/quali_gate_detector_node'
        
        # ^ if i leave both lines in: Exception when getting parameters: Failed to retrieve parameter names for node '/quali_gate_detector_node'
        
        self.get_value("clahe_limit")
    
def main(args=None):
    rclpy.init(args=args)
    detector = QualiGateDetector()

    executor = MultiThreadedExecutor()
    executor.add_node(detector)
    executor.spin()
    
    rclpy.shutdown()
        
if __name__=='__main__':
    main()
