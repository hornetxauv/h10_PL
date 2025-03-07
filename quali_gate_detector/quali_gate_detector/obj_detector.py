# from __future__ import print_function
# import math
# https://github.com/computervisioneng/train-yolov8-custom-dataset-step-by-step-guide/blob/master/local_env/predict_video.py
import ultralytics
import numpy as np # need numpy < 2 https://stackoverflow.com/questions/71689095/how-to-solve-the-pytorch-runtimeerror-numpy-is-not-available-without-upgrading

import cv2
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from custom_msgs.msg import GateDetection
# from control_panel.control_panel import create_control_panel, ControlPanelItem as CPI

detection_interval = 4
#model_path = '/home/bb/ros_workspaces/h10_workspace/h9.onnx'
model_path = '/home/aa/ML/front_yolov8n_070424_1.engine'
threshold = 0.5

class ObjDetector(Node):
    is_playing = False
    original_img = None
    prev_msg = None
    counter = 0

    def __init__(self):
        super().__init__("detector")
        self.get_logger().info("init")
        self.get_logger().info(ultralytics.__version__)
        #self.model = ultralytics.YOLO(model_path, task="detect")
        # Load the exported TensorRT model
        self.model = ultralytics.YOLO(model_path, task="detect")
        self.pub_debug_img = self.create_publisher(CompressedImage, "/perc/debug_img", 10)
        self.pub_detection = self.create_publisher(
            GateDetection,
            "/perc/quali_gate", 10)
        self.sub_image_feed = self.create_subscription(
            CompressedImage,
            # "/left/image_raw", #for feed from session3 rosbag
            # "/left/compressed", #for feed from session3 rosbag
            "/left/image_raw/compressed", #for live feed from v4l2
            self.image_feed_callback,
            10)
        self.bridge = CvBridge()

        # timer_period = 0.5
        # self.timer = self.create_timer(timer_period, self.when_not_playing)

    def image_feed_callback(self, msg):
        if self.counter == detection_interval:
            self.process_image(msg)
        else:
            self.counter += 1
        self.is_playing = True
        self.prev_msg = msg

    # def when_not_playing(self):
    #     if self.is_playing == True:
    #         self.is_playing = False
    #         return
    #     self.process_image(self.prev_msg)

    def process_image(self, msg):
        if (msg == None):
            return

        # Feel free to modify this callback function, or add other functions in any way you deem fit
        # Here is sample code for converting a coloured image to gray scale using opencv
        cv_img = self.bridge.compressed_imgmsg_to_cv2(msg)
        img_height, img_width, channels = cv_img.shape
        # height and width is 480, 640
        # self.get_logger().info(f"img_height: {img_height}, img_width: {img_width}")
        self.original_img = cv_img
        
        # # https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html
        # hsv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
        
        # # https://stackoverflow.com/questions/39308030/how-do-i-increase-the-contrast-of-an-image-in-python-opencv
        # h, s, v = cv2.split(hsv_img)

        # # Applying CLAHE to L-channel
        # # feel free to try different values for the limit and grid size:
        # clahe = cv2.createCLAHE(clipLimit=values['clahe limit'].value/10, tileGridSize=(8,8))
        # cl = clahe.apply(v)

        # # merge the CLAHE enhanced L-channel with the a and b channel
        # hsv_clahe = cv2.merge((h,s,cl))
        # bgr_clahe = cv2.cvtColor(hsv_clahe, cv2.COLOR_HSV2BGR)
        # # self.pub_img(bgr_clahe, encoding="bgr8")
        
        results = self.model.predict(self.original_img, imgsz=[640, 640])[0]
        
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result

        #if (len(results) > 0):
            if score > threshold:
                name = results.names[int(class_id)].upper()
                colour = (0, 255, 0)
                # cv2.rectangle(cv_img, (int(x1), int(y1)), (int(x2), int(y2)), colour, 4)
                cv2.putText(cv_img, name, (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, colour, 3, cv2.LINE_AA)

                self.get_logger().info(f"detected {name}")
                if name == "GATE":
                    gate_centre = [(x1+x2)/2, (y1+y2)/2]
                    cv2.circle(cv_img, [int(i) for i in gate_centre], 7, colour, -1)

                    msg = GateDetection()
                    msg.width = 0.0
                    msg.dx = 0.0
                    msg.dy = 0.0

                    #if gate_centre:
                    msg.width = float(x2-x1)
                    msg.dx = float(img_width/2 - gate_centre[0])
                    msg.dy = float(img_height/2 - gate_centre[1])
                    msg.sides_ratio = 1.0 #ML won't be able to give us sides ratio for now
                    self.get_logger().info("pub detection")
                    self.pub_detection.publish(msg)
        
        self.pub_img(cv_img, encoding="bgr8")
    
    def _pub_img(self, pubber, frame, encoding="mono8"):
        img_msg = self.bridge.cv2_to_imgmsg(frame, encoding=encoding)
        pubber.publish(img_msg)

    def pub_img(self, frame, **kwargs):
        img_msg = self.bridge.cv2_to_compressed_imgmsg(frame)
        self.pub_debug_img.publish(img_msg)
        # self._pub_img(self.pub_debug_img, frame, **kwargs)
    
def main(args=None):

    rclpy.init(args=args)
    detector = ObjDetector()
    rclpy.spin(detector)

    # Below lines are not strictly necessary
    detector.destroy_node()
    rclpy.shutdown()
        
if __name__=='__main__':
    main()
