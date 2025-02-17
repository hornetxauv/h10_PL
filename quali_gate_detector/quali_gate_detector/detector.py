from __future__ import print_function
import math
import cv2
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from custom_msgs.msg import GateDetection

# for foxglove params
from rclpy.executors import MultiThreadedExecutor
from ament_index_python.packages import get_package_share_directory
from controls_movement.param_helper import read_pid_yaml_and_generate_parameters

max_cnt_width = 200
min_cnt_height = 40
min_cnt_area = 500

class QualiGateDetector(Node):
    is_playing = False
    original_img = None
    prev_msg = None

    def __init__(self):
        super().__init__("quali_gate_detector_node")
        package_directory = get_package_share_directory('quali_gate_detector')
        self.declare_parameter('config_location', rclpy.Parameter.Type.STRING)
        config_location = package_directory + self.get_parameter('config_location').get_parameter_value().string_value
        self.declare_parameters(namespace='', parameters=read_pid_yaml_and_generate_parameters('quali_gate_detector_node', config_location))

        self.pub_debug_img = self.create_publisher(Image, "/perc/debug_img", 10)
        self.pub_debug_img_2 = self.create_publisher(Image, "/perc/debug_img_2", 10)
        self.pub_debug_img_3 = self.create_publisher(Image, "/perc/debug_img_3", 10)
        self.pub_debug_img_4 = self.create_publisher(Image, "/perc/debug_img_4", 10)
        self.pub_gate_detection = self.create_publisher(
            GateDetection,
            "/perc/quali_gate", 10)
        self.sub_image_feed = self.create_subscription(
            CompressedImage,
            #"/left/compressed", #for feed from session3 rosbag
            "/left/image_raw/compressed", #for live feed from v4l2
            self.image_feed_callback,
            10)
        self.bridge = CvBridge()

        timer_period = 0.5
        self.timer = self.create_timer(timer_period, self.when_not_playing)
    
    def get_value(self, param_name: str):
        return int(self.get_parameter(param_name).get_parameter_value().double_value)

    def image_feed_callback(self, msg):
        self.process_image(msg)
        self.is_playing = True
        self.prev_msg = msg

    def when_not_playing(self):
        if self.is_playing == True:
            self.is_playing = False
            return
        self.process_image(self.prev_msg)

    def process_image(self, msg):
        if (msg == None):
            return

        # Feel free to modify this callback function, or add other functions in any way you deem fit
        # Here is sample code for converting a coloured image to gray scale using opencv
        cv_img = self.bridge.compressed_imgmsg_to_cv2(msg)
        img_height, img_width, channels = cv_img.shape
        self.original_img = cv_img
        
        # https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html
        hsv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
        
        # https://stackoverflow.com/questions/39308030/how-do-i-increase-the-contrast-of-an-image-in-python-opencv
        h, s, v = cv2.split(hsv_img)

        # Applying CLAHE to L-channel
        # feel free to try different values for the limit and grid size:
        clahe = cv2.createCLAHE(clipLimit=self.get_value("clahe_limit"), tileGridSize=(8,8))
        cl = clahe.apply(v)

        # merge the CLAHE enhanced L-channel with the a and b channel
        hsv_clahe = cv2.merge((h,s,cl))
        bgr_clahe = cv2.cvtColor(hsv_clahe, cv2.COLOR_HSV2BGR)
        # self.pub_img(bgr_clahe, encoding="bgr8")
        
        poles_mask = self.find_poles(hsv_clahe)
        self.pub_img(poles_mask)#, encoding="bgr8")

        erode_first = self.get_value("erosion_first")
        morph_iterations = self.get_value("morph_iterations")
        second_morph = poles_mask

        for i in range(morph_iterations):
            first_morph = self.erode(second_morph) if erode_first else self.dilate(poles_mask)
            second_morph = self.dilate(first_morph) if erode_first else self.erode(poles_mask)

        self.pub_img_2(first_morph)#, encoding="bgr8")
        self.pub_img_3(second_morph)#, encoding="bgr8")
        
        cnts = self.filter_contours(second_morph, False)
        (drawn_img, bbox_heights) = self.draw_contours_and_bbox(bgr_clahe, cnts)
        centres = self.find_and_draw_centres(drawn_img, cnts)
        drawn_img, gate_centre = self.find_and_draw_midpoint(drawn_img, centres)
        self.pub_img_4(drawn_img, encoding="bgr8")

        msg = GateDetection()
        msg.width = 0.0
        msg.dx = 0.0
        msg.dy = 0.0
        msg.sides_ratio = 0.0

        if gate_centre:
            msg.width = float(self.find_distance(centres))
            msg.dx = float(img_width/2 - gate_centre[0])
            msg.dy = float(img_height/2 - gate_centre[1])
            # array_msg = Float32MultiArray()
            # array_msg.data=[msg.dx, msg.dy, msg.distance]
            # self.pub_gate_detection.publish(array_msg)
        # else:
        #     array_msg = Float32MultiArray()
        #     array_msg.data=[0.0,0.0,0.0]
        #     self.pub_gate_detection.publish(array_msg)

        if len(bbox_heights) == 2:
            if bbox_heights[0] != 0 and bbox_heights[1] != 0:
                msg.sides_ratio = float(bbox_heights[0]/bbox_heights[1])

        self.pub_gate_detection.publish(msg)

    def find_poles(self, frame):
        if self.get_value("min_H") < self.get_value("max_H"):
            return cv2.inRange(frame, (self.get_value("min_H"), self.get_value("min_S"), self.get_value("min_V")), (self.get_value("max_H"), self.get_value("max_S"), self.get_value("max_V")))
        # divide the H values by 2 because our slider is 0-360 but cv2 takes 0-180
        else:
            first = cv2.inRange(frame, (0, self.get_value("min_S"), self.get_value("min_V")), (self.get_value("max_H"), self.get_value("max_S"), self.get_value("max_V")))
            # 180 used to be values["min H"].maximum/2
            second = cv2.inRange(frame, (self.get_value("min_H"), self.get_value("min_S"), self.get_value("min_V")), (180, self.get_value("max_S"), self.get_value("max_V")))
            return cv2.bitwise_or(first, second)

    def erode(self, mask):
        erosion_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.get_value("e_kernel_w"), self.get_value("e_kernel_h")))
        eroded = cv2.erode(mask, erosion_kernel, iterations=self.get_value("erosion_iterations"))
        return eroded
    
    def dilate(self, mask):
        dilation_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.get_value("d_kernel_w"), self.get_value("d_kernel_h")))
        dilated = cv2.dilate(mask, dilation_kernel, iterations=self.get_value("dilation_iterations"))
        return dilated

    def filter_contours(self, mask, remove_top=True):
        image_height = mask.shape[0]
        #find contours
        cnts, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if cnts == None:
            return mask

        # print("1",[cv2.contourArea(i) for i in cnts])
        
        # remove contours that are too small
        cnts = list(filter(lambda c: cv2.contourArea(c) > min_cnt_area, cnts))
        # print("2",[cv2.contourArea(i) for i in cnts])
        # remove contours that are too wide or short
        def filter_by_dimensions(cnt):
            _, _, w, h = cv2.boundingRect(cnt)
            # print(w,h)
            return (w < self.get_value("max_cnt_w") and h > self.get_value("min_cnt_h"))
        cnts = list(filter(lambda c: filter_by_dimensions(c), cnts))
        # print("3",[cv2.contourArea(i) for i in cnts])
        # remove contours in top 30% of image
        def get_cnt_centre(cnt):
            M = cv2.moments(cnt)
            if M['m00'] != 0:
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                return (cx, cy)
        if remove_top:
            cnts = list(filter(lambda c: get_cnt_centre(c)[1] > 0.3 * image_height, cnts))
        # print("4",[cv2.contourArea(i) for i in cnts])
        # #remove contours in bottom 10% of image
        # cnts = list(filter(lambda c: get_cnt_centre(c)[1] < 0.9 * image_height, cnts))

        if len(cnts):
            def find_top_2(arr):
                first = None
                second = None
                for i in arr:
                    area_i = cv2.contourArea(i)
                    # print(area_i)
                    area_first = 0 if first is None else cv2.contourArea(first)
                    area_second = 0 if second is None else cv2.contourArea(second)
                    if area_i > area_first:
                        second = first
                        first = i
                    elif area_i == area_first:
                        second = i
                    elif area_i > area_second:
                        second = i
                return list(filter(lambda x:x is not None, [first, second]))
            
            # cnts = [max(cnts, key=cv2.contourArea)]
            cnts = find_top_2(cnts)

        return cnts
    
    def draw_contours_and_bbox(self, img, cnts, label=None, colour=(0, 255, 0)):
        bbox_heights = []
        for cnt in cnts:
            img = cv2.drawContours(img, cnt, -1, (0,255,0), 1)
            x,y,w,h = cv2.boundingRect(cnt)
            bbox_heights.append(h)
            img = cv2.rectangle(img, (x, y), (x + w, y + h), colour, 3)

            dimensionsStr = f'({w}, {h}); {str(cv2.contourArea(cnt))}'
            if label:
                label += f" ({dimensionsStr})"
            else:
                label = dimensionsStr
            img = cv2.putText(img, label, (x+10, y-10), 0, 1, colour, 2)
        
        return (img, bbox_heights)
    
    def find_and_draw_centres(self, img, cnts, colour=(0, 0, 255)):
        centres=[]
        for cnt in cnts:
            M = cv2.moments(cnt)
            if M['m00'] != 0:
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                centre = (cx, cy)
                centres.append(centre)
                cv2.drawContours(img, [cnt], -1, colour, 2)
                cv2.circle(img, centre, 7, colour, -1)
                # cv2.putText(img, "center", (cx - 20, cy - 20),
                #         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return centres
    
    def find_distance(self, centres):
        if len(centres) < 2:
            return 0
        dx=0
        dy=0
        x=centres[-1][0]
        y=centres[-1][1]
        for c in centres:
            dx += abs(c[0]-x)
            dy += abs(c[1]-y)
        return math.sqrt(math.pow(dx, 2) + math.pow(dy, 2))

    def find_and_draw_midpoint(self, img, centres, colour=(0, 0, 255)):
        l=len(centres)
        if l < 2:
            return img, None
        x=0
        y=0
        for c in centres:
            x += c[0]
            y += c[1]
        gate_centre = (round(x/l), round(y/l))
        cv2.circle(img, gate_centre, 7, colour, -1)
        return img, gate_centre
    
    def _pub_img(self, pubber, frame, encoding="mono8"):
        img_msg = self.bridge.cv2_to_imgmsg(frame, encoding=encoding)
        pubber.publish(img_msg)

    def pub_img(self, frame, **kwargs):
        self._pub_img(self.pub_debug_img, frame, **kwargs)
    
    def pub_img_2(self, frame, **kwargs):
        self._pub_img(self.pub_debug_img_2, frame, **kwargs)

    def pub_img_3(self, frame, **kwargs):
        self._pub_img(self.pub_debug_img_3, frame, **kwargs)

    def pub_img_4(self, frame, **kwargs):
        self._pub_img(self.pub_debug_img_4, frame, **kwargs)
    
    # def pub_img_2(self, frame, encoding="mono8"):
    #     img_msg = self.bridge.cv2_to_imgmsg(frame, encoding=encoding)
    #     self.pub_debug_img_2.publish(img_msg)
    
def main(args=None):
    rclpy.init(args=args)
    detector = QualiGateDetector()

    executor = MultiThreadedExecutor()
    executor.add_node(detector)
    executor.spin()
    
    rclpy.shutdown()
        
if __name__=='__main__':
    main()
