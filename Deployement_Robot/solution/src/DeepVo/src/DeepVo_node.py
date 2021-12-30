#!/usr/bin/env python3
import numpy as np
import rospy
import rospkg

from duckietown.dtros import DTROS, NodeType, TopicType, DTParam, ParamType
from sensor_msgs.msg import CompressedImage, Image
from duckietown_msgs.msg import Twist2DStamped, EpisodeStart
import cv2
from object_detection.model import Wrapper
from cv_bridge import CvBridge

import onnxruntime as rt
import onnx

from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms

import CNN_run_updated



class DeepVoNode(DTROS):
    """
    Performs the DeepVo training and testing based o the images frames to calculate the estimated position
    
    """

    def __init__(self, node_name):

        # Initialize the DTROS parent class
        super(DeepVoNode, self).__init__(
            node_name=node_name,
            node_type=NodeType.LOCALIZATION
        )
        self.initialized = False
        self.log("Initializing!")
	    # get the name of the robot
        self.veh = rospy.get_namespace().strip("/")
	
        #Init the parameters
        self.resetParameters()
        self.theta_ref=np.deg2rad(0.0)
        self.omega=0.00
        
        self.log("Load kinematicks calibrations...")
        self.R=0.0318
        self.baseline=0.1
        self.read_params_from_calibration_file()
        
        self.CNN_ACTIVITY = False
	
        # Construct publishers
        car_cmd_topic = f"/{self.veh}/joy_mapper_node/car_cmd"
        self.pub_car_cmd = rospy.Publisher(
            car_cmd_topic,
            Twist2DStamped,
            queue_size=1,
            dt_topic_type=TopicType.CONTROL
        )

        episode_start_topic = f"/{self.veh}/episode_start"
        rospy.Subscriber(episode_start_topic,
                         EpisodeStart,
                         self.cb_episode_start,
                         queue_size=1)


        self. pub_detections_image = rospy.Publisher(
            "~object_detections_img", Image, queue_size=1, dt_topic_type=TopicType.DEBUG
        )	
        
        # Construct subscribers
        self.sub_image = rospy.Subscriber(
            f"/{self.veh}/camera_node/image/compressed",
            CompressedImage,
            self.image_cb,
            buff_size=10000000,
            queue_size=1
        )
        
        self.bridge= CvBridge()
        self.log("Finished model loading!")
        self.frame_id = 0
        self.first_image_received = False
        self.initialized = True
        self.log("Initialized!")

 ########################       
   def image_cb(self, image1, image2, image_msg):
        if not self.initialized:
            self.pub_car_commands(True, image_msg.header)
            return

        # Decode from compressed image with OpenCV
        try:
            image = self.bridge.compressed_imgmsg_to_cv2(image_msg) #TODO Modify cv2 to PIL
        except ValueError as e:
            self.logerr('Could not decode image: %s' % e)
            return

        old_image = image
        from dt_device_utils import DeviceHardwareBrand, get_device_hardware_brand
        if get_device_hardware_brand() != DeviceHardwareBrand.JETSON_NANO:  # if in sim
            if self._debug:
                print("Assumed an image was bgr and flipped it to rgb")
            old_img = image
            image = image[...,::-1].copy()  # image is bgr, flip it to rgb

        old_image = cv2.resize(old_img, (64,64))
        image = cv2.resize(image, (64,64))

        #Put image source to self memory image
        #image1 = self.bridge.cv2_to_imgmsg(old_image, encoding="bgr8")
        #image2 = self.bridge.cv2_to_imgmsg(image, encoding="bgr8")
        image1=Image.fromarray(cv2.cvtColot(old_image,cv2.COLOR_BGR2RGB))
        image2=Image.fromarray(cv2.cvtColot(image,cv2.COLOR_BGR2RGB))

    	CNN_results=CNN_run_updated.CNN_processing(image1,image2)
        
        #Here best way without bug is to put two images on the self system
        #image_msg = self.bridge.cv2_to_imgmsg(old_image, encoding="bgr8")
        #self.pub_detections_image.publish(image_msg)
##########################
   
        
#    def run_CNN(self,img1,img2):
#    	"""
#    	"""
#        #if img1 or img2 is empty
#        if not (img1 or img2)
#            return False
#
#        #if CNN_result is not defined or empty instead of (0,0,0)	
#    	if not(CNN_results):
#    		#TODO Definie CNN_results data type format output
#    		return False
#        
#    	CNN_results=CNN_run_updated.CNN_processing(img1,img2)
        

if __name__ == "__main__":
    # Initialize the node
    DeepVo_node = DeepVoNode(node_name="DeepVo_node")
    # Keep it spinning
    rospy.spin()
        
        
