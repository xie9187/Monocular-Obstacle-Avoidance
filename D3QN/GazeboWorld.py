import rospy
import math
import time
import numpy as np
import cv2
import copy
import tf
import random

from geometry_msgs.msg import Twist
from geometry_msgs.msg import Quaternion
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.msg import ModelState
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from kobuki_msgs.msg import BumperEvent
from sklearn.cluster import MiniBatchKMeans

class GazeboWorld():
	def __init__(self):
		 # initiliaze
		rospy.init_node('GazeboWorld', anonymous=False)

		#-----------Default Robot State-----------------------
		self.set_self_state = ModelState()
		self.set_self_state.model_name = 'mobile_base' 
		self.set_self_state.pose.position.x = 0.5
		self.set_self_state.pose.position.y = 0.
		self.set_self_state.pose.position.z = 0.
		self.set_self_state.pose.orientation.x = 0.0
		self.set_self_state.pose.orientation.y = 0.0
		self.set_self_state.pose.orientation.z = 0.0
		self.set_self_state.pose.orientation.w = 1.0
		self.set_self_state.twist.linear.x = 0.
		self.set_self_state.twist.linear.y = 0.
		self.set_self_state.twist.linear.z = 0.
		self.set_self_state.twist.angular.x = 0.
		self.set_self_state.twist.angular.y = 0.
		self.set_self_state.twist.angular.z = 0.
		self.set_self_state.reference_frame = 'world'

		#------------Params--------------------
		self.depth_image_size = [160, 128]
		self.rgb_image_size = [304, 228]
		self.bridge = CvBridge()

		self.object_state = [0, 0, 0, 0]
		self.object_name = []

		# 0. | left 90/s | left 45/s | right 45/s | right 90/s | acc 1/s | slow down -1/s
		self.action_table = [0.4, 0.2, np.pi/6, np.pi/12, 0., -np.pi/12, -np.pi/6]
		
		self.self_speed = [.4, 0.0]
		self.default_states = None
		
		self.start_time = time.time()
		self.max_steps = 10000

		self.depth_image = None
		self.bump = False

		#-----------Publisher and Subscriber-------------
		self.cmd_vel = rospy.Publisher('cmd_vel_mux/input/navi', Twist, queue_size = 10)
		self.set_state = rospy.Publisher('gazebo/set_model_state', ModelState, queue_size = 10)
		self.resized_depth_img = rospy.Publisher('camera/depth/image_resized',Image, queue_size = 10)
		self.resized_rgb_img = rospy.Publisher('camera/rgb/image_resized',Image, queue_size = 10)

		self.object_state_sub = rospy.Subscriber('gazebo/model_states', ModelStates, self.ModelStateCallBack)
		self.depth_image_sub = rospy.Subscriber('camera/rgb/image_raw', Image, self.RGBImageCallBack)
		self.rgb_image_sub = rospy.Subscriber('camera/depth/image_raw', Image, self.DepthImageCallBack)
		self.laser_sub = rospy.Subscriber('scan', LaserScan, self.LaserScanCallBack)
		self.odom_sub = rospy.Subscriber('odom', Odometry, self.OdometryCallBack)
		self.bumper_sub = rospy.Subscriber('mobile_base/events/bumper', BumperEvent, self.BumperCallBack)

		rospy.sleep(2.)

		# What function to call when you ctrl + c    
		rospy.on_shutdown(self.shutdown)


	def ModelStateCallBack(self, data):
		# self state
		idx = data.name.index("mobile_base")
		quaternion = (data.pose[idx].orientation.x,
					  data.pose[idx].orientation.y,
					  data.pose[idx].orientation.z,
					  data.pose[idx].orientation.w)
		euler = tf.transformations.euler_from_quaternion(quaternion)
		roll = euler[0]
		pitch = euler[1]
		yaw = euler[2]
		self.self_state = [data.pose[idx].position.x, 
					  	  data.pose[idx].position.y,
					  	  yaw,
					  	  data.twist[idx].linear.x,
						  data.twist[idx].linear.y,
						  data.twist[idx].angular.z]
		for lp in xrange(len(self.object_name)):
			idx = data.name.index(self.object_name[lp])
			quaternion = (data.pose[idx].orientation.x,
						  data.pose[idx].orientation.y,
						  data.pose[idx].orientation.z,
						  data.pose[idx].orientation.w)
			euler = tf.transformations.euler_from_quaternion(quaternion)
			roll = euler[0]
			pitch = euler[1]
			yaw = euler[2]

			self.object_state[lp] = [data.pose[idx].position.x, 
									 data.pose[idx].position.y,
									 yaw]

		if self.default_states is None:
			self.default_states = copy.deepcopy(data)


	def DepthImageCallBack(self, img):
		self.depth_image = img

	def RGBImageCallBack(self, img):
		self.rgb_image = img

	def LaserScanCallBack(self, scan):
		self.scan_param = [scan.angle_min, scan.angle_max, scan.angle_increment, scan.time_increment,
						   scan.scan_time, scan.range_min, scan. range_max]
		self.scan = np.array(scan.ranges)

	def OdometryCallBack(self, odometry):
		self.self_linear_x_speed = odometry.twist.twist.linear.x
		self.self_linear_y_speed = odometry.twist.twist.linear.y
		self.self_rotation_z_speed = odometry.twist.twist.angular.z

	def BumperCallBack(self, bumper_data):
		if bumper_data.state == BumperEvent.PRESSED:
			self.bump = True
		else:
			self.bump = False

	def GetDepthImageObservation(self):
		# ros image to cv2 image

		try:
			cv_img = self.bridge.imgmsg_to_cv2(self.depth_image, "32FC1")
		except Exception as e:
			raise e
		try:
			cv_rgb_img = self.bridge.imgmsg_to_cv2(self.rgb_image, "bgr8")
		except Exception as e:
			raise e
		cv_img = np.array(cv_img, dtype=np.float32)
		# resize
		dim = (self.depth_image_size[0], self.depth_image_size[1])
		cv_img = cv2.resize(cv_img, dim, interpolation = cv2.INTER_AREA)

		cv_img[np.isnan(cv_img)] = 0.
		cv_img[cv_img < 0.4] = 0.
		cv_img/=(10./255.)

		# cv_img/=(10000./255.)
		# print 'max:', np.amax(cv_img), 'min:', np.amin(cv_img)
		# cv_img[cv_img > 5.] = -1.

		# # inpainting
		# mask = copy.deepcopy(cv_img)
		# mask[mask == 0.] = 1.
		# mask[mask != 1.] = 0.
		# mask = np.uint8(mask)
		# cv_img = cv2.inpaint(np.uint8(cv_img), mask, 3, cv2.INPAINT_TELEA)

		# # guassian noise
		# gauss = np.random.normal(0., 0.5, dim)
		# gauss = gauss.reshape(dim[1], dim[0])
		# cv_img = np.array(cv_img, dtype=np.float32)
		# cv_img = cv_img + gauss
		# cv_img[cv_img<0.00001] = 0.

		# # smoothing
		# kernel = np.ones((4,4),np.float32)/16
		# cv_img = cv2.filter2D(cv_img,-1,kernel)


		cv_img = np.array(cv_img, dtype=np.float32)
		cv_img*=(10./255.)

		# cv2 image to ros image and publish
		try:
			resized_img = self.bridge.cv2_to_imgmsg(cv_img, "passthrough")
		except Exception as e:
			raise e
		self.resized_depth_img.publish(resized_img)
		return(cv_img/5.)

	def GetRGBImageObservation(self):
		# ros image to cv2 image
		try:
			cv_img = self.bridge.imgmsg_to_cv2(self.rgb_image, "bgr8")
		except Exception as e:
			raise e
		# resize
		dim = (self.rgb_image_size[0], self.rgb_image_size[1])
		cv_resized_img = cv2.resize(cv_img, dim, interpolation = cv2.INTER_AREA)
		# cv2 image to ros image and publish
		try:
			resized_img = self.bridge.cv2_to_imgmsg(cv_resized_img, "bgr8")
		except Exception as e:
			raise e
		self.resized_rgb_img.publish(resized_img)
		return(cv_resized_img)

	def PublishDepthPrediction(self, depth_img):
		# cv2 image to ros image and publish
		cv_img = np.array(depth_img, dtype=np.float32)
		try:
			resized_img = self.bridge.cv2_to_imgmsg(cv_img, "passthrough")
		except Exception as e:
			raise e
		self.resized_depth_img.publish(resized_img)

	def GetLaserObservation(self):
		scan = copy.deepcopy(self.scan)
		scan[np.isnan(scan)] = 30.
		return scan

	def GetSelfState(self):
		return self.self_state;

	def GetSelfLinearXSpeed(self):
		return self.self_linear_x_speed

	def GetSelfOdomeSpeed(self):
		v = np.sqrt(self.self_linear_x_speed**2 + self.self_linear_y_speed**2)
		return [v, self.self_rotation_z_speed]

	def GetTargetState(self, name):
		return self.object_state[self.TargetName.index(name)]

	def GetSelfSpeed(self):
		return np.array(self.self_speed)

	def GetBump(self):
		return self.bump

	def SetObjectPose(self, name='mobile_base', random_flag=False):
		quaternion = tf.transformations.quaternion_from_euler(0., 0., np.random.uniform(-np.pi, np.pi))
		if name is 'mobile_base' :
			object_state = copy.deepcopy(self.set_self_state)
			object_state.pose.orientation.x = quaternion[0]
			object_state.pose.orientation.y = quaternion[1]
			object_state.pose.orientation.z = quaternion[2]
			object_state.pose.orientation.w = quaternion[3]
		else:
			object_state = self.States2State(self.default_states, name)

		self.set_state.publish(object_state)

	def States2State(self, states, name):
		to_state = ModelState()
		from_states = copy.deepcopy(states)
		idx = from_states.name.index(name)
		to_state.model_name = name
		to_state.pose = from_states.pose[idx]
		to_state.twist = from_states.twist[idx]
		to_state.reference_frame = 'world'
		return to_state


	def ResetWorld(self):
		self.SetObjectPose() # reset robot
		for x in xrange(len(self.object_name)): 
			self.SetObjectPose(self.object_name[x]) # reset target
		self.self_speed = [.4, 0.0]
		self.step_target = [0., 0.]
		self.step_r_cnt = 0.
		self.start_time = time.time()
		rospy.sleep(0.5)

	def Control(self, action):
		if action <2:
			self.self_speed[0] = self.action_table[action]
			# self.self_speed[1] = 0.
		else:
			self.self_speed[1] = self.action_table[action]
		move_cmd = Twist()
		move_cmd.linear.x = self.self_speed[0]
		move_cmd.linear.y = 0.
		move_cmd.linear.z = 0.
		move_cmd.angular.x = 0.
		move_cmd.angular.y = 0.
		move_cmd.angular.z = self.self_speed[1]
		self.cmd_vel.publish(move_cmd)

	def shutdown(self):
		# stop turtlebot
		rospy.loginfo("Stop Moving")
		self.cmd_vel.publish(Twist())
		rospy.sleep(1)

	def GetRewardAndTerminate(self, t):
		terminate = False
		reset = False
		[v, theta] = self.GetSelfOdomeSpeed()
		laser = self.GetLaserObservation()
		reward = v * np.cos(theta) * 0.2 - 0.01

		if self.GetBump() or np.amin(laser) < 0.46 or np.amin(laser) == 30.:
			reward = -10.
			terminate = True
			reset = True
		if t > 500:
			reset = True

		return reward, terminate, reset	
	
		