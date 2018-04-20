from __future__ import print_function
from RealWorld import RealWorld

import tensorflow as tf
import random
import numpy as np
import time
import rospy
import models
import cv2

ACTIONS = 7 # number of valid actions
SPEED = 2 # DoF of speed
GAMMA = 0.99 # decay rate of past observations
OBSERVE = 10. # timesteps to observe before training
EXPLORE = 20000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.001 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 8 # size of minibatch
MAX_EPISODE = 20000
DEPTH_IMAGE_WIDTH = 160
DEPTH_IMAGE_HEIGHT = 128
RGB_IMAGE_HEIGHT = 228
RGB_IMAGE_WIDTH = 304
CHANNEL = 3
TAU = 0.001 # Rate to update target network toward primary network
H_SIZE = 8*10*64
IMAGE_HIST = 4

def variable_summaries(var):
	"""Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
	with tf.name_scope('summaries'):
		mean = tf.reduce_mean(var)
	tf.summary.scalar('mean', mean)
	with tf.name_scope('stddev'):
		stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
	tf.summary.scalar('stddev', stddev)
	tf.summary.scalar('max', tf.reduce_max(var))
	tf.summary.scalar('min', tf.reduce_min(var))
	tf.summary.histogram('histogram', var)
		
def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev = 0.01)
	return tf.Variable(initial, name="weights")

def bias_variable(shape):
	initial = tf.constant(0., shape = shape)
	return tf.Variable(initial, name="bias")

def conv2d(x, W, stride_h, stride_w):
	return tf.nn.conv2d(x, W, strides = [1, stride_h, stride_w, 1], padding = "SAME")


class QNetwork(object):
	"""docstring for ClassName"""
	def __init__(self, sess, depth_predict):
		# network weights
		# input 128x160x1
		with tf.name_scope("Conv1"):
			W_conv1 = weight_variable([10, 14, IMAGE_HIST, 32])
			variable_summaries(W_conv1)
			b_conv1 = bias_variable([32])
		# 16x20x32
		with tf.name_scope("Conv2"):
			W_conv2 = weight_variable([4, 4, 32, 64])
			variable_summaries(W_conv2)
			b_conv2 = bias_variable([64])
		# 8x10x64
		with tf.name_scope("Conv3"):
			W_conv3 = weight_variable([3, 3, 64, 64])
			variable_summaries(W_conv3)
			b_conv3 = bias_variable([64])
		# 8x10x64
		with tf.name_scope("FCValue"):
			W_value = weight_variable([H_SIZE, 512])
			variable_summaries(W_value)
			b_value = bias_variable([512])
			# variable_summaries(b_ob_value)

		with tf.name_scope("FCAdv"):
			W_adv = weight_variable([H_SIZE, 512])
			variable_summaries(W_adv)
			b_adv = bias_variable([512])
			# variable_summaries(b_adv)

		with tf.name_scope("FCValueOut"):
			W_value_out = weight_variable([512, 1])
			variable_summaries(W_value_out)
			b_value_out = bias_variable([1])
			# variable_summaries(b_ob_value_out)

		with tf.name_scope("FCAdvOut"):
			W_adv_out = weight_variable([512, ACTIONS])
			variable_summaries(W_adv_out)
			b_adv_out = bias_variable([ACTIONS])
			# variable_summaries(b_ob_adv_out)	

		# input layer
		self.state = tf.placeholder("float", [None, DEPTH_IMAGE_HEIGHT, DEPTH_IMAGE_WIDTH, IMAGE_HIST])
		# Conv1 layer
		h_conv1 = tf.nn.relu(conv2d(self.state, W_conv1, 8, 8) + b_conv1)
		# print('conv1:', h_conv1)
		# Conv2 layer
		h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2, 2, 2) + b_conv2)
		# print('conv2:', h_conv2)
		# Conv2 layer
		h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1, 1) + b_conv3)
		# print('conv3:', h_conv3)
		h_conv3_flat = tf.reshape(h_conv3, [-1, H_SIZE])

		# FC ob value layer
		h_fc_value = tf.nn.relu(tf.matmul(h_conv3_flat, W_value) + b_value)
		value = tf.matmul(h_fc_value, W_value_out) + b_value_out

		# FC ob adv layer
		h_fc_adv = tf.nn.relu(tf.matmul(h_conv3_flat, W_adv) + b_adv)		
		advantage = tf.matmul(h_fc_adv, W_adv_out) + b_adv_out
		
		# Q = value + (adv - advAvg)
		advAvg = tf.expand_dims(tf.reduce_mean(advantage, axis=1), axis=1)
		advIdentifiable = tf.subtract(advantage, advAvg)
		self.readout = tf.add(value, advIdentifiable)

		# define the ob cost function
		self.a = tf.placeholder("float", [None, ACTIONS])
		self.y = tf.placeholder("float", [None])
		self.readout_action = tf.reduce_sum(tf.multiply(self.readout, self.a), axis=1)
		self.td_error = tf.square(self.y - self.readout_action)
		self.cost = tf.reduce_mean(self.td_error)
		self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.cost)

def updateTargetGraph(tfVars,tau):
	total_vars = len(tfVars)
	op_holder = []
	for idx,var in enumerate(tfVars[0:total_vars/2]):
		op_holder.append(tfVars[idx+total_vars/2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars/2].value())))
	return op_holder

def updateTarget(op_holder,sess):
	for op in op_holder:
		sess.run(op)

def TestNetwork():
	sess = tf.InteractiveSession()
	# define depth_net 
	depth_state = tf.placeholder("float", [None, RGB_IMAGE_HEIGHT, RGB_IMAGE_WIDTH, CHANNEL])
	depth_net = models.ResNet50UpProj({'data': depth_state}, 1, 1, True)
	depth_predict = tf.divide(depth_net.get_output(), 5.)
	depth_net_var = tf.trainable_variables()
	print("  [*] printing DepthNet variables")
	for idx, v in enumerate(depth_net_var):
		print("  var {:3}: {:15}   {}".format(idx, str(v.get_shape()), v.name))
	depth_net_saver = tf.train.Saver(depth_net_var, max_to_keep=1)

	# define q network
	with tf.device("/cpu:0"):
		with tf.name_scope("OnlineNetwork"):
			online_net = QNetwork(sess, depth_predict)
		with tf.name_scope("TargetNetwork"):
			target_net = QNetwork(sess, depth_predict)
	rospy.sleep(1.)

	reward_var = tf.Variable(0., trainable=False)
	reward_epi = tf.summary.scalar('reward', reward_var)
	# define summary
	merged_summary = tf.summary.merge_all()
	summary_writer = tf.summary.FileWriter('./logs', sess.graph)

	# Initialize the World and variables
	env = RealWorld()
	print('Environment initialized')
	# init_op = tf.global_variables_initializer()
	# sess.run(init_op)

	# get the first state 
	rgb_img_t1 = env.GetRGBImageObservation()
	terminal = False

	# Loading depth network
	depth_checkpoint = tf.train.get_checkpoint_state('saved_networks/depth')
	if depth_checkpoint and depth_checkpoint.model_checkpoint_path:
		print('Loading from checkpoint:', depth_checkpoint)
		depth_net_saver.restore(sess, depth_checkpoint.model_checkpoint_path)
		print("Depth network model loaded:", depth_checkpoint.model_checkpoint_path)
	
	# saving and loading Q networks
	episode = 0
	q_net_params = []

	# Find variables of q network
	for variable in tf.trainable_variables():
		variable_name = variable.name
		if variable_name.find('OnlineNetwork') >= 0:
			q_net_params.append(variable)
	for variable in tf.trainable_variables():
		variable_name = variable.name
		if variable_name.find('TargetNetwork') >= 0:
			q_net_params.append(variable)
	print("  [*] printing QNetwork variables")
	for idx, v in enumerate(q_net_params):
		print("  var {:3}: {:15}   {}".format(idx, str(v.get_shape()), v.name))
	q_net_saver = tf.train.Saver(q_net_params, max_to_keep=1)

	checkpoint = tf.train.get_checkpoint_state("saved_networks/Q")
	print('checkpoint:', checkpoint)
	if checkpoint and checkpoint.model_checkpoint_path:
		q_net_saver.restore(sess, checkpoint.model_checkpoint_path)
		print("Q network successfully loaded:", checkpoint.model_checkpoint_path)
	else:
		print("Could not find old Q network weights")
		
	# start training
	epsilon = INITIAL_EPSILON
	r_epi = 0.
	r_cache = []
	random_flag = False
	t_max = 0
	t = 0
	rate = rospy.Rate(5)
	loop_time = time.time()
	last_loop_time = loop_time
	while not rospy.is_shutdown():
		episode += 1
		env.ResetWorld()
		t = 0
		terminal = False
		reset = False
		action_index = 0
		# first observation
		rgb_img_t = env.GetRGBImageObservation()
		depth_img_t1 = sess.run(depth_predict, feed_dict={depth_state: [rgb_img_t]})[0]
		depth_img_t1 = np.reshape(depth_img_t1, (DEPTH_IMAGE_HEIGHT, DEPTH_IMAGE_WIDTH))
		depth_imgs_t1 = np.stack((depth_img_t1, depth_img_t1, depth_img_t1, depth_img_t1), axis=2)
		while not rospy.is_shutdown():
			rgb_img_t1 = env.GetRGBImageObservation()
			reward_t, terminal, reset = env.GetRewardAndTerminate(t)
			depth_imgs_t = depth_imgs_t1
			rgb_img_t = rgb_img_t1

			depth_img_t1 = sess.run(depth_predict, feed_dict={depth_state: [rgb_img_t1]})[0]

			env.PublishDepthPrediction(depth_img_t1*5.)

			depth_img_t1 = np.reshape(depth_img_t1, (DEPTH_IMAGE_HEIGHT, DEPTH_IMAGE_WIDTH, 1))
			depth_imgs_t1 = np.append(depth_img_t1, depth_imgs_t1[:, :, :(IMAGE_HIST - 1)], axis=2)

			# choose an action epsilon greedily
			a = sess.run(online_net.readout, feed_dict = {online_net.state : [depth_imgs_t1]})
			readout_t = a[0]
			a_t = np.zeros([ACTIONS])
			action_index = np.argmax(readout_t)
			a_t[action_index] = 1
			# Control the agent
			env.Control(action_index)
			print('action:', action_index)

			t += 1
			last_loop_time = loop_time
			loop_time = time.time()
			rate.sleep()


def main():
	TestNetwork()

if __name__ == "__main__":
	main()

