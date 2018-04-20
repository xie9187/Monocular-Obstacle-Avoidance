import tensorflow as tf 
import numpy as np 
import cv2
import pickle
import random
import models
import copy
import time
import math
import os

from cv_bridge import CvBridge, CvBridgeError
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

CHANNEL = 3
DEPTH_IMAGE_WIDTH = 160
DEPTH_IMAGE_HEIGHT = 128
RGB_IMAGE_WIDTH = 304
RGB_IMAGE_HEIGHT = 228
MAX_STEP = 200
MAX_EPOCH = 150
BATCH = 16

def output_predict(predict, kinect, rgb, epoch, step):
	max_val = 10.
	kinect[kinect>max_val] = max_val
	if np.max(kinect) != 0:
		kinect_save = (kinect/max_val)*255.0
		# print('kinect_max', np.amax(kinect))
	else:
		kinect_save = kinect*255.0
	kinect_save=np.uint8(kinect_save)
	name = "data/%04d" % epoch + "_%04d_kinect.png" % step
	cv2.imwrite(name,kinect_save)

	predict[predict>max_val] = max_val
	if np.max(predict) != 0:
		predict_save = (predict/max_val)*255.0
		# print('predict_max', np.amax(predict))
	else:
		predict_save = predict*255.0
	predict_save=np.uint8(predict_save)
	name = "data/%04d" % epoch + "_%04d_predicted.png" % step
	cv2.imwrite(name,predict_save)

	name = "data/%04d" % epoch + "_%04d_rgb.png" % step
	cv2.imwrite(name,rgb)

def SetDiff(first, second):
	second = set(second)
	return [item for item in first if item not in second]

def normalize_rgb(rgb_images, value):
	rgb_images = np.asarray(rgb_images).astype(float)
	rgb_images /= 255.
	for x in xrange(3):
		rgb_images[:, :, :, x] -= value[x]
	return rgb_images

def consecutive_sample(data, start, end):
	# return a list
	part = []
	for x in xrange(start, end):
		part.append(data[x])
	return part

with tf.Session() as sess:
	# Construct network and define loss function
	state = tf.placeholder("float", [None, RGB_IMAGE_HEIGHT, RGB_IMAGE_WIDTH, CHANNEL])
	net = models.ResNet50UpProj({'data': state}, BATCH, 1, True)
	depth_predict = net.get_output()
	depth_kinect = tf.placeholder("float", [None, DEPTH_IMAGE_HEIGHT, DEPTH_IMAGE_WIDTH, 1])
	img_mask = tf.placeholder("float", [None, DEPTH_IMAGE_HEIGHT, DEPTH_IMAGE_WIDTH, 1])

	print 'Loading initial network param'
	init_saver = tf.train.Saver()     
	init_saver.restore(sess, '../init_network/NYU_FCRN.ckpt')

	d_show = tf.subtract(tf.multiply(depth_predict, img_mask), tf.multiply(depth_kinect, img_mask))
	abs_d_show = tf.abs(d_show)
	c = tf.divide(tf.reduce_max(abs_d_show), 5.)
	berHu = tf.where(tf.less_equal(abs_d_show, c), abs_d_show, tf.square(d_show))
	loss = tf.reduce_mean(tf.reduce_mean(berHu, 1))

	train_step = tf.train.AdamOptimizer(5e-5).minimize(loss)

	train_loss_var = tf.Variable(0., trainable=False)
	train_loss_sum = tf.summary.scalar('training_loss', train_loss_var)
	test_loss_var = tf.Variable(0., trainable=False)
	test_loss_sum = tf.summary.scalar('testing_loss', test_loss_var)
	merged_summary = tf.summary.merge_all()
	summary_writer = tf.summary.FileWriter('./logs', sess.graph)

	print 'Initializing var'
	uninitialized_vars = []
	start_time = time.time()
	for var in tf.global_variables():
		try:
			sess.run(var)
		except tf.errors.FailedPreconditionError:               
			uninitialized_vars.append(var)
	init_new_vars_op = tf.variables_initializer(uninitialized_vars)
	print("  [*] printing unitialized variables")
	for idx, v in enumerate(uninitialized_vars):
		print("  var {:3}: {:15}   {}".format(idx, str(v.get_shape()), v.name))
	sess.run(init_new_vars_op)
	print 'Var initialized, time:', time.time() - start_time

	trainable_var = tf.trainable_variables()
	print "  [*] printing trainable variables"
	for idx, v in enumerate(trainable_var):
		print("  var {:3}: {:15}   {}".format(idx, str(v.get_shape()), v.name))

	depth_net_saver = tf.train.Saver(trainable_var, max_to_keep=1)
	checkpoint = tf.train.get_checkpoint_state('saved_network')
	if checkpoint and checkpoint.model_checkpoint_path:
		print 'Loading from checkpoint:', checkpoint
		depth_net_saver.restore(sess, checkpoint.model_checkpoint_path)
		print "Depth network model loaded:", checkpoint.model_checkpoint_path
	else:
		print 'No new model'

	print 'Loading data'
	training_data = pickle.load(open('../rgb_depth_images_training_real.p', "rb"))
	testing_data = pickle.load(open('../rgb_depth_images_testing_real.p', "rb"))
	print 'Data loaded'

	Step = 0
	for epoch in xrange(1,MAX_EPOCH+1):
		np.random.shuffle(training_data)
		np.random.shuffle(testing_data)
		for step in xrange(1,int(len(training_data)/(BATCH))+1):
			start_time = time.time()
			training_batch_real = consecutive_sample(training_data, (step-1)*BATCH, step*BATCH)

			rgb_img = [d[0] for d in training_batch_real]
			depth_img = [d[1] for d in training_batch_real]
			depth_img = np.reshape(depth_img, [-1, DEPTH_IMAGE_HEIGHT, DEPTH_IMAGE_WIDTH, 1])
			mask = [d[2] for d in training_batch_real]
			mask = np.reshape(mask, [-1, DEPTH_IMAGE_HEIGHT, DEPTH_IMAGE_WIDTH, 1])

			# flip
			if np.random.rand() > 0.5:
				rgb_img = np.flip(rgb_img, axis=2)
				depth_img = np.flip(depth_img, axis=2)
				mask = np.flip(mask, axis=2)

			training_loss = 0.
			depth_predict_value, _, training_loss = sess.run([depth_predict, train_step, loss], 
				 						feed_dict = { state : rgb_img,
													  depth_kinect : depth_img,
													  img_mask : mask})

			if step % 10 == 0:
				testing_batch = random.sample(testing_data, BATCH)
				rgb_img_test = [d[0] for d in testing_batch]
				depth_img_test = [d[1] for d in testing_batch]
				depth_img_test = np.reshape(depth_img_test, [-1, DEPTH_IMAGE_HEIGHT, DEPTH_IMAGE_WIDTH, 1])
				mask_test = [d[2] for d in testing_batch]
				mask_test = np.reshape(mask_test, [-1, DEPTH_IMAGE_HEIGHT, DEPTH_IMAGE_WIDTH, 1])
				depth_predict_value, testing_loss, difference= sess.run([depth_predict, loss, abs_d_show], 
																  feed_dict = { state : rgb_img_test,
																				depth_kinect : depth_img_test,
																				img_mask : mask_test})
				# if epoch % 10 == 0:
				output_predict(depth_predict_value[0], depth_img_test[0], np.asarray(rgb_img_test)[0], epoch, step)
				# print("epoch: {:2} | step: {:3} | traning loss: {:.4f} | testing loss: {:.4f}, time: {:.2f}"\
				# 		.format(epoch, step, training_loss, testing_loss, (time.time()-start_time)/60.))
				summary_str = sess.run(merged_summary, feed_dict={train_loss_var: training_loss, 
																  test_loss_var: testing_loss})
				summary_writer.add_summary(summary_str, Step)
				Step = Step + 1

		# full testing
		testing_loss_buff = []
		for step in xrange(int(len(testing_data)/(BATCH))):
			testing_batch = consecutive_sample(testing_data, (step-1)*BATCH, step*BATCH)
			rgb_img_test = [d[0] for d in testing_batch]
			depth_img_test = [d[1] for d in testing_batch]
			mask_test = [d[2] for d in testing_batch]
			mask_test = np.reshape(mask_test, [-1, DEPTH_IMAGE_HEIGHT, DEPTH_IMAGE_WIDTH, 1])

			depth_predict_value, testing_loss, difference = sess.run([depth_predict, loss, abs_d_show], 
															  feed_dict = { state : rgb_img_test,
																			depth_kinect : depth_img_test,
																			img_mask : mask_test})
			testing_loss_buff.append(testing_loss)

		print("epoch: {:3} | traning loss: {:.4f} | testing loss: {:.4f}, time: {:.2f}"\
				.format(epoch, training_loss, np.mean(testing_loss_buff), (time.time()-start_time)/60.))
		if epoch % 30 == 0:
			depth_net_saver.save(sess, 'saved_network/DepthNet', global_step = epoch)
