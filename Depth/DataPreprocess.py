import numpy as np 
import cv2
import rospy
import random
import copy
import pickle

from collections import deque
from pathlib import Path

def ismember(A, B):
    return [ np.sum(a == B) for a in A ]

def rgb_data_color_aug(rgb_images):
	rgb_images = np.asarray(rgb_images).astype(float)
	noise = np.random.rand(4) * 0.4 + 0.8
	print noise
	rgb_images = rgb_images * noise[0]
	for x in xrange(3):
		rgb_images[:, :, :, x] = rgb_images[:, :, :, x] * noise[x + 1]
	rgb_images[rgb_images>255.] = 255.
	rgb_images = np.uint8(rgb_images)
	return rgb_images

def crop_img(img):
	size = np.shape(img)
	alpha = 0.1
	height_start = int(size[0]*alpha/1.5)
	height_end = size[0] - int(size[0]*alpha/2.1)
	alpha = 0.15
	width_start = int(size[1]*alpha/2.5)
	width_end = size[1] - int(size[1]*alpha/1.8)
			
	if len(size) == 2:
		cropped_img = img[height_start : height_end, width_start : width_end]
	elif len(size) == 3:
		cropped_img = img[height_start : height_end, width_start : width_end, :]

	return cropped_img

file_num = 1000
D_training = deque()
D_testing = deque()
testing = 0.3
cnt = 0
path = './data/'
inpaint_flag = False


depth_dim = (160, 128)
rgb_dim = (304, 228)

for i in xrange(1, file_num + 1):
	print 'img', i
	depth_file = Path(path+'depth'+str(i)+'.png')
	rgb_file = Path(path+'rgb'+str(i)+'.jpg')
	if depth_file.is_file() and rgb_file.is_file():
		cv_depth_img = cv2.imread(path+'depth'+str(i)+'.png', 0)
		cv_depth_img = crop_img(cv_depth_img)
		cv_depth_img = np.array(cv_depth_img, dtype=np.float32)
		cv_depth_img = cv2.resize(cv_depth_img, depth_dim, interpolation = cv2.INTER_NEAREST)
		cv_depth_img *= (10./255.)
		
		mask = copy.deepcopy(cv_depth_img)
		mask[mask == 0.] = 1.
		mask[mask != 1.] = 0.
		mask = 1 - mask

		# print np.shape(cv_depth_img)
		# print np.mean(cv_depth_img)
		# cv2.normalize(cv_depth_img, cv_depth_img, 0.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
		# cv_depth_img=np.uint8(cv_depth_img*255.0)
		# cv2.imshow('depth image',cv_depth_img)
		# cv2.waitKey(0)
		# mask=np.uint8(mask*255.0)
		# cv2.imshow('mask',mask)
		# cv2.waitKey(0)
		
		cv_rgb_img = cv2.imread(path+'rgb'+str(i)+'.jpg')
		cv_rgb_img = crop_img(cv_rgb_img)
		cv_resized_rgb_img = cv2.resize(cv_rgb_img, rgb_dim, interpolation = cv2.INTER_AREA)
		same_flag = False
		cnt += 1
		if cnt == 1:
			last_rgb_img = cv_resized_rgb_img
			current_rgb_img = cv_resized_rgb_img

			last_depth_img = cv_depth_img
			current_depth_img = cv_depth_img

		else:
			last_rgb_img = current_rgb_img
			current_rgb_img = cv_resized_rgb_img

			if np.array_equal(current_rgb_img, last_rgb_img):
				print "The same rgb image"
				same_flag = True

			last_depth_img = current_depth_img
			current_depth_img = cv_depth_img
			
			if np.array_equal(current_depth_img, last_depth_img):
				print "The same depth image"
				same_flag = True

		save_rgb_img = current_rgb_img

		# cv2.imshow('rgb image',cv_resized_rgb_img)
		# cv2.waitKey(0)

		cv_depth_img = np.reshape(cv_depth_img, (depth_dim[1], depth_dim[0], 1))
		if not same_flag:
			if random.uniform(0, 1.0) > .1:
				D_training.append((save_rgb_img, cv_depth_img, mask))
				# D_training.append((np.flip(save_rgb_img, axis=1), np.flip(cv_depth_img, axis=1), np.flip(mask, axis=1)))
			else:
				D_testing.append((save_rgb_img, cv_depth_img, mask))
				# D_testing.append((np.flip(save_rgb_img, axis=1), np.flip(cv_depth_img, axis=1), np.flip(mask, axis=1)))

print 'rgb size:', cv_resized_rgb_img.shape
print 'depth size:', cv_depth_img.shape
pickle.dump(D_training, open("rgb_depth_images_training_real.p", "wb"))
pickle.dump(D_testing, open("rgb_depth_images_testing_real.p", "wb"))
print 'D_training:',len(D_training), '| D_testing:', len(D_testing)

