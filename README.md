# Towards Monocular Vision based Obstacle Avoidance through Deep Reinforcement Learning

By Linhai Xie, Sen Wang, Niki trigoni, Andrew Markham.

The tensorflow implmentation for the paper: [Towards Monocular Vision based Obstacle Avoidance through Deep Reinforcement Learning](https://arxiv.org/abs/1706.09829)

## Contents
0. [Introduction](#introduction)
0. [Prerequisite](#Prerequisite)
0. [Instruction](#instruction)
0. [Citation](#citation)

## Introduction

This repository contains:

1.Training code for [FCRN](https://arxiv.org/abs/1606.00373). We write our own training code but build the mode directly with the code provided [here](https://github.com/iro-cp/FCRN-DepthPrediction). (We retain Iro's license in the repository)

2.Data preprocessing code for training FCRN.

3.Training code for D3QN(Double Deep Q Network with Dueling architecture) with a turtlebot2 in Gazebo simulator.

4.Testing code for D3QN with a turtlebot2 in real world

5.The interface code between tensorflow and ros

The network model for D3QN is slightly different from the paper as we find this version has a better performance.

The vedio of our real world experiments is available at [Youtube](https://www.youtube.com/watch?v=qNIVgG4RUDM)

## Prerequisites

Tensorflow > 1.1

ROS Kinetic

cv2

## Instruction
**Retraining FCRN**

We have a example dataset collected with a turtlebot in folder ```/Depth/data``` which contains 1k labeled rgb-depth images. 
We recommand to collected more than 10k images to retrain a good model based on the initial model.

After collecting enough data, use `DataPreprocess.py` to generate training and testing datasets.

Download the [initial model of FCRN](http://campar.in.tum.de/files/rupprecht/depthpred/NYU_FCRN-checkpoint.zip) into the folder '''/Depth/init_network'''

Then use `training.py` in ```/Depth/FCRN``` to retrain the model.

Finally copy the generated checkpoint and model file from ```/Depth/saved_network``` to ```/D3QN/saved_networks/depth```

**Training D3QN in Gazebo**

Launch the Gazebo world with our world file with command:

```roslaunch turtlebot_gazebo turtlebot_world.launch world_file:=/PATH TO/Monocular-Obstacle-Avoidance/D3QN/world/SquareWorld.world```

Start training with:

```python D3QN_training```

**Testing D3QN in real world**

```python D3QN_testing```

## Citation

If you use this method in your research, please cite:

	@inproceedings{xie2017towards,
		  title = "Towards Monocular Vision based Obstacle Avoidance through Deep Reinforcement Learning",
		  author = "Xie, Linhai and Wang, Sen and Markham, Andrew and Trigoni, Niki",
		  year = "2017",
		  booktitle = "RSS 2017 workshop on New Frontiers for Deep Learning in Robotics",
	}



