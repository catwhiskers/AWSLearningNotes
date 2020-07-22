# USAGE
# python social_distance_detector.py --input pedestrians.mp4
# python social_distance_detector.py --input pedestrians.mp4 --output output.avi

# import the necessary packages
from pyimagesearch import social_distancing_config as config
from pyimagesearch.detection import detect_people
from scipy.spatial import distance as dist
import numpy as np
import argparse
import imutils
import cv2
import os
import awscam
import greengrasssdk
import threading

iotTopic = '$aws/things/{}/infer'.format(os.environ['AWS_IOT_THING_NAME'])

modelPath = "/opt/awscam/artifacts"
# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([modelPath, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([modelPath, "yolov3.weights"])
configPath = os.path.sep.join([modelPath, "yolov3.cfg"])
client = greengrasssdk.client('iot-data')

def lambda_handler(event, context):
    return

def infinite_infer_run():
	# load our YOLO object detector trained on COCO dataset (80 classes)
	print("[INFO] loading YOLO from disk...")
	net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

	# check if we are going to use GPU
	if config.USE_GPU:
		# set CUDA as the preferable backend and target
		print("[INFO] setting preferable backend and target to CUDA...")
		net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
		net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

	# determine only the *output* layer names that we need from YOLO
	ln = net.getLayerNames()
	ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

	# initialize the video stream and pointer to output video file
	print("[INFO] accessing video stream...")

	writer = None

	# loop over the frames from the video stream
	while True:
		# read the next frame from the file
		ret, frame = awscam.getLastFrame()
		# resize the frame and then detect people (and only people) in it
		frame = imutils.resize(frame, width=700)
		results = detect_people(frame, net, ln,
			personIdx=LABELS.index("person"))

		# initialize the set of indexes that violate the minimum social
		# distance
		violate = set()
		# ensure there are *at least* two people detections (required in
		# order to compute our pairwise distance maps)
		if len(results) >= 2:
			# extract all centroids from the results and compute the
			# Euclidean distances between all pairs of the centroids
			centroids = np.array([r[2] for r in results])
			D = dist.cdist(centroids, centroids, metric="euclidean")

			# loop over the upper triangular of the distance matrix
			for i in range(0, D.shape[0]):
				for j in range(i + 1, D.shape[1]):
					# check to see if the distance between any two
					# centroid pairs is less than the configured number
					# of pixels
					if D[i, j] < config.MIN_DISTANCE:
						# update our violation set with the indexes of
						# the centroid pairs
						violate.add(i)
						violate.add(j)



		# draw the total number of social distancing violations on the
		# output frame
		msg ='{ "violation":'+str(len(violate))+'}'
		client.publish(topic=iotTopic, payload=msg)
	threading.Timer(15, greengrass_infinite_infer_run).start()

infinite_infer_run()




