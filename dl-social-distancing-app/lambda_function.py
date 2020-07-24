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
from threading import Thread, Event

iotTopic = '$aws/things/{}/infer'.format(os.environ['AWS_IOT_THING_NAME'])

modelPath = "/opt/awscam/artifacts"
# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([modelPath, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([modelPath, "yolov3.weights"])
configPath = os.path.sep.join([modelPath, "yolov3.cfg"])
client = greengrasssdk.client('iot-data')

class LocalDisplay(Thread):
    """ Class for facilitating the local display of inference results
        (as images). The class is designed to run on its own thread. In
        particular the class dumps the inference results into a FIFO
        located in the tmp directory (which lambda has access to). The
        results can be rendered using mplayer by typing:
        mplayer -demuxer lavf -lavfdopts format=mjpeg:probesize=32 /tmp/results.mjpeg
    """
    def __init__(self, resolution):
        """ resolution - Desired resolution of the project stream """
        # Initialize the base class, so that the object can run on its own
        # thread.
        super(LocalDisplay, self).__init__()
        # List of valid resolutions
        RESOLUTION = {'1080p' : (1920, 1080), '720p' : (1280, 720), '480p' : (858, 480)}
        if resolution not in RESOLUTION:
            raise Exception("Invalid resolution")
        self.resolution = RESOLUTION[resolution]
        # Initialize the default image to be a white canvas. Clients
        # will update the image when ready.
        self.frame = cv2.imencode('.jpg', 255*np.ones([640, 480, 3]))[1]
        self.stop_request = Event()

    def run(self):
        """ Overridden method that continually dumps images to the desired
            FIFO file.
        """
        # Path to the FIFO file. The lambda only has permissions to the tmp
        # directory. Pointing to a FIFO file in another directory
        # will cause the lambda to crash.
        result_path = '/tmp/results.mjpeg'
        # Create the FIFO file if it doesn't exist.
        if not os.path.exists(result_path):
            os.mkfifo(result_path)
        # This call will block until a consumer is available
        with open(result_path, 'wb') as fifo_file:
            while not self.stop_request.isSet():
                try:
                    # Write the data to the FIFO file. This call will block
                    # meaning the code will come to a halt here until a consumer
                    # is available.
                    fifo_file.write(self.frame.tobytes())
                except IOError:
                    continue

    def set_frame_data(self, frame):
        """ Method updates the image data. This currently encodes the
            numpy array to jpg but can be modified to support other encodings.
            frame - Numpy array containing the image data of the next frame
                    in the project stream.
        """
        ret, jpeg = cv2.imencode('.jpg', cv2.resize(frame, self.resolution))
        if not ret:
            raise Exception('Failed to set frame data')
        self.frame = jpeg

    def join(self):
        self.stop_request.set()


def lambda_handler(event, context):
    return

def infinite_infer_run():
	# load our YOLO object detector trained on COCO dataset (80 classes)
	print("[INFO] loading YOLO from disk...")
	net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
	# set up local display
	local_display = LocalDisplay('480p')
	local_display.start()
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
		if len(results) >= -1:
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


			for (i, (prob, bbox, centroid)) in enumerate(results):
				# extract the bounding box and centroid coordinates, then
				# initialize the color of the annotation
				(startX, startY, endX, endY) = bbox
				(cX, cY) = centroid
				color = (0, 255, 0)

				# if the index pair exists within the violation set, then
				# update the color
				if i in violate:
					color = (0, 0, 255)

				# draw (1) a bounding box around the person and (2) the
				# centroid coordinates of the person,
				cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
				cv2.circle(frame, (cX, cY), 5, color, 1)

			# draw the total number of social distancing violations on the
			# output frame
			text = "Social Distancing Violations: {}".format(len(violate))
			cv2.putText(frame, text, (10, frame.shape[0] - 25),
						cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)
			print('draw frame!')
			local_display.set_frame_data(frame)


		# draw the total number of social distancing violations on the
		# output frame
		# local_display.set_frame_data(frame)
		msg ='{ "violation":'+str(len(violate))+'}'
		client.publish(topic=iotTopic, payload=msg)
	threading.Timer(15, infinite_infer_run).start()

infinite_infer_run()





