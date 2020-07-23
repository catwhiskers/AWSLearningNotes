import json
import awscam
import mo
import cv2
import greengrasssdk
import os
import onnx
import onnxruntime
import numpy as np
import cv2 


modelPath = "/opt/awscam/artifacts"
iotTopic = '$aws/things/{}/infer'.format(os.environ['AWS_IOT_THING_NAME'])
PATH_TO_CKPT = os.path.join(modelPath,'yolov5s.onnx')
client = greengrasssdk.client('iot-data')


def lambda_handler(event, context):
    return

def infinite_infer_run():
    session = onnxruntime.InferenceSession(PATH_TO_CKPT)
    """ Run the DeepLens inference loop frame by frame"""
    # Load the model here
    while True:
        # Get a frame from the video stream
        ret, frame = awscam.getLastFrame()
        if ret == False:
            raise Exception("Failed to get frame from the stream")
                            
        img = cv2.resize(frame, dsize=(640, 640), interpolation=cv2.INTER_CUBIC)
        res = img[:, :, ::-1].transpose(2, 0, 1)
        res = np.expand_dims( res ,axis=0).astype(np.float32)    

        # Perform the actual detection by running the model with the image as input
        outcome = session.run(None, {"images":res})
        #only want inferences that have a prediction score of 50% and higher
        msg = '{'
        for idx, val in enumerate(outcome):
            msg += str(val) 
        msg = msg.rstrip(',')
        msg +='}'
            
        client.publish(topic=iotTopic, payload = msg)
            

    # Asynchronously schedule this function to be run again in 15 seconds
    Timer(15, greengrass_infinite_infer_run).start()
        
infinite_infer_run()




