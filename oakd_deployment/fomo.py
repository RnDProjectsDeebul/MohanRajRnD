#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np
import argparse
import time

'''
FastDepth demo running on device.
https://github.com/dwofk/fast-depth


Run as:
python3 -m pip install -r requirements.txt
python3 main.py

Onnx taken from PINTO0309, added scaling flag, and exported to blob:
https://github.com/PINTO0309/PINTO_model_zoo/tree/main/146_FastDepth
'''

# --------------- Arguments ---------------
# parser = argparse.ArgumentParser()
# parser.add_argument("-w", "--width", help="select model width for inference", default=320, type=int)

# args = parser.parse_args()

# choose width and height based on model
# if args.width == 320:
#     NN_WIDTH, NN_HEIGHT = 320, 256
# elif args.width == 640:
#     NN_WIDTH, NN_HEIGHT = 640, 480
# else:
#     raise ValueError(f"Width can be only 320 or 640, not {args.width}")

NN_WIDTH, NN_HEIGHT = 384, 512
NN_PATH = "models/fomo.blob"

# --------------- Pipeline ---------------
# Start defining a pipeline
pipeline = dai.Pipeline()
pipeline.setOpenVINOVersion(version = dai.OpenVINO.VERSION_2021_4)

# Define a neural network
detection_nn = pipeline.create(dai.node.NeuralNetwork)
detection_nn.setBlobPath(str(NN_PATH))
detection_nn.setNumPoolFrames(4)
detection_nn.input.setBlocking(False)
detection_nn.setNumInferenceThreads(2)

# Define camera
cam = pipeline.create(dai.node.ColorCamera)
cam.setPreviewSize(NN_WIDTH, NN_HEIGHT)
cam.setInterleaved(False)
cam.setFps(40)
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)

# Create outputs
xout_cam = pipeline.create(dai.node.XLinkOut)
xout_cam.setStreamName("cam")

xout_nn = pipeline.create(dai.node.XLinkOut)
xout_nn.setStreamName("nn")

# Link
cam.preview.link(detection_nn.input)
detection_nn.passthrough.link(xout_cam.input)
detection_nn.out.link(xout_nn.input)


# --------------- Inference ---------------
# Pipeline defined, now the device is assigned and pipeline is started
with dai.Device(pipeline) as device:

    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    q_cam = device.getOutputQueue("cam", 4, blocking=False)
    q_nn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

    start_time = time.time()
    counter = 0
    fps = 0
    layer_info_printed = False
    while True:
        in_frame = q_cam.get()
        in_nn = q_nn.get()

        frame = in_frame.getCvFrame()

        # Get output layer
        pred = np.array(in_nn.getFirstLayerFp16())

        # Scale depth to get relative depth
        d_max = np.max(pred)

        print(pred)
        print(pred.shape)
        break

        # Concatenate NN input and produced depth
        #cv2.imshow("Detections", cv2.hconcat([frame, d_max]))



        if cv2.waitKey(1) == ord('q'):
            break