import numpy as np
import cv2
import depthai as dai
import torch


p = dai.Pipeline()
p.setOpenVINOVersion(dai.OpenVINO.VERSION_2021_4)


nn = p.create(dai.node.NeuralNetwork)
nn.setBlobPath("models/iris.blob")




nn_xin = p.create(dai.node.XLinkIn)
nn_xin.setStreamName("nn_in")
nn_xin.out.link(nn.input)


nn_xout = p.create(dai.node.XLinkOut)
nn_xout.setStreamName("nn_out")
nn.out.link(nn_xout.input)

shape = (1,4)
a = np.ones(shape, dtype=np.int8)

with dai.Device(p) as device:
    infeed = device.getInputQueue('nn_in')
    outfeed = device.getOutputQueue(name="nn_out", maxSize=1, blocking=False)

    nn_data = dai.NNData()
    nn_data.setData(a)
    infeed.send(nn_data)

    result = outfeed.get()
    if result is not None:
        print(result)
        print("---------")
        detections = result.detections
        print(detections)
        print("hello")
    print("hai")