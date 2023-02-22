import numpy as np
import cv2
import depthai as dai

SHAPE = 32

p = dai.Pipeline()
p.setOpenVINOVersion(dai.OpenVINO.VERSION_2021_4)

camRgb = p.create(dai.node.ColorCamera)
camRgb.setPreviewSize(SHAPE, SHAPE)
camRgb.setInterleaved(False)


nn = p.create(dai.node.NeuralNetwork)
nn.setBlobPath("models/resnet_openvino_2021.4_8shave.blob")
camRgb.preview.link(nn.input)


nn_xout = p.create(dai.node.XLinkOut)
nn_xout.setStreamName("nn")
nn.out.link(nn_xout.input)

rgb_xout = p.create(dai.node.XLinkOut)
rgb_xout.setStreamName("rgb")
camRgb.preview.link(rgb_xout.input)


with dai.Device(p) as device:
    qNn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)
    qCam = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

    count = 0
    while True:
        # cv2.imshow("Color", qCam.get().getCvFrame())
        # if cv2.waitKey(1) == ord('q'):
        #     break

        result = np.array(qNn.get().getData())
        print(result)
        print("--------------")
        count +=1
        if count == 5:
            break