import depthai
import time
import warnings
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore")

def main():
    pipeline = depthai.Pipeline()

    model_path = "models/DUQ_LeNet.blob"
    condition_name = "oakd_time/DUQ_LeNet"
    nn = pipeline.createNeuralNetwork()
    nn.setBlobPath(model_path)

    cam = pipeline.createColorCamera()
    cam.setPreviewSize(28, 28)  #size for inference
    cam.setInterleaved(False)
    cam.setFps(30)

    cam.preview.link(nn.input)

    xout_nn = pipeline.createXLinkOut()
    xout_nn.setStreamName('nn')
    nn.out.link(xout_nn.input)


    time_elapsed_runs = []
    with depthai.Device(pipeline) as device:
        nn_queue = device.getOutputQueue(name='nn', maxSize=1, blocking=False)

        for _ in range(2):
            nn_queue.get()

        for i in range(102):  
            start_time = time.perf_counter()

            device.startPipeline()
            end_time = time.perf_counter()

            in_nn = nn_queue.get()

            inference_time = end_time - start_time
            time_elapsed = round((inference_time*1000),3)
            print(time_elapsed)
            if i > 1:
                time_elapsed_runs.extend([time_elapsed])

        time_dict = {"time":np.array(time_elapsed_runs)}
        time_df = pd.DataFrame(time_dict)
        time_df.to_csv(path_or_buf= condition_name+'_quant'+'_time.csv')
        print("completed")

if __name__ == "__main__":
    main()
