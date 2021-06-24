# YOffleNet + Deep Sort with PyTorch

<img src=".\images\demo.gif" alt="demo" style="zoom:50%;" />

## Introduction

This repository contains a moded version of  PyTorch YOLOv3 (https://github.com/ultralytics/yolov3) and Pytorch YOLOv3 + Deep Sort (https://github.com/mikel-brostrom/Yolov3_DeepSort_Pytorch). It is a program that counts the number of people and predicts the direction of movement using the bounding box coordinates obtained by tracking algorithm.

This project was carried out as a 2021-1 MIP study with advisor Ph.D Young-Keun Kim

## Description

The implementation is based on two papers:

- Simple Online and Realtime Tracking with a Deep Association Metric
https://arxiv.org/abs/1703.07402
- YOffleNet : Light-weight model of YOLOv4

## Requirements

Anaconda virtual environment was used to prevent library version conflicts used in other projects.

To implement the deep learning, GPU is necessary. My PC have CPU (Intel(R) Core(TM) i7-10750H) and GPU (NVIDIA GeForce GTX 1650 Ti). Install CUDA and cuDNN for your personal PC GPU version.

It is recommended to use the VS Code program for efficient debugging.

For the installation process of these three applications (Anaconda, CUDA & cuDNN, VS Code), please refer to the following link. Follow Step 1, 2 & 4

https://ykkim.gitbook.io/dlip/deep-learning-for-perception/installation-guide-for-deep-learning

## Tutorial 

##### GitHub

1. Go to the https://github.com/hkim1207/2021MIP
2. Download all the files in your local PC

##### Anaconda

1. open the Anaconda Prompt (anaconda3)
   
2. Change the current location to the directory where the Python file to be executed is in.

   `cd C:\Users\hkim\Documents\GitHub\2021MIP` 

3. Create a new conda environment

   `conda create -n YOffleNet_Tracking python=3.7` 

4. Activate the YOffleNet_Tracking environment

   `conda activate YOffleNet_Tracking`

   <img src=".\images\image-20210625005459680.png" alt="image-20210625005459680" style="zoom:50%;" />

5. Install pytorch and torch family. 

   `conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=10.2 -c pytorch`

   <img src=".\images\image-20210625015216696.png" alt="image-20210625015216696" style="zoom:50%;" />

6. Install the other libraries to run the track_YOffleNet.py

   `pip install -U -r requirements.txt`

   <img src=".\images\image-20210625010415772.png" alt="image-20210625010415772" style="zoom:50%;" />

## Tracking

`track.py` runs tracking on any video source:

```bash
python track_YOffleNet.py --weights YOffleNet.pt --source test.mp4 --device 0...
```

- Video:  `--source file.mp4`
- Webcam:  `--source 0`
- RTSP stream:  `--source rtsp://170.93.143.139/rtplive/470011e600ef003a004ee33696235daa`
- HTTP stream:  `--source http://wmccpinetop.axiscam.net/mjpg/video.mjpg`

If you don't want to make the commend long, you can set the default parameters in the code.

```python
parser.add_argument('--weights', nargs='+', type=str, default='YOffleNet/Weight/COCO/yolov4s.pt', help='model.pt path(s)')
parser.add_argument('--source', type=str, default='YOffleNet/data/videos/test.mp4', help='source')  
parser.add_argument('--output', type=str, default='YOffleNet_Out/yolov4s(cpu)', help='output folder')
parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
```

Then, you just commend

```bash
python track_YOffleNet.py
```

<img src=".\images\image-20210625015652131.png" alt="image-20210625015652131" style="zoom:50%;" />

## Info

If you have a question about the code or setting the environment, contact to me via e-mail

21700208@handong.edu
