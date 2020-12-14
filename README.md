# Detection of persons in a Youtube video


<p align="center">    
    <a href='MISS_DIOR_The_new_Eau_de_Parfum_detect.mp4'>
        <img src="screenshot.png" width="50%"/>
    <a/>
<p/>

<p align="center">^^click here^^<p/>


## Objective

Given a video (here the [Dior - Eau de Parfum](https://www.youtube.com/watch?v=h4s0llOpKrU) commercial),
generate another video that shows the existence of humans within the original video by drawing boxes around them
for each frame.

## Methodology

I use the [YOLOv3](https://pjreddie.com/darknet/yolo/) model to detect the persons in the video.

To filter out false positive detections as much as possible, I keep only predictions that meet one of
the following conditions:
- the prediction confidence is above some threshold,
- the IoU score between the current bounding box and the bounding boxes of the previous frame is
    above some threshold.

The algorithm to detect the persons is pretty straightforward:
1. split the video into frames,
2. apply the YOLO detection system on batches of frames,
3. for each frame, draw a bounding box for each person detected if one of the above condition is satisfied.
4. recombine the frames into the final video.

## Implementation remarks

The input video is automatically downloaded in the same folder as the script `detect.py`,
and the output video is created at the same location.

For the YOLO detection system, I use the [Ultralytics open-source implemenation](https://github.com/ultralytics/yolov5), pretrained on the [COCO](https://cocodataset.org/#home) dataset.
The model is automatically loaded from [Pytorch Hub](https://github.com/ultralytics/yolov5) and
the weights are saved in the same folder as the script `detect.py`.

The inference time is much lower if a GPU is available.
In that case the batch size has to be adapted to the GPU memory size.

## How to run the detection

To generate the video, run the script `detect.py`.

But before doing it, set up a python virtual environment with __Python >= 3.8__ and __Pytorch >= 1.6__
(see details [here](https://pytorch.org/get-started/locally/)).
As mentioned earlier, the processing time will drastically improve if a GPU is available.
Then, install the required packages:
```
python -m pip install -U -r requirements.txt
```

## Result

To see the output video, click on the screenshot above.
