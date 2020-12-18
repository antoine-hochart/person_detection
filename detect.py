import os
import pytube
import cv2
import numpy as np
import torch
import moviepy.editor as moved

from time import time
from tqdm import tqdm

######################################################################
# GLOBAL PARAMETERS

URL = 'https://youtu.be/h4s0llOpKrU'
VID_DIR = os.path.dirname(__file__)
BATCH_SIZE = 16 # no. of frames to process simultaneously
CONF_THRESH = 0.45 # threshold for detection confidence
IOU_THRESH = 0.25 # threshold for IoU score between boxes in successive frames
MODEL = 'yolov5l' # 'yolov5X' where X = s, m, l, x

######################################################################
# UTILITY FUNCTIONS

def get_area(box):
    """ Compute area of box,
        assuming that box is non-empty if a < c and b < d
        where box = (a, b, c, d).
    """
    return max(box[2] - box[0], 0) * max(box[3] - box[1], 0)


def get_iou(box1, box2):
    """ Compute IoU score between box1 and box 2 """
    xa = max(box1[0], box2[0])
    ya = max(box1[1], box2[1])
    xb = min(box1[2], box2[2])
    yb = min(box1[3], box2[3])

    intersect = get_area((xa, ya, xb, yb))
    union = get_area(box1) + get_area(box2) - intersect

    return intersect / union


def get_max_iou(box, boxes):
    """ Compute the maximum IoU score between box and the elements in boxes """
    iou = [get_iou(box, b) for b in boxes]
    return max(iou, default=0)

######################################################################
# DOWNLOAD VIDEO

# instantiate YouTube object
yt = pytube.YouTube(URL)
# set video name
video_title = '_'.join(word for word in yt.title.split() if word.isalpha())
# set input/output file path
os.makedirs(VID_DIR, exist_ok=True)
fpath_in = os.path.join(VID_DIR, video_title + '.mp4')
fpath_out = os.path.join(VID_DIR, video_title + '_detect.mp4')
fpath_temp = os.path.join(VID_DIR, video_title + '_temp.mp4')
# download the Youtube video
if not os.path.isfile(fpath_in):
    print("Downloading video {}...".format(video_title))
    yt.streams.first().download(VID_DIR)
    os.rename(os.path.join(VID_DIR, yt.title+'.mp4'), fpath_in)
    print("Done")
else:
    print("Video {} already downloaded".format(video_title))

######################################################################
# LOAD YOLOv3 MODEL
print("\nLoading YOLOv3 model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
model = torch.hub.load('ultralytics/yolov5', MODEL, pretrained=True)
model = model.autoshape().to(device).eval()

######################################################################
# EXTRACT VIDEO FRAMES

# set video source
cap = cv2.VideoCapture(fpath_in)

# get video metrics
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# extract video frames
print("\nExtracting video frames...")
frames = []
for _ in tqdm(range(n_frames), ascii=True):
    ret, frame = cap.read()
    frames.append(frame)
cap.release()

######################################################################
# BATCH DETECTION

print("\nDetecting persons in video frames...")
results = []
for i in tqdm(range(0, len(frames), BATCH_SIZE), ascii=True):
    # create batch with RGB images
    imgs = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            for frame in frames[i:i+BATCH_SIZE]]
    # inference
    outputs = model(imgs, size=max(frame_width, frame_height))
    results += outputs.pred # outputs.pred is a list of Tensors,
                            # one for each image in imgs

######################################################################
# OUTPUT VIDEO

# instantiate output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(fpath_out, fourcc, fps, (frame_width, frame_height))

# draw boxes around detected persons (class=0)
print("\nDrawing boxes on video frames...")
t0 = time()
boxes_prev = [] # to keep track of boxes in previous frame
for frame, preds in zip(frames, results):
    # preds is a Tensor of size N x 6 where N is the no. of predictions
    boxes_current = [] # to save boxes of current predictions
    if preds is not None:
        for pred in preds:
            *box, conf, cl = pred.detach().cpu().numpy()
            box = tuple(map(int, box))
            if int(cl) == 0 and \
                (conf > CONF_THRESH or get_max_iou(box, boxes_prev) > IOU_THRESH):
                boxes_current.append(box)
                cv2.rectangle(frame, box[:2], box[2:], (0, 255, 0), 2)
                # cv2.putText(frame, '{:.2f}'.format(conf), (box[0]+5, box[-1]-5),
                #     cv2.FONT_HERSHEY_PLAIN, 1.25, (0, 255, 0), 2)
    boxes_prev = boxes_current
    out.write(frame)
print("Done ({:.2f}s)".format(time() - t0))
out.release()

######################################################################
# MERGE AUDIO AND VIDEO TRACKS

videoclip = moved.VideoFileClip(fpath_out)
audioclip = moved.AudioFileClip(fpath_in)

videoclip = videoclip.set_audio(audioclip)
videoclip.write_videofile(fpath_temp, fps=fps)
os.replace(fpath_temp, fpath_out)
