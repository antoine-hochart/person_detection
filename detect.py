import argparse
import os
import sys
import glob
import pytube
import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import moviepy.editor as moved

from time import time
from tqdm import tqdm

######################################################################
# ARGUMENT PARSER

parser = argparse.ArgumentParser(
    description="Detects persons in a Youtube video.")
parser.add_argument('--url', type=str, default='https://youtu.be/h4s0llOpKrU')
parser.add_argument('--with_gpu', action='store_true',
        help="Whether to use the GPU if available.")
parser.add_argument('--stride', type=int, default=1,
        help="Frame stride for the detection model.")
parser.add_argument('--batch_size', type=int, default=1,
        help="Number of frames that the detection model process simultaneously.")
parser.add_argument('--conf0', type=float, default=0.30,
        help="Minimum confidence score for detections.")
parser.add_argument('--min_conf', type=float, default=0.70,
        help="Threshold for the maximum confidence level of an object across two frames.")
parser.add_argument('--min_iou', type=float, default=0.66,
        help="Threshold for the IoU of bounding boxes of the same object across two frames.")
parser.add_argument('--model', type=str, default='retinanet',
        help="Model for object detection: fasterrcnn or retinanet (default).")
parser.add_argument('--with_conf', action='store_true',
        help="Whether to display confidence scores.")
args = parser.parse_args()

######################################################################
# GLOBAL PARAMETERS

VIDEO_DIR = os.path.dirname(__file__)

URL = args.url
WITH_GPU = args.with_gpu
STRIDE = args.stride
BATCH_SIZE = args.batch_size
CONF0 = args.conf0
MIN_CONF = args.min_conf
MIN_IOU = args.min_iou
MODEL = args.model
WITH_CONF = args.with_conf

######################################################################
# UTILITY FUNCTIONS

def image2tensor(frame):
    """ Convert OpenCV image to Pytorch tensor. """
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
        ])
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = transform(img)
    return img


def get_area(box):
    """ Compute area of box,
        assuming that box is non-empty if a < c and b < d
        where box = (a, b, c, d).
    """
    return max(box[2] - box[0], 0) * max(box[3] - box[1], 0)


def get_iou(box1, box2):
    """ Compute IoU score between box1 and box2. """
    xa = max(box1[0], box2[0])
    ya = max(box1[1], box2[1])
    xb = min(box1[2], box2[2])
    yb = min(box1[3], box2[3])
    intersection = get_area((xa, ya, xb, yb))
    union = get_area(box1) + get_area(box2) - intersection
    return intersection / union


def get_iou0(box1, box2):
    """ Compute IoU score between box1 and box2
        when boxes are TRANSLATED to the origin.
    """
    w = np.minimum(box1[2] - box1[0] + 1, box2[2] - box2[0] + 1)
    h = np.minimum(box1[3] - box1[1] + 1, box2[3] - box2[1] + 1)
    intersection = w * h
    union =  get_area(box2) + get_area(box1) - intersection
    return intersection / union

######################################################################

start = time()

######################################################################
# DOWNLOAD YOUTUBE VIDEO

# instantiate YouTube object
yt = pytube.YouTube(URL)

# set video name
video_title = '_'.join(word for word in yt.title.split() if word.isalpha())

# load Youtube video
try:
    # check if video already downloaded
    path_video = glob.glob(os.path.join(VIDEO_DIR, video_title+'.*'))[0]
    _, extension = os.path.splitext(path_video)
    print("Video {}{} already downloaded".format(video_title, extension))
except IndexError:
    # if not, download video
    try:
        print("Downloading Youtube video {}...".format(yt.title))
        os.makedirs(os.path.abspath(VIDEO_DIR), exist_ok=True)
        path_video = yt.streams.first().download(VIDEO_DIR)
        _, extension = os.path.splitext(path_video)
        print("Done")
    except AttributeError:
        print("Impossible to download Youtube video with pytube.")
        sys.exit()

# set input/output video file path
path_video_in = os.path.join(VIDEO_DIR, video_title+extension)
path_video_out = os.path.join(VIDEO_DIR, video_title+'_detect'+extension)
path_video_temp = os.path.join(VIDEO_DIR, video_title+'_temp'+extension)
os.rename(path_video, path_video_in)

######################################################################
# LOAD DETECTION AND FEATURE EXTRACTION MODELS

device = torch.device("cuda" if torch.cuda.is_available() and WITH_GPU else "cpu") 

print("\nLoading {} model on {}...".format(MODEL.upper(), device))
t0 = time()

# load detection model
if MODEL == 'retinanet':
    model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True)
elif MODEL == 'fasterrcnn':
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
else:
    raise ValueError("Model not recognized. "\
        "Pretrained models only exist for 'fasterrcnn' and 'retinanet'.")

# freeze model
for param in model.parameters():
    param.requires_grad = False

# send model to device and set eval mode
model.to(device)
model.eval()

print("Done ({:.2f}s)".format(time() - t0))

######################################################################
# EXTRACT VIDEO FRAMES

print("\nExtracting video frames...")
t0 = time()

# set video source
cap = cv2.VideoCapture(path_video_in)

# get video metrics
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("> {} frames".format(n_frames))
print("> {} fps".format(fps))
print("> W x H = {} x {}".format(frame_width, frame_height))

# extract video frames
frames = []
for _ in range(n_frames):
    ret, frame = cap.read()
    frames.append(frame)
cap.release()

print("Done ({:.2f}s)".format(time() - t0))

######################################################################
# DETECTION

print("\nDetecting persons on video frames...")
t0 = time()

# indices of frames to make detection on
frame_indices = list(range(0, n_frames, STRIDE))
if n_frames % STRIDE != 1:
    # add index of last frame
    frame_indices.append(n_frames-1)

# batch detection
results = []
for i in tqdm(range(0, len(frame_indices), BATCH_SIZE), ascii=True):
    # create batch with list of tensors
    imgs = [image2tensor(frames[idx]) for idx in frame_indices[i:i+BATCH_SIZE]]
    imgs = torch.stack(imgs).to(device)
    # inference
    with torch.no_grad():
        outputs = model(imgs)
    results += outputs

timedelta = time() - t0
print("Done ({:.0f}m {:.0f}s)".format(timedelta//60, timedelta%60))

######################################################################
# FILTER PREDICTIONS WITH LOW CONF SCORE

print("\nRemoving predictions with a confidence score lower than {}...".format(conf0))
t0 = time()

output_boxes = []
output_scores = []
for outputs in results:
    mask = (outputs['labels'] == 1) * (outputs['scores'] > CONF0)
    # boxes
    boxes = outputs['boxes'][mask]
    boxes = boxes.detach().cpu().numpy()
    boxes = boxes.astype(int)
    output_boxes.append(boxes)
    # confidence scores
    scores = outputs['scores'][mask]
    scores = scores.detach().cpu().numpy()
    output_scores.append(scores)

print("Done ({:.2f}s)".format(time() - t0))

######################################################################
# INTERPOLATE BOXES

print("\nInterpolating boxes between frames...")
t0 = time()

final_boxes = [[] for _ in range(n_frames)]
final_scores = [[] for _ in range(n_frames)]

for i in range(1, len(output_boxes)):
    boxes0 = output_boxes[i-1] # boxes in previous frame
    scores0 = output_scores[i-1] # confidence scores in previous frame
    k = frame_indices[i-1] # index of previous frame
    boxes1 = output_boxes[i] # boxes in current frame
    scores1 = output_scores[i] # confidence scores in current frame
    l = frame_indices[i] # index of current frame
    for box1, score1 in zip(boxes1, scores1):
        best_iou = 0
        best_score = 0
        interpolated_boxes = []
        idx0 = None
        for j, (box0, score0) in enumerate(zip(boxes0, scores0)):
            # check that box0 and box1 have a similar shape
            if get_iou0(box0, box1) < MIN_IOU:
                continue
            # interpolate boxes between box0 and box1
            boxes = [
                ((1 - alpha) * box0 + alpha * box1).astype(int)
                for alpha in np.linspace(0, 1, l-k+1)
                ]
            # compute IoU between successive interpolated boxes
            iou = min([get_iou(b1, b2) for b1, b2 in zip(boxes[:-1], boxes[1:])])
            if iou> best_iou:
                best_iou= iou
                best_score = max(score0, score1)
                interpolated_boxes = boxes
                idx0 = j
        if best_iou > MIN_IOU and best_score > MIN_CONF:
            boxes0 = np.delete(boxes0, idx0, 0)
            if tuple(interpolated_boxes[0]) not in final_boxes[k]:
                final_boxes[k].append(tuple(interpolated_boxes[0]))
                final_scores[k].append(best_score)
            for j, box in enumerate(interpolated_boxes[1:]):
                final_boxes[k+j+1].append(tuple(box))
                final_scores[k+j+1].append(best_score)

print("Done ({:.2f}s)".format(time() - t0))

######################################################################
# OUTPUT VIDEO

# instantiate output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(path_video_out, fourcc, fps, (frame_width, frame_height))

# draw boxes around detected persons (class=0)
print("\nDrawing boxes on video frames...")
t0 = time()
for frame, boxes, scores in zip(frames, final_boxes, final_scores):
    for box, score in zip(boxes, scores):
        cv2.rectangle(frame, box[:2], box[2:], (0, 255, 0), 2)
        if WITH_CONF:
            cv2.putText(frame, '{:.2f}'.format(score), (box[0]+5, box[-1]-5),
                cv2.FONT_HERSHEY_PLAIN, 1.25, (0, 255, 0), 2)
    out.write(frame)
print("Done ({:.2f}s)".format(time() - t0))
out.release()

######################################################################
# MERGE AUDIO AND VIDEO TRACKS

print("\nMerging audio and video...")
t0 = time()
videoclip = moved.VideoFileClip(path_video_out)
audioclip = moved.AudioFileClip(path_video_in)

videoclip = videoclip.set_audio(audioclip)
videoclip.write_videofile(path_video_temp, fps=fps)
os.replace(path_video_temp, path_video_out)
print("Done ({:.2f}s)".format(time() - t0))

######################################################################

totaltime = time() - start
print("\nTotal time: {:.0f}m {:.0f}s".format(totaltime//60, totaltime%60))
