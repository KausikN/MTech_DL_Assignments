'''
Library for basic video functions
'''

# Imports
import os
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Main Functions
def GetFramesFromVideo(path=None, max_frames=-1):
    '''
    Get frames from video
    '''
    vid = cv2.VideoCapture(path)
    
    frames = []
    frameCount = 0
    # Check if vid opened
    if (vid.isOpened()== False): 
        print("Error opening video stream or file")
    # Read until video is completed
    while(vid.isOpened() and ((not (frameCount == max_frames)) or (max_frames == -1))):
        # Capture frame
        ret, frame = vid.read()
        if ret == True:
            frames.append(frame)
            frameCount += 1
        else: 
            break
    # Release vid object
    vid.release()

    return frames

def VideoEffect(pathIn, pathOut, EffectFunc, max_frames=-1, speedUp=1, fps=20.0, size=None):
    '''
    Apply effect to video
    '''
    # Get frames and apply the effect
    frames = GetFramesFromVideo(path=pathIn, max_frames=max_frames)
    frames = frames[::int(speedUp)]
    print("Video Frames:", len(frames))
    frames_effect = []
    for frame in tqdm(frames):
        frame = EffectFunc(frame)
        frames_effect.append(frame)
        
    # Save
    if os.path.splitext(pathOut)[-1] == ".gif":
        frames_effect_image = []
        for frame in frames_effect: frames_effect_image.append(Image.fromarray(frame))
        frames_effect = frames_effect_image
        extraFrames = []
        if len(frames_effect) > 1:
            extraFrames = frames_effect[1:]
        frames_effect[0].save(pathOut, save_all=True, append_images=extraFrames, format="GIF", loop=0)
    else:
        if size is None: size = (frames[0].shape[1], frames[0].shape[0])
        out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        for frame in frames_effect:
            out.write(frame)
        out.release()

# Run