#-*- coding: utf-8 -*-
# Run in server

import os.path
import numpy
import cv2
import datetime

def MainPath():
    main_path = os.path.join(os.path.expanduser('~'), 'my_project')
    return main_path

def VideoPath():
    video_path = os.path.join(MainPath(),'Videos', "my_video.mp4")
    return video_path

# Video info
# H264 MPEG-4 AVC (part 10) (avc1)
# Resolution 1280 x 720
# Frame rate 29.97 frames/second
# 
# Sound ON
# MPEG AAC Audio mp4a
# Channels Stereo (2)
# Sample rate 48000 Hz
# 32 Bits per sample

def frame_to_timestamp(framenum):
    seconds = framenum//30
    minutes = seconds//60
    hours = minutes//60
    minutes = minutes%60
    seconds = seconds%60
    frame = framenum%30
    timestamp = '{0:02d}.{1:02d}.{2:02d}.{3:02d}'.format(hours,minutes,seconds,frame)
    return timestamp

def timestamp_to_class(timestamp):
    hours,minutes,seconds,frame = [int(i) for i in timestamp.split('.')]
    first_classification = 0
    second_classification = 0
    timeval = datetime.time(hours, minutes, seconds)
    # as a sample, I used these times 00:00:00 ~ 00:01:00, 00:01:00 ~ 00:02:00, 00:02:00 ~ end
    # write here your tags with times
    if datetime.time(0,0,0)<=timeval<=datetime.time(0,1,0): first_classification, second_classification = (1,1)
    elif datetime.time(0,1,0)<timeval<=datetime.time(0,2,0): first_classification, second_classification = (2,1)
    elif datetime.time(0,2,0)<timeval: first_classification, group = (0,0)
    return first_classification, second_classification

def video_capture(mode='grayscale_resize_1to10'):
    # for video analysis, smaller pictures are faster and produce better results,
    # but in case you want larger pictures I made several ifs.
    vidcap = cv2.VideoCapture(VideoPath())
    success,image = vidcap.read()
    success_message = 'Success' if success else 'Failed'
    framenum = 0
    timestamp = frame_to_timestamp(framenum)
    print('Read frame {} : {}'.format(timestamp,success_message))
    while success:
        first_classification, second_classification = timestamp_to_class(timestamp)
        if mode == 'raw':
            capture = "frame_{}_{}_firstclass_{}_secondclass_{}.jpg".format(str(framenum).zfill(6), timestamp, first_classification, second_classification)
            capture_filename = os.path.join(MainPath(), 'Videos', 'frames', 'raw_frames', capture)
            cv2.imwrite(capture_filename, image)  # save frame as JPEG file
            print('Write: Success') if os.path.exists(capture_filename) else print('Write: Failed')
        elif mode == 'grayscale':
            capture = "grayscale_frame_{}_{}_firstclass_{}_secondclass_{}.jpg".format(str(framenum).zfill(6), timestamp, first_classification, second_classification)
            capture_filename = os.path.join(MainPath(), 'Videos', 'frames', 'grayscale_frames', capture)
            processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(capture_filename, processed_image)  # save frame as JPEG file
            print('Write: Success') if os.path.exists(capture_filename) else print('Write: Failed')
        elif mode == 'grayscale_resize_1to10':
            capture = "grayscale_rezised_1to10_frame_{}_{}_firstclass_{}_secondclass_{}.jpg".format(str(framenum).zfill(6), timestamp, first_classification, second_classification)
            capture_filename = os.path.join(MainPath(), 'Videos', 'frames', 'grayscale_resize_1to10_frames', capture)
            processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            resized_image = cv2.resize(processed_image, (0,0), fx=0.1, fy=0.1)
            cv2.imwrite(capture_filename, resized_image)  # save frame as JPEG file
            print('Write: Success') if os.path.exists(capture_filename) else print('Write: Failed')
        elif mode == 'grayscale_resize_1to5':
            capture = "grayscale_rezised_1to5_frame_{}_{}_firstclass_{}_secondclass_{}.jpg".format(str(framenum).zfill(6), timestamp, first_classification, second_classification)
            capture_filename = os.path.join(MainPath(), 'Videos', 'frames', 'grayscale_resize_1to5_frames', capture)
            processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            resized_image = cv2.resize(processed_image, (0,0), fx=0.2, fy=0.2)
            cv2.imwrite(capture_filename, resized_image)  # save frame as JPEG file
            print('Write: Success') if os.path.exists(capture_filename) else print('Write: Failed')
        elif mode == 'grayscale_resize_1to2':
            capture = "grayscale_rezised_1to2_frame_{}_{}_firstclass_{}_secondclass_{}.jpg".format(str(framenum).zfill(6), timestamp, first_classification, second_classification)
            capture_filename = os.path.join(MainPath(), 'Videos', 'frames', 'grayscale_resize_1to2_frames', capture)
            processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            resized_image = cv2.resize(processed_image, (0,0), fx=0.5, fy=0.5)
            cv2.imwrite(capture_filename, resized_image)  # save frame as JPEG file
            print('Write: Success') if os.path.exists(capture_filename) else print('Write: Failed')
        ###
        success,image = vidcap.read()
        success_message = 'Success' if success else 'Failed'
        framenum += 1
        timestamp = frame_to_timestamp(framenum)
        print('Read frame {} : {}'.format(timestamp,success_message))

def main():
    # print('Begin Video Capture, save to frames.jpg: mode = raw')
    # video_capture(mode='raw')
    # print('Begin Video Capture, save to frames.jpg: mode = grayscale')
    # video_capture(mode='grayscale')
    print('Begin Video Capture, save to frames.jpg: mode = grayscale_rezise_1:10')
    video_capture(mode='grayscale_resize_1to10')
    # print('Begin Video Capture, save to frames.jpg: mode = grayscale_rezise_1:5')
    # video_capture(mode='grayscale_resize_1to5')
    # print('Begin Video Capture, save to frames.jpg: mode = grayscale_rezise_1:2')
    # video_capture(mode='grayscale_resize_1to2')
    print('Finished!')

if __name__ == '__main__':
    main()