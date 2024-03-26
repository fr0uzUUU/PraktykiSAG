import xml.etree.ElementTree as ET
import h5py
import argparse
import numpy as np
import cv2
import datetime
import os


parser = argparse.ArgumentParser(description='XmlToHdf5')
parser.add_argument('--annotations_name', type=str,
                    help='Name of the annotations.xml file')
parser.add_argument('--video_path', type=str,
                    help='Path to the .mp4 file')
args = parser.parse_args()
tree = ET.parse(args.annotations_name)
root = tree.getroot()
meta = root.find('meta')
source = meta.find('source')
f = h5py.File("DPT_" + source.text + ".hdf5", "w")
video_capture = cv2.VideoCapture(os.path.join(args.video_path, source.text))
task = meta.find('task')
owner = task.find('owner')
username = owner.find('username')
name = f.create_group(source.text)

frame_number_from_capture = 0

for image_node in root.iter('image'):
    points = image_node.find('points')
    ret, image_frame = video_capture.read()

    if ret is False:
        print("Tu mnie wywalilo")
        break

    xml_frame_number = image_node.attrib['name']
    xml_frame_number_splitted = xml_frame_number.split("_")

    frame_number_after_split = None
    if xml_frame_number_splitted[1] == "000000":
        frame_number_after_split = 0
    else:
        frame_number_after_split = int(xml_frame_number_splitted[1].lstrip("0"))

    if frame_number_after_split != frame_number_from_capture:
        print("frame_number not correct")
        break
    frame_number_before_incrementation = frame_number_from_capture
    frame_number_from_capture = frame_number_from_capture + 1

    if points is None:
        continue
    else:
        print(frame_number_after_split, frame_number_before_incrementation)

    frame = name.create_group(image_node.attrib['id'])
    frame.create_dataset('frame2', data=image_frame, compression="gzip", compression_opts=1)

    frame.attrs['frame_width'] = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame.attrs['frame_height'] = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame.attrs['Author'] = username.text
    frame.attrs['Date'] = str(datetime.datetime.now())

    feet_size = 0
    feet = None
    ball_size = 0
    ball = None
    match_ball_size = 0
    match_ball = None

    for points in image_node.iter('points'):
        points_value = points.attrib['points']
        x, y = points_value.split(",")
        if points.attrib['label'] == 'FOOT':
            if feet_size == 0:
                feet = frame.create_dataset('FEET', (1, 2), maxshape=(None, 2), dtype="i")
                feet[feet_size] = int(float(x)), int(float(y))
                feet_size = feet_size + 1
            else:
                feet.resize((feet_size + 1, 2))
                feet[feet_size] = int(float(x)), int(float(y))
                feet_size = feet_size + 1
        elif points.attrib['label'] == 'BALL':
            if ball_size == 0:
                ball = frame.create_dataset('BALL', (1, 2), maxshape=(None, 2), dtype="i")
                ball[ball_size] = int(float(x)), int(float(y))
                ball_size = ball_size + 1
            else:
                ball.resize((ball_size + 1, 2))
                ball[ball_size] = int(float(x)), int(float(y))
                ball_size = ball_size + 1
        elif points.attrib['label'] == 'MATCH_BALL':
            if match_ball_size == 0:
                match_ball = frame.create_dataset('MATCH_BALL', (1, 2), maxshape=(None, 2), dtype="i")
                match_ball[match_ball_size] = int(float(x)), int(float(y))
                match_ball_size = match_ball_size + 1
            else:
                match_ball.resize((match_ball_size + 1, 2))
                match_ball[match_ball_size] = int(float(x)), int(float(y))
                match_ball_size = match_ball_size + 1

f.close()

