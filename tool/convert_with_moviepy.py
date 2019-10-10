from moviepy.editor import*
import cv2
import numpy as np
import os
import sys
import argparse
cur_dir = os.getcwd()

parser = argparse.ArgumentParser(description="convert movies")
group1 = parser.add_mutually_exclusive_group()
group2 = parser.add_mutually_exclusive_group()
group1.add_argument("-d", "--add", action="store_true", help="add movies")
group1.add_argument("-w", "--write", action="store_true", help="overwrite movies")
group2.add_argument("-a", "--all", action="store_true", help="convert all movies")
group2.add_argument("-s", "--select", nargs="*", help="convert selected movies (you need to input movie names with extension)")
args = parser.parse_args()

rdir = "music_original"
wdir = "music"
rdir_path = os.path.join(cur_dir, rdir)
wdir_path = os.path.join(cur_dir, wdir)
if args.all:
    mnames = os.listdir(rdir_path)
elif args.select:
    mnames = args.selected
else:
    raise RuntimeError("option or movie names should be set")
    traceback.print_exc()

rpaths = []
wpaths = []

for mname in mnames:
    rpath = os.path.join(rdir_path, mname)
    wpath = os.path.join(wdir_path, mname)
    if os.path.isfile(wpath) and args.write:
        os.remove(wpath)
    if (not os.path.isfile(wpath)):
        rpaths.append(rpath)
        wpaths.append(wpath)

print("{} movies entry".format(len(rpaths)))

for idx, rpath in enumerate(rpaths):
    wpath = wpaths[idx]
    cap = cv2.VideoCapture(rpath)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_ = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()

    if W < H:
        W, H = H, W
        video = VideoFileClip(rpath).resize((W, H))
    cut_area = int((W - H * 16 / 9) / 2)
    X_start = cut_area
    Y_start = 0
    X_length = W - cut_area
    Y_length = H
    video = video.crop(x1=X_start,y1=Y_start,x2=X_length,y2=Y_length)

    save_path = wpath
    video.resize((960, 540)).write_videofile(wpath,fps=fps_)
