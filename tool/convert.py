import cv2
import numpy as np
import os
import sys
from tqdm import tqdm
import argparse
import pdb
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
    mnames = args.select
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
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    fnum = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc('m','p','4', 'v')
    video  = cv2.VideoWriter(wpath, fourcc, fps, (960, 540))

    for _ in tqdm(range(fnum)):
        ret, img = cap.read()
        if ret:
            if W < H:
                cut_area = int((H - W * 16 / 9) / 2)
                X_start = 0
                Y_start = cut_area
                X_length = W
                Y_length = H - cut_area
                out = img[Y_start : Y_length, X_start : X_length, :]

                height = out.shape[0]
                width = out.shape[1]

                center = (int(width / 2), int(height / 2))
                angle = -90.0
                scale = 1.0
                trans = np.array([[0., -1., height], [1., 0., 0.]], dtype=np.float64)
                out = cv2.warpAffine(out, trans, (height, width))

            else:
                cut_area = int((W - H * 16 / 9) / 2)
                X_start = cut_area
                Y_start = 0
                X_length = W - cut_area
                Y_length = H
                out = img[Y_start : Y_length, X_start : X_length, :]

            out = cv2.resize(out, (960, 540))
#            cv2.imshow("frame", out)
            video.write(out)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            break

    cap.release()
    video.release()
    cv2.destroyAllWindows()
