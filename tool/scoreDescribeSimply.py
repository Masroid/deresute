import numpy as np
import cv2
import sys
import os
import codecs
import argparse
import traceback
from tqdm import tqdm
cur_dir = os.getcwd()

parser = argparse.ArgumentParser(description="recognize musics and create scores")
group1 = parser.add_mutually_exclusive_group()
group2 = parser.add_mutually_exclusive_group()
group1.add_argument("-d", "--add", action="store_true", help="add scores")
group1.add_argument("-w", "--write", action="store_true", help="overwrite scores")
group2.add_argument("-a", "--all", action="store_true", help="recognize all musics")
group2.add_argument("-s", "--select", nargs="*", help="recognize selected musics (you need to input music names with extension)")
args = parser.parse_args()


sdir = "score"
sdir_path = os.path.join(cur_dir, sdir)
pdir = "simple_score"
pdir_path = os.path.join(cur_dir, pdir)

if args.all:
    snames = os.listdir(sdir_path)
elif args.select:
    snames = args.select
else:
    raise RuntimeError("option or score names should be set")
    traceback.print_exc()

spaths = []
ppaths = []

for sname in snames:
    spath = os.path.join(sdir, sname)
    ppath = os.path.join(pdir, sname)
    if os.path.isfile(ppath) and args.write:
        os.remove(ppath)
    if not os.path.isfile(ppath):
        spaths.append(spath)
        ppaths.append(ppath)

print("{} scores entry".format(len(spaths)))

def icon2simbol(icon):
    if icon == "single":
        return "g"
    elif icon == "long":
        return "l"
    elif icon == "flick":
        return "f"
    elif icon == "slide":
        return "s"
    elif icon == "slidem":
        return "m"

for idx, spath in enumerate(spaths):
    ppath = ppaths[idx]
    icons = ["single", "long", "flick", "slide", "slidem"]
    inums = np.zeros(len(icons), dtype=np.int32)
    mdata = []
    with open(spath) as f:
        for s_line in f:
            marg = s_line.rstrip("\n").split()
            if len(marg) == 3:
                marg[0] = int(marg[0])
                marg[1] = int(marg[1])
            mdata.append(marg)
    fps = float(mdata.pop(0)[0])
    combo = len(mdata)

    lf = mdata[combo-1][0]
    sdata = np.full((lf+1, 5), "-")
    frame = 0
    i = 0

    while i < combo:
        if mdata[i][0] == frame:
            sdata[frame, mdata[i][1]] = icon2simbol(mdata[i][2])
            i += 1
        else:
            frame += 1
        if frame > lf:
            break

    for i in range(lf, -1, -1):
        print("%4d %s%s%s%s%s"\
        % (i, sdata[i,0], sdata[i,1], sdata[i,2], sdata[i,3], sdata[i,4] ),\
        file=codecs.open(ppath, "a", "utf-8"))
