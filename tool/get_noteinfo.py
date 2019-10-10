import numpy as np
import cv2
import sys
import os
import codecs
import argparse
import traceback
from tqdm import tqdm
os.chdir("..")
cur_dir = os.getcwd()

parser = argparse.ArgumentParser(description="recognize musics and create scores")
group1 = parser.add_mutually_exclusive_group()
group2 = parser.add_mutually_exclusive_group()
group1.add_argument("-d", "--add", action="store_true", help="add act data")
group1.add_argument("-w", "--write", action="store_true", help="overwrite act data")
group2.add_argument("-a", "--all", action="store_true", help="calculate all scores")
group2.add_argument("-s", "--select", nargs="*", help="calculate selected scores (you need to input score names with extension)")
args = parser.parse_args()

sdir = "score"
sdir_path = os.path.join(cur_dir, sdir)
ndir = "note_info"
ndir_path = os.path.join(cur_dir, ndir)

if args.all:
    snames = os.listdir(sdir_path)
elif args.select:
    snames = args.select
else:
    raise RuntimeError("option or score names should be set")
    traceback.print_exc()

spaths = []
npaths = []

for sname in snames:
    spath = os.path.join(sdir, sname)
    npath = os.path.join(ndir, sname)
    if os.path.isfile(npath) and args.write:
        os.remove(npath)
    if not os.path.isfile(npath):
        spaths.append(spath)
        npaths.append(npath)

print("{} scores entry".format(len(spaths)))

for idx, spath in enumerate(spaths):
    npath = npaths[idx]
    icons = ["single", "long", "flick", "slide", "slidem"]
    inums = np.zeros(len(icons), dtype=np.int32)
    mdata = []
    with open(spath) as f:
        for s_line in f:
            marg = s_line.rstrip("\n").split()
            if len(marg) == 3:
                for idx, icon in enumerate(icons):
                    if icon == marg[2]:
                        inums[idx] += 1
            mdata.append(marg)
    fps = float(mdata.pop(0)[0])
    combo = len(mdata)

    inums_ = np.zeros(len(icons)-1, dtype=np.int32)
    inums_[0:3] = inums[0:3]
    inums_[3] = inums[3] + inums[4]

    print("max combo is %4d" % (combo),file=codecs.open(npath, "a", "utf-8"))
    for i in range(len(inums_)):
        print("%6s : %4d" % (icons[i], inums_[i]),file=codecs.open(npath, "a", "utf-8"))
