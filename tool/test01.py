import numpy as np
import cv2
import sys
import os
import argparse
import traceback
from tqdm import tqdm
cur_dir = os.getcwd()

"""
parser = argparse.ArgumentParser(description="recognize musics and create scores")
group = parser.add_mutually_exclusive_group()
group.add_argument("-aa", "--allAdd", action="store_true", help="recognize all musics that have not been done and add those scores")
group.add_argument("-aw", "--allWrite", action="store_true", help="recognize all musics and overwrite all scores")
group.add_argument("-sa", "--selectAdd", nargs="*", help="recognize selected musics and add those score")
group.add_argument("-sw", "--selectWrite", nargs="*", help="recognize selected musics and overwrite those score")
args = parser.parse_args()
print(args)
"""

parser = argparse.ArgumentParser(description="recognize musics and create scores")
group1 = parser.add_mutually_exclusive_group()
group2 = parser.add_mutually_exclusive_group()
group1.add_argument("-d", "--add", action="store_true", help="add scores")
group1.add_argument("-w", "--write", action="store_true", help="overwrite scores")
group2.add_argument("-a", "--all", action="store_true", help="recognize all musics")
group2.add_argument("-s", "--select", nargs="*", help="recognize selected musics (you need to input music names with extension)")
args = parser.parse_args()
#parser.add_argument("music_names", nargs="*", help="")
#args = parser.parse_args()

print(args)

mdir = "music"
sdir = "score"

"""
mdir_path = os.path.join(cur_dir, mdir)
if args.allAdd or args.allWrite:
    mnames = os.listdir(mdir_path)
elif args.selectAdd:
    mnames = args.selectAdd
elif args.selectWrite:
    mnames = args.selectWrite
else:
    raise RuntimeError("set option or music names")
    traceback.print_exc()

print(mnames)
"""
