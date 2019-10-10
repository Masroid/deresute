import numpy as np
import cv2
import sys
import os
import glob
import argparse
import traceback
from tqdm import tqdm
import pdb
cur_dir = os.getcwd()

parser = argparse.ArgumentParser(description="recognize musics and create scores")
group1 = parser.add_mutually_exclusive_group()
group2 = parser.add_mutually_exclusive_group()
group1.add_argument("-d", "--add", action="store_true", help="add scores")
group1.add_argument("-w", "--write", action="store_true", help="overwrite scores")
group2.add_argument("-a", "--all", action="store_true", help="recognize all musics")
group2.add_argument("-s", "--select", nargs="*", help="recognize selected musics (you need to input music names with extension)")
args = parser.parse_args()

mdir = "music"
sdir = "score"

mdir_path = os.path.join(cur_dir, mdir)
if args.all:
    mnames = os.listdir(mdir_path)
elif args.select:
    mnames = args.select
else:
    raise RuntimeError("option or music names should be set")
    traceback.print_exc()

mpaths = []
spaths = []
sdir_path = os.path.join(cur_dir, sdir)
for mname in mnames:
    sname = mname.replace(".mp4", ".txt")
    spath = os.path.join(sdir_path, sname)
    if os.path.isfile(spath) and args.write:
        os.remove(spath)
    if (not os.path.isfile(spath)) or args.write:
        mpath = os.path.join(mdir_path, mname)
        mpaths.append(mpath)
        spaths.append(spath)

print("{} music data entry".format(len(mpaths)))

tdir = "template"
tdir_path = os.path.join(cur_dir, tdir)
tpaths = glob.glob(os.path.join(tdir_path,"*"))

icons = ["single", "long", "flick", "slide", "slidem", "none"]
slides = ["slide", "slidem"]
icon_avr1 = []
icon_avr2 = []
icon_max = []
icon_min = []

#get icon data
idir = "icondata"
ifile = "icondata.txt"
idir_path = os.path.join(cur_dir, idir)
idata_path = os.path.join(idir_path, ifile)
ipaths1 = []
ipaths2 = []
for icon in icons:
    iname = icon + ".txt"
    ipaths1.append(os.path.join(idir_path, *["icondata1", iname]))
    ipaths2.append(os.path.join(idir_path, *["icondata2", iname]))

itmp = []
with open(idata_path) as f:
    for s_line in f:
        itmp.append(s_line.split())

inumt = len(itmp)
idatat = np.zeros((inumt, 3), dtype = np.float64)
iconst = []

for i in range(inumt):
    iconst.append(itmp[i][0])
    idatat[i,0] = np.array(itmp[i][1]).astype(np.float64)
    idatat[i,1] = np.array(itmp[i][2]).astype(np.float64)
    idatat[i,2] = np.array(itmp[i][3]).astype(np.float64)

for ipath in ipaths1:
    itmp = []
    with open(ipath) as f:
        for s_line in f:
            itmp.append(s_line.split())

    icon_avr1.append(np.array(itmp[1]).astype(np.float64))

for ipath in ipaths2:
    itmp = []
    with open(ipath) as f:
        for s_line in f:
            itmp.append(s_line.split())

    icon_avr2.append(np.array(itmp[1]).astype(np.float64))

# set thresholds
th1 = 25.
th2 = 25.
ths1 = np.array([th1, th1, th1], dtype=np.float64)
ths2 = np.array([th2, th2, th2], dtype=np.float64)
ths = []

for idx, icon in enumerate(icons):
        if (icon == "slide") or (icon == "slidem"):
            ths.append(ths2)
        else:
            ths.append(ths1)




# set functions
def start_predict(count, i, mean):
    last_icons[i] = now_icons[i]
    th = 20.

    if np.sum(abs(icon_avr2[icons.index("none")] - np.array(mean)) < np.array([th, th, th])) == 3:
        now_icons[i] = "none"

    else:
        now_icons[i] = "null"

def icon_classify_a(count, i, mean):
    th = 10.
    ths = np.array([th, th, th], dtype=np.float64)
    for idx, icon in enumerate(slides):
        if (np.abs(mean - idatat[3 + idx]) < ths).all():
            now_icons[i] = icon
            sflag[i] = True
            break

def icon_classify_b(count, i, mean1, mean2, path):
    last_icons2[i] = last_icons[i]
    last_icons[i] = now_icons[i]
    sub = np.zeros(3, dtype=np.float64)

    sub = (last_means[i,:] - mean1)  / last_means[i]

    if np.sum(sub) < 0:

        for idx, icon in enumerate(icons):
            if sflag[i]:
                    sflag[i] = False
    #                now_icons[i] = last_icons[i]
    #                last_icons[i] = last_icons2[i]
                    break

            if ((np.abs(mean2 - icon_avr2[idx]) < ths[idx]).all() or (np.abs(mean1 - icon_avr1[idx]) < ths[idx]).all()):
                if ((icon != "slide") and (icon != "slidem")):
                    now_icons[i] = icon
                    break

            else:
                now_icons[i] = "null"

    #    if (not(last_icons[i] == "none" or last_icons[i] == "null")) and (now_icons[i] != last_icons[i]):
        if (now_icons[i] != "none") and (now_icons[i] != "null"):
            mdata = [count, i, last_icons[i]]
    #        print([count, i, last_icons[i]])
            with open(path, mode = "a") as f:
                f.write(" ".join(map(str,mdata)) + "\n")

def non_max_suppression_slow(w, h, overlapThresh, *boxes):
    if len(boxes) == 0:
        return []

    pick = []

    x1 = boxes[0]
    y1 = boxes[1]
    x2 = x1 + w
    y2 = y1 + h

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    if H < W:
        idxs = np.argsort(-x2)
    else:
        idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        suppress = [last]

        for pos in range(0, last):
            j = idxs[pos]

            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])

            ww = max(0, xx2 - xx1 + 1)
            hh = max(0, yy2 - yy1 + 1)

            overlap = float(ww * hh) / float(area[j])

            if overlap > overlapThresh:
                suppress.append(pos)

        idxs = np.delete(idxs, suppress)

    nb0 = []
    nb1 = []
    for k in pick:
        nb0.append(boxes[0][k])
        nb1.append(boxes[1][k])

    new_boxes = tuple([np.array(nb0), np.array(nb1)])
    return new_boxes



for mpath in mpaths:
    print("music name : {}\n".format(os.path.basename(mpath)))
    spath = spaths[mpaths.index(mpath)]

    # get the video's information
    cap = cv2.VideoCapture(mpath)
    W = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    W_stand = 1920
    H = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    H_stand = 886
    fps = cap.get(cv2.CAP_PROP_FPS)
    fnum = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # get the template image for template matching
    tmp = cv2.imread(tpaths[0])
    tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
    w, h = tmp.shape[::-1]
    overlapThresh = 0.3

    tmp2 = cv2.imread(tpaths[1])
    tmp2 = cv2.cvtColor(tmp2, cv2.COLOR_BGR2GRAY)
    w2, h2 = tmp2.shape[::-1]

    # set variables
    count = 0  # to record the frame number
    start = 0  # to decide the strat frame
    mdata = [] # to record the music score as a list of string
    flag = 0   #
    sflag = [False] * 5
    scount = [0] * 5
    nones = np.array(["none", "none", "none", "none", "none"], dtype = "<U6")
    last_icons2 = np.array(["null", "null", "null", "null", "null"], dtype = "<U6")
    last_icons = np.array(["null", "null", "null", "null", "null"], dtype = "<U6")
    now_icons = np.array(["null", "null", "null", "null", "null"], dtype = "<U6")
    last_means = np.zeros((5, 3), dtype=np.float64)

    with open(spath, mode = "a") as f:
        f.write(str(fps) + "\n")


    # resize the template size
    if H_stand != H:
        tmp = cv2.resize(tmp, dsize=None, fx=H/H_stand, fy=H/H_stand, interpolation=cv2.INTER_CUBIC)
        w, h = tmp.shape[::-1]

    for _ in tqdm(range(fnum)):
        ret, img = cap.read()

        if ret:
            if flag == 0:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                res =  cv2.matchTemplate(gray, tmp, cv2.TM_CCOEFF_NORMED)
                threshold = 0.89
                loc = np.where(res > threshold)
                boxes = non_max_suppression_slow(w, h, overlapThresh, *loc[::-1])


                for pt in zip(*boxes):
                    cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

                if len(boxes[0]) == 5 & len(boxes[1]) == 5:
                    flag = 1
                    bloc = boxes[0] + w - boxes[0][0]

            elif flag == 1:
                for i in range(5):
                    b0, b1 = boxes[0][i], boxes[1][i]
                    part = img[b1 + int(1/10*h) : b1 + int(9/10*h), b0 + int(1/10*w)  : b0 + int(9/10*w), :]
                    mean = np.mean(np.mean(part, axis = 0), axis = 0)
                    start_predict(count, i, mean)
                    if (now_icons == nones).all():
                        count = -36
                        flag = 2

            else:
                ba, br, bl = boxes[1][0], boxes[0][4], boxes[0][0]
                imgc = img[int(ba-h*20/100):int(ba+h*8/10), int(bl-w/2):int(br+w*3/2), :]
                Wc, Hc, Cc = imgc.shape
                gray2 = cv2.cvtColor(imgc, cv2.COLOR_BGR2GRAY)
                res2 =  cv2.matchTemplate(gray2, tmp2, cv2.TM_CCOEFF_NORMED)
                threshold2 = 0.60
                loc2 = np.where(res2 > threshold2)
                boxes2 = non_max_suppression_slow(w2, h2, overlapThresh, *loc2[::-1])

                for i in range(len(boxes2[0])):
                    b20, b21 = boxes2[0][i], boxes2[1][i]
                    part = imgc[b21: b21 + h2, b20 : b20 + w2, :]
                    mean = np.mean(np.mean(part, axis = 0), axis = 0)
                    idx = np.argmin(np.abs((b20+w2/2)-bloc))
                    icon_classify_a(count, idx, mean)

                for i in range(5):
                    b0, b1 = boxes[0][i], boxes[1][i]
                    part1 = img[b1 : b1 + h, b0 : b0 + w, :]
                    mean1 = np.mean(np.mean(part1, axis = 0), axis = 0)
                    part2 = img[b1 + int(1/10*h) : b1 + int(9/10*h), b0 + int(1/10*w)  : b0 + int(9/10*w), :]
                    mean2 = np.mean(np.mean(part2, axis = 0), axis = 0)
                    icon_classify_b(count, i, mean1, mean2, spath)

            count += 1
#            cv2.imshow('frame',img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            break

    cap.release()
    cv2.destroyAllWindows()
