import numpy as np
import sys
import os
import codecs
import argparse
import traceback
from tqdm import tqdm
import pdb
cur_dir = os.getcwd()

parser = argparse.ArgumentParser(description="recognize musics and create scores")
group1 = parser.add_mutually_exclusive_group()
group2 = parser.add_mutually_exclusive_group()
group3 = parser.add_mutually_exclusive_group()
group1.add_argument("-d", "--add", action="store_true", help="add act data")
group1.add_argument("-w", "--write", action="store_true", help="overwrite act data")
group2.add_argument("-a", "--all", action="store_true", help="calculate all scores")
group2.add_argument("-s", "--select", nargs="*", help="calculate selected scores (you need to input score names with extension)")
parser.add_argument("--motif", nargs=1, help="input the motif magnification")
group3.add_argument("--focus", action="store", nargs="?",  const=True, default=False, help="set focus")
group3.add_argument("--synergy", action="store", nargs="?",  const=True, default=False, help="set synergy")
args = parser.parse_args()

sdir = "score"
sdir_path = os.path.join(cur_dir, sdir)
adir = "act"
adir_path = os.path.join(cur_dir, adir)

if args.all:
    snames = os.listdir(sdir_path)
elif args.select:
    snames = args.select
else:
    raise RuntimeError("option or score names should be set")
    traceback.print_exc()

spaths = []
apaths = []

fosec = 0
sysec = 0

for sname in snames:
    if args.focus:
        if args.focus == True:
            aname = sname.replace(".txt","_focus.txt")
        else:
            fosec = int(args.focus)
            aname = sname.replace(".txt","_focus" + str(fosec) + ".txt")

    elif args.synergy:
        if args.synergy == True:
            aname = sname.replace(".txt","_synergy.txt")
        else:
            sysec = int(args.synergy)
            aname = sname.replace(".txt","_synergy" + str(sysec) + ".txt")

    else:
        aname = sname
    spath = os.path.join(sdir, sname)
    apath = os.path.join(adir, aname)
    if os.path.isfile(apath) and args.write:
        os.remove(apath)
    if not os.path.isfile(apath):
        spaths.append(spath)
        apaths.append(apath)

print("{} scores entry".format(len(spaths)))

score_bonus = [
    ["sb", "h", 4, 3.0],  \
    ["ol", "m", 6, 4.5], ["sb", "m", 6, 4.5], \
    ["cc", "h", 7, 4.5], ["sb", "h", 7, 4.5], \
    ["ol", "m", 7, 6.0], ["la", "h", 7, 6.0], ["fa", "h", 7, 6.0], ["fa24", "h", 7, 6.0], \
    ["cc", "h", 9, 6.0], ["sb", "h", 9, 6.0], \
    ["ol", "m", 9, 7.5], ["la", "h", 9, 6.0], ["fa", "h", 9, 7.5], ["sb", "m", 9, 7.5], \
    ["cc", "h", 11, 7.5], ["sb", "h", 11, 7.5], \
    ["cc", "m", 11, 9.0], ["sa", "h",11, 9.0], ["sb", "m", 11, 9.0], \
    ["cc", "h", 13, 9.0], ["sb", "h",13, 9.0]
]

combo_bonus = [
    ["cb", "h", 4, 3.0], \
    ["cb", "m", 6, 4.5], \
    ["cb", "h", 7, 4.5], ["cb", "m", 7, 6.0], \
    ["cb", "h", 9, 6.0], ["cb", "m", 9, 7.5], \
    ["cb", "h", 11, 7.5], ["cb", "m", 11, 9.0], \
    ["cb", "h", 13, 9.0]
]

perfect_support = [
    ["ps", "h", 9, 4.5], \
    ["ps", "h", 12, 6.0], \
    ["ps", "h", 15, 7.5],
]

recovery = [
    ["ar", "h", 5, 3.0], \
    ["lr", "h", 8, 4.5], ["ar", "h", 8, 4.5], \
    ["lr", "m", 9, 6.0], ["ar", "m", 9, 6.0], \
    ["lr", "h", 10, 6.0], \
    ["lr", "h", 11, 6.0], ["ar", "h", 11, 6.0], \
    ["lr", "h", 13, 7.5], ["ar", "m", 13, 9.0]
]

skill_boost = [
    ["bo", "h", 7, 6.0], ["es", "h", 7, 4.5], \
    ["bo", "h", 8, 7.5], \
    ["bo", "h", 10, 9.0],
]

life_sparkle = [
    ["ls", "h", 7, 4.5], \
    ["ls", "h", 9, 6.0], \
    ["ls", "h", 11, 7.5], \
    ["ls", "h", 13, 9.0]
]

skills = score_bonus

if args.focus:
    itrs = [["fo", "m", 6, 4.5], ["fo", "h", 7, 4.5], ["fo", "h", 9, 6.0], ["fo", "h", 11, 7.5]]
    fosy = True
    if fosec != 0:
        for itr in itrs:
            if itr[2] == fosec:
                itrs = [itr]
                break

elif args.synergy:
    itrs = [["sy", "h", 7, 4.5], ["sy", "h", 9, 6.0], ["sy", "h", 11, 7.5]]
    fosy = True
    if sysec != 0:
        for itr in itrs:
            if itr[2] == sysec:
                itrs = [itr]
                break
else:
    itrs = [["--", "h", 1, 0.]]
    fosy = False

if args.motif:
    skills.append(["mt", "h", 7, 4.5])
    motif = float(args.motif[0])

def prob_to_num(symbol):
    if symbol == "h":
        return 1.
    elif symbol == "m":
        return 0.9425

def calcMag(mag1, mag2, prob1, prob2):
    mag = max(mag1, mag2) * prob1 * prob2 \
        + mag1 * prob1 * (1. - prob2) \
        + mag2 * (1. - prob1) * prob2 \
        + 1. * (1. - prob1) * (1. - prob2)
    return mag

def skill_to_mag(skill, icon):
    if skill == "ol":
        return 1.18
    elif skill == "cc":
        return 1.22
    elif skill == "la":
        if icon == "long":
            return 1.30
        else:
            return 1.10
    elif skill == "fa":
        if icon == "flick":
            return 1.30
        else:
            return 1.10
    elif skill == "fa24":
        if icon == "flick":
            return 1.24
        else:
            return 1.08
    elif skill == "sa":
        if icon == "slide" or icon == "slidem":
            return 1.40
        else:
            return 1.10
    elif skill == "sb":
        return 1.17
    elif skill == "mt" and args.motif:
        return (1. + motif / 100.)
    elif skill == "fo":
        return 1.16
    elif skill == "sy":
        return 1.16
    elif skill == "--":
        return 1.
    else:
        raise RuntimeError("skill is not defined")
        traceback.print_exc()

def symbol_to_skill(symbol):
    if symbol == "ol":
        return "オーバーロード"
    elif symbol == "cc":
        return "コンセントレーション"
    elif symbol == "la":
        return "30%ロングアクト"
    elif symbol == "fa":
        return "30%フリックアクト"
    elif symbol == "fa24":
        return "24%フリックアクト"
    elif symbol == "sa":
        return "40%スライドアクト"
    elif symbol == "sb":
        return "スコアボーナス"
    elif symbol == "mt":
        return "モチーフ（" + str(int(motif)) + "%）"
    elif symbol == "fo":
        return "フォーカス"
    elif symbol == "sy":
        return "シナジー"
    else:
        raise RuntimeError("skill is not defined")
        traceback.print_exc()

def symbol_to_prob(symbol):
    if symbol == "h":
        return "高確率"
    elif symbol == "m":
        return "中核率"
    else:
        raise RuntimeError("prob is not defined")
        traceback.print_exc()




for path_idx, spath in enumerate(spaths):
    apath = apaths[path_idx]
    mdata = []
    with open(spath) as f:
        for s_line in f:
            marg = s_line.rstrip("\n").split()
            if len(marg) == 3:
                marg[0] = float(marg[0])
                marg[1] = int(marg[1])
            mdata.append(marg)
    fps = float(mdata.pop(0)[0])
    combo = len(mdata)

    scores = np.zeros((len(skills), 2), dtype=np.float64)

    for itd in range(len(itrs)):

        tmpscores = np.zeros(len(skills), dtype=np.float64)
        tmpflags = np.full(len(skills), False)
        tmpfosyf = np.full(len(itrs), False)

        idata = np.zeros(4, dtype=np.int)
        lc, nc = 0, 0
        count = 0
        idx = 0
        step = 0
        flag = [False] * 14

        fin_count = mdata[-1][0] - 3.0 * fps

        steps = []
        steps.append(np.floor(combo * 0.05))
        steps.append(np.floor(combo * 0.10))
        steps.append(np.floor(combo * 0.25))
        steps.append(np.floor(combo * 0.50))
        steps.append(np.floor(combo * 0.70))
        steps.append(np.floor(combo * 0.80))
        steps.append(np.floor(combo * 0.90))
        steps.append(np.floor(combo * 1.00))

        cr = [1., 1.1, 1.2, 1.3, 1.4, 1.5, 1.7, 2.0]

        while count <= mdata[-1][0] + 1:
            now_sec = count / fps

            if idx >= steps[step]:
                step += 1

            if count == mdata[idx][0]:
                for si, skill in enumerate(skills):
#                    pdb.set_trace()
                    if ((now_sec > skill[2]) and (now_sec % skill[2] < skill[3]) and count < fin_count) \
                    or (count >= fin_count and tmpflags[si]):
                        tmpflags[si] = True
                        mag1 = skill_to_mag(skill[0], mdata[idx][2])
                        prob1 = prob_to_num(skill[1])
                        if ((now_sec > itrs[itd][2]) and (now_sec % itrs[itd][2] < itrs[itd][3]) and count < fin_count) \
                        or (count >= fin_count and tmpfosyf[itd]):
                            tmpfosyf[itd] = True
                            mag2 = skill_to_mag(itrs[itd][0], mdata[idx][2])
                            prob2 = prob_to_num(itrs[itd][1])
                            mag = calcMag(mag1, mag2, prob1, prob2)
                        else:
                            tmpfosyf[itd] = False
                            mag = mag1 * prob1 + 1. * (1. - prob1)
                        tmpscores[si] += mag * cr[step]

                    else:
                        tmpflags[si] = False
                        if ((now_sec > itrs[itd][2]) and (now_sec % itrs[itd][2] < itrs[itd][3]) and count < fin_count) \
                        or (count >= fin_count and tmpfosyf[itd]):
                            tmpfosyf[itd] = True
                            mag2 = skill_to_mag(itrs[itd][0], mdata[idx][2])
                            prob2 = prob_to_num(itrs[itd][1])
                            mag = mag2 * prob2 + 1. * (1. - prob2)
                            tmpscores[si] += mag * cr[step]
                        else:
                            tmpfosyf[itd] = False
                            tmpscores[si] += cr[step]

                idx += 1
            if idx == combo:
#                pdb.set_trace()
                break
            if not (count == mdata[idx][0]):
                count += 1

        for si, skill in enumerate(skills):
            if tmpscores[si] > scores[si][0]:
                scores[si][0] = tmpscores[si]
                scores[si][1] = itd

    itds = np.zeros(len(skills), dtype=np.int32)
    itds[:] = scores[:,1].astype(np.int32)
    sorted = np.argsort(-scores[:,0])
    if fosy:
        for i in range(len(skills)):
            si = sorted[i]
            print("%2d\t%2d秒 %s %1.1f秒発動 %10s，\tスコア:%4.2f (%2d秒 %s %1.1f %5s)"\
            % (i+1, skills[si][2], symbol_to_prob(skills[si][1]), skills[si][3], symbol_to_skill(skills[si][0]), scores[si, 0], \
            itrs[itds[si]][2], symbol_to_prob(itrs[itds[si]][1]),  itrs[itds[si]][3], symbol_to_skill(itrs[itds[si]][0])), \
            file=codecs.open(apath, "a", "utf-8"))
    else:
        for i in range(len(skills)):
            si = sorted[i]
            print("%2d\t%2d秒 %s %1.1f秒発動 %10s，\tスコア:%4.2f"\
            % (i+1, skills[si][2], symbol_to_prob(skills[si][1]), skills[si][3], symbol_to_skill(skills[si][0]), scores[si, 0]), \
            file=codecs.open(apath, "a", "utf-8"))
