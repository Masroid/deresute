import numpy as np
import pdb

th = 6000000
fan1 = np.zeros(1, dtype="i8")
fan2 = np.zeros(1, dtype="i8")
fan3 = np.zeros(1, dtype="i8")
spec1, spec2, spec3 = 0, 0, 0
stamina = 0
smag = 1.1
cmag = 1.1
score = 1000.
n = 2

while fan1 < th:
    if spec1 <= 741000:
        mag = 1.3 + np.ceil(spec1 / 7410.) / 1000.
        fan1[0] += (3 * 5 * np.ceil(score * cmag * mag * n)).astype(np.int64)
        spec1 += 15571 * smag + 671
    elif spec1 <= 2031000:
        mag = 1.4 + np.ceil((spec1 - 741000.) / 2150.) / 1000.
        fan1[0] += (3 * 5 * np.ceil(score * cmag * mag * n)).astype(np.int64)
        spec1 += 15571 + 671
    elif spec1 <= 3300000:
        mag = 2. + np.ceil((spec1 - 2031000) / ((spec1 - 2031000) / (3300000 - 2031000) * 550 + 1995)) / 1000
        fan1[0] += (3 * 5 * np.ceil(score * cmag * mag * n)).astype(np.int64)
        spec1 += 15571 + 671
    else:
        mag = 2.5
        fan1[0] += (3 * 5 * np.ceil(score * cmag * mag * n)).astype(np.int64)
        spec1 += 15571 + 671

    stamina += 40

stamina1 = stamina
stamina = 0

while fan2 < th:
    if spec2 <= 741000:
        mag = 1.3 + np.ceil(spec2 / 7410.) / 1000.
        fan2[0] += 3 * 5 * np.ceil(score * cmag * mag * n)
        spec2 += 19228 * smag + 828
    elif spec2 <= 2031000:
        mag = 1.4 + np.ceil((spec2 - 741000.) / 2150.) / 1000.
        fan2[0] += 3 * 5 * np.ceil(score * cmag * mag * n)
        spec2 += 19228 + 828
    elif spec2 <= 3300000:
        mag = 2. + np.ceil((spec2 - 2031000) / ((spec2 - 2031000) / (3300000 - 2031000) * 550 + 1995)) / 1000
        fan2[0] += 3 * 5 * np.ceil(score * cmag * mag * n)
        spec2 += 19228 + 828
    else:
        mag = 2.5
        fan2[0] += 3 * 5 * np.ceil(score * cmag * mag * n)
        spec2 += 19228

    stamina += 45

stamina2 = stamina
stamina = 0

while fan3 < th:
    if spec3 <= 741000:
        mag = 1.3 + np.ceil(spec3 / 7410.) / 1000.
        fan3[0] += 3 * 5 * np.ceil(score * cmag * mag * n)
        spec3 += 22990 * smag + 990
    elif spec3 <= 2031000:
        mag = 1.4 + np.ceil((spec3 - 741000.) / 2150.) / 1000.
        fan3[0] += 3 * 5 * np.ceil(score * cmag * mag * n)
        spec3 += 22990 + 990
    elif spec3 <= 3300000:
        mag = 2. + np.ceil((spec3 - 2031000) / ((spec3 - 2031000) / (3300000 - 2031000) * 550 + 1995)) / 1000
        fan3[0] += 3 * 5 * np.ceil(score * cmag * mag * n)
        spec3 += 22990 + 990
    else:
        mag = 2.5
        fan3[0] += 3 * 5 * np.ceil(score * cmag * mag * n)
        spec3 += 22990

    stamina += 50

stamina3 = stamina

print("40の場合\t試行回数:{}，消費スタミナ:{}，ファン数:{}，観客動員数:{}".format(stamina1 / 40, stamina1, fan1[0], spec1))
print("45の場合\t試行回数:{}，消費スタミナ:{}，ファン数:{}，観客動員数:{}".format(stamina2 / 45, stamina2, fan2[0], spec2))
print("50の場合\t試行回数:{}，消費スタミナ:{}，ファン数:{}，観客動員数:{}".format(stamina3 / 50, stamina3, fan3[0], spec3))
