#!/usr/bin/env python
# encoding: utf-8

'''
Levenstein.py
An example of getting Levenstein Distance
Author: Daewung Kim (skywalker.deja@gmail.com)
'''

import argparse
import numpy as np

def editdist(src, trg):
    D0 = np.arange(len(trg) + 1)
    D1 = np.arange(len(trg) + 1)
    for i in range(len(src)):
        D1[0] = i + 1
        c = src[i]
        for j in range(len(trg)):
            if (c == trg[j]):
                D1[j + 1] = D0[j]
            else:
                D1[j + 1] = min([D1[j], D0[j], D0[j + 1]]) + 1
        D0, D1 = D1, D0
    return D0[-1]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get Levenstein Distance')
    parser.add_argument('--source', '-s', default='text toast', help='Source text')
    parser.add_argument('--target', '-t', default='next post', help='Target text')
    args = parser.parse_args()

    print("dist(%s, %s) = %s" % (args.source, args.target, editdist(args.source, args.target)))
