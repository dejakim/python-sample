#!/usr/bin/env python
# encoding: utf-8

'''
interp.py
An example of using interp2d for smoothing image
Author: Daewung Kim (skywalker.deja@gmail.com)
'''

import numpy as np
import cv2
import argparse
from scipy import interpolate
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(description='Smoothing image by applying interp2d')
parser.add_argument('--input', '-i', help='Input file')
parser.add_argument('--output', '-o', help='Output file')
args = parser.parse_args()

img = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
if img is None:
    print('Could not open input file: %s' % args.input)
    exit(0)

src = img.astype(np.float32) / 255.0

h,w = src.shape[:2]
x = np.linspace(0,w-1,8).astype(np.int32)
y = np.linspace(0,h-1,8).astype(np.int32)
xx,yy = np.meshgrid(x, y)
z = src[xx,yy]

f = interpolate.interp2d(x, y, z, kind='cubic')

xnew = np.arange(0, w-1)
ynew = np.arange(0, h-1)
znew = f(xnew, ynew)

#znew = np.floor(znew * 16.0) / 16.0

if args.output is None:
    plt.imshow(znew),plt.title('Result')
    plt.xticks([]), plt.yticks([])
    plt.show()
else:
    cv2.imwrite(args.output, znew * 255.0)
    print('Save result to: %s' % args.output)
