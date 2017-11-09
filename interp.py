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


def stair(a, c):
    return (np.floor(a * c) / c)

parser = argparse.ArgumentParser(description='Smoothing image by applying interp2d')
parser.add_argument('--input', '-i', help='Input file')
parser.add_argument('--output', '-o', help='Output file')
parser.add_argument('--grid', '-g', type=int, default=16, help='Grid size')
parser.add_argument('--size', '-s', type=int, nargs='+', default=[1024, 1024], help='Target size')
parser.add_argument('--level', '-l', type=float, default=23.0, help='Stair level')
args = parser.parse_args()

img = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
if img is None:
    print('Could not open input file: %s' % args.input)
    exit(0)

src = img.astype(np.float32) / 255.0    # Rescale value 255 -> 1.0
src = cv2.resize(src, tuple(args.size)) # Resize to target size
m = np.amin(src); M = np.amax(src)
src = (src - m) / (M - m)               # Normalize

h,w = src.shape[:2]
x = np.linspace(0,w-1,args.grid).astype(np.int32)
y = np.linspace(0,h-1,args.grid).astype(np.int32)
xx,yy = np.meshgrid(x, y)
z = src[yy,xx]

f = interpolate.interp2d(x, y, z, kind='cubic')

xn = np.arange(0, w-1)
yn = np.arange(0, h-1)
zn = f(xn, yn)

g, b = np.modf(zn * 255.0)
r, a = np.modf(zn * 65535.0)
res = cv2.merge((r * 255.0, g * 255.0, b))

if args.output is None:
    plt.imshow(stair(zn, args.level)),plt.title('Result')
    plt.xticks([]), plt.yticks([])
    plt.show()
else:
    cv2.imwrite(args.output, res)
    print('Save result to: %s' % args.output)
