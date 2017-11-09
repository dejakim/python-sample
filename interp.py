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
parser.add_argument('--grid', '-g', default=16, help='Grid size')
parser.add_argument('--size', '-s', nargs='+', default=[1024, 1024], type=int, help='Target size')
args = parser.parse_args()

img = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
if img is None:
    print('Could not open input file: %s' % args.input)
    exit(0)

src = cv2.equalizeHist(img).astype(np.float32)
src = cv2.resize(src, tuple(args.size))
src = src * 0.7 / 255.0

h,w = src.shape[:2]
x = np.linspace(0,w-1,args.grid).astype(np.int32)
y = np.linspace(0,h-1,args.grid).astype(np.int32)
xx,yy = np.meshgrid(x, y)
z = src[yy,xx]

f = interpolate.interp2d(x, y, z, kind='cubic')

xn = np.arange(0, w-1)
yn = np.arange(0, h-1)
zn = f(xn, yn)

zn = np.floor(zn * 16.0) / 16.0

if args.output is None:
    plt.imshow(zn),plt.title('Result')
    plt.xticks([]), plt.yticks([])
    plt.show()
else:
    cv2.imwrite(args.output, zn * 255.0)
    print('Save result to: %s' % args.output)
