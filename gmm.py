#!/usr/bin/env python
# encoding: utf-8

'''
gmm.py
A simple implementation of Gaussian Mixture Model
Author: Daewung Kim (skywalker.deja@gmail.com)
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

EPS = 0.001
MAX_ITER = 2000

def partition(a,n):
    p = []
    idx = 0
    step = int(len(a) / n)
    for i in range(n-1):
        p.append(a[idx : idx+step])
        idx += step
    p.append(a[idx : len(a)])
    return p

def cov(a,m):
    (n, d) = a.shape[:2]
    s = a - np.array([m,] * n)
    c = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            for k in range(n):
                c[i,j] += s[k,i] * s[k,j]
            c[i,j] /= (n - 1)
    return c

def gaussian(x,m,t,a):
    z = x - m
    return np.exp(-0.5 * (z.dot(t)).dot(z)) / a

def gmm(data, K):
    (N, D) = data.shape[:2]
    m = np.zeros((K,D)) # mean
    s = np.zeros((K,D,D)) # covariance
    p = np.zeros(K) # mixing coeff
    n = np.zeros(K) # effective number
    r = np.zeros((N,K)) # responsibility
    #
    # Initialize
    seg = partition(data, K)
    for k in range(K):
        m[k] = np.average(seg[k], axis=0)
        s[k] = cov(seg[k], m[k])
        p[k] = 1.0

    lnp = 0.0
    for i in range(MAX_ITER):
        g = np.zeros(K)
        t = np.zeros((K,D,D))
        #
        # Expectation
        for k in range(K):
            g[k] = np.power(2.0 * np.pi, D * 0.5) * np.sqrt(np.linalg.det(s[k]))
            t[k] = np.linalg.inv(s[k])
        lnpnew = 0.0
        for j in range(N):
            b = 0.0
            for k in range(K):
                r[j,k] = gaussian(data[j], m[k], t[k], g[k])
                b += r[j,k]
            r[j] /= b
            lnpnew += np.log(b)
        # Maximization
        for k in range(K):
            n[k] = 0.0
            for j in range(N):
                n[k] += r[j,k]

            p[k] = n[k] / N

            s[k] = np.zeros((D,D))
            for j in range(N):
                z = data[j] - m[k]
                s[k] += r[j,k] * np.outer(z,z)
            s[k] /= n[k]

            m[k] = np.zeros(D)
            for j in range(N):
                m[k] += r[j,k] * data[j]
            m[k] /= n[k]

        if abs(lnpnew - lnp) < EPS:
            break
        lnp = lnpnew
    return r

def plot(pts, grp):
    df = pd.DataFrame({"x": [v[0] for v in pts],
                       "y": [v[1] for v in pts],
                       "cluster": [np.argmax(v) for v in grp]
                       })
    sns.lmplot("x", "y", data=df, fit_reg=False, size=6, hue="cluster", legend=False)
    plt.show()

if __name__ == '__main__':
    numpts = 2000
    data = np.zeros((numpts, 2))
    for i in range(numpts):
        dice = np.random.random()
        if dice < 0.3:
            data[i] = np.array([np.random.normal(0.0, 0.9), np.random.normal(0.0, 0.9)])
        elif dice < 0.6:
            data[i] = np.array([np.random.normal(1.0, 0.5), np.random.normal(2.0, 0.3)])
        else:
            data[i] = np.array([np.random.normal(3.0, 0.2), np.random.normal(1.0, 0.5)])

    label = gmm(data, 2)
    plot(data, label)
