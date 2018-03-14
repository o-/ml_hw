#!/usr/bin/env python3

import scipy.io
import numpy as np
import time
import sys
from subprocess import call
import os
import matplotlib
import matplotlib.pyplot as plt
import random

data = sys.argv[1]

mat = scipy.io.loadmat(data)

train_x = mat['X_trn']
train_y = [1 if y == 1 else -1 for [y] in mat['Y_trn']]
test_x = mat['X_tst']
test_y = [1 if y == 1 else -1 for [y] in mat['Y_tst']]

tol = 0.00001
max_passes = 3500

def smo(x, y, C):
    random.seed(42)
    m = len(y)
    a = np.zeros(len(y))
    b = 0
    passes = 0

    # cache a*y to speedup f below
    ay = np.multiply(a, y)

    def f(q):
        sv = [np.dot(x, q) for x in x]
        return b + np.sum(np.multiply(sv, ay))

    iters = 0
    while passes < max_passes:
        iters += 1
        num_changed_alphas = 0
        for i in range(m):
            Ei = f(x[i])-y[i]
            #print("Ei ", i, " ", Ei, " ", f(x[i]), " ", x[i], " ", y[i], " ", y[i]*Ei )
            if (y[i]*Ei < -tol and a[i] < C) or (y[i]*Ei > tol and a[i] > 0):
                j = i
                while j == i:
                    j = random.randint(0, m-1)
                Ej = f(x[j]) - y[j]
                #print("Ej ", j, " ", Ej, " ", f(x[j]), " ", x[j], " ", y[j], " ", y[j]*Ej )
                old_a = a[j]

                if y[i] != y[j]:
                    L = max(0, a[j]-a[i])
                    H = min(C, C+a[j]-a[i])
                else:
                    L = max(0, a[i]+a[j]-C)
                    H = min(C, a[i]+a[j])
                if H==L:
                    continue

                dxi = np.dot(x[i],x[i])
                dxj = np.dot(x[j],x[j])
                dxij = np.dot(x[i],x[j])
                n = 2*dxij - dxi - dxj

                if n >= 0:
                    continue

                new_a = a[j] - (y[j]*(Ei-Ej) / n)
                new_a = min(new_a, H)
                new_a = max(new_a, L)
                diff = new_a - old_a
                if abs(diff) < 1e-5:
                    continue

                a[j] = new_a
                a[i] += y[j]*y[i]*(old_a - new_a)
                if 0 < a[i] and a[i] < C:
                    b1 = b - Ei - y[i]*diff*dxi - y[j]*diff*dxij
                    b = b1
                else:
                    if 0 < a[j] and a[j] < C:
                        b2 = b - Ej - y[i]*diff*dxij - y[j]*diff*dxj
                        b = b2
                    else:
                        b1 = b - Ei - y[i]*diff*dxi - y[j]*diff*dxij
                        b2 = b - Ej - y[i]*diff*dxij - y[j]*diff*dxj
                        b = (b1+b2)/2

                ay = np.multiply(a, y)
                num_changed_alphas += 1
            if num_changed_alphas == 0:
                passes += 1
            else:
                passes = 0

    # print ("a = ", a)
    # print ("b = ", b)
    # print ("converged in ", iters)
    return f

def frange(x, y, jump):
  while x < y:
    yield x
    x += jump

def get_err(xs, ys, predict):
    pred = [predict(x)>0 for x in xs]
    ok = 0
    nok = 0
    for i in range(len(pred)):
        p = pred[i]
        y = ys[i]
        if (p and y==1) or (not p and y==-1):
            ok += 1
        else:
            nok += 1
    return nok/(ok+nok)

from sklearn.svm import SVC

start = float(sys.argv[2])
step = float(sys.argv[4]) if len(sys.argv) > 4 else 0.05
end = float(sys.argv[3]) if len(sys.argv) > 3 else start+ (step/2)

for C in frange(start, end, step):
    clf = SVC(C=C, kernel='linear')
    clf.fit(train_x, train_y)
    predict2 = lambda x: clf.predict([x])
    print("C=%0.02f  sklearn: train_err=%0.02f%%  test_err=%0.02f%%" % \
            (C,\
            get_err(train_x, train_y, predict2)*100, \
            get_err(test_x, test_y, predict2)*100 \
            ))
    predict1 = smo(train_x, train_y, C)
    print("        SMO:     train_err=%0.02f%%  test_err=%0.02f%%" % \
            ( \
            get_err(train_x, train_y, predict1)*100, \
            get_err(test_x, test_y, predict1)*100, \
            ))
