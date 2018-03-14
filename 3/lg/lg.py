#!/usr/bin/env python3

import scipy.io
import numpy as np
import time
import sys
from subprocess import call
import os
import matplotlib
import matplotlib.pyplot as plt

def lg(xs, ys, n, rate, limit):

    tt = np.zeros(len(xs[0]))
    error = float("Inf")

    while n > 0:
        n -= 1

        # predict using current value of theta (tt)
        pred = np.dot(xs, tt)
        pred = 1 / (1 + np.exp(-pred))

        # update tt
        err = ys - pred
        grd = np.dot(xs.T, err)
        tt += rate*grd

        if n % 1000 == 0:
            new_error = np.sum(err * err)
            # convergence if the error does not improve anymore
            if abs(error - new_error) < limit:
                return tt
            error = new_error
            print(error, tt)

    return tt


data = sys.argv[1]

mat = scipy.io.loadmat(data)

train_x = mat['X_trn']
train_y = [x for [x] in mat['Y_trn']]
test_x = mat['X_tst']
test_y = [x for [x] in mat['Y_tst']]

# Do logistic regression
t1 = time.time()
tt = lg(train_x, train_y, 100000, 0.1, 1e-3)
t2 = time.time()

print("\n%s: θ=%s\ttook: %7.02fms" % (data, tt, (t2-t1)*1000))

slope = -tt[0]/tt[1]
def predict(x1, x2):
    return x2 > x1*slope

# get the confusion matrix
def confusion(xs, ys):
    pred = [predict(x1, x2) for [x1, x2] in xs]
    tt = 0
    tf = 0
    ft = 0
    ff = 0
    for i in range(len(pred)):
        p = pred[i]
        y = ys[i]
        if p:
            if y==1:
                tt += 1
            else:
                tf += 1
        else:
            if y==1:
                ft += 1
            else:
                ff += 1
    return [[tt, tf],[ft,ff]]

print("Confusion matrix for training data: ", \
        confusion(train_x, train_y))
print("Confusion matrix for test data: ", \
        confusion(test_x, test_y))

# plot results
test_x1 = [x for [x,_] in test_x]
test_x2 = [x for [_,x] in test_x]
train_x1 = [x for [x,_] in train_x]
train_x2 = [x for [_,x] in train_x]

line_x2 = [min(train_x2), max(train_x2)]
line_x1 = [min(train_x2)/slope, max(train_x2)/slope]

plt.scatter(train_x1, train_x2, c=train_y)
plt.scatter(test_x1, test_x2, \
        c=['red' if y==1 else 'green' for y in test_y])
plt.plot(line_x1, line_x2)
plt.title(data)
plt.show()
