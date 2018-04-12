#!/usr/bin/env python3


# Attribution note:
#
# I misunderstood the notes. I thought layer 2 would go from z^2 to a^2, but
# it turns out it goes from z^1 to a^2. I had a hard time to find this bug
# and had to resort to delta debugging against the following reference
# implementation: http://neuralnetworksanddeeplearning.com/chap3.html


import scipy.io
import numpy as np
import time
import sys
from subprocess import call
import os
import matplotlib
import matplotlib.pyplot as plt
import math
import sklearn.datasets

mat = scipy.io.loadmat("dataset.mat")

np.random.seed(12345)

train_x = mat['X_train']
train_y = mat['Y_train']
test_x = mat['X_test']
test_y = mat['Y_test']

# X, y = sklearn.datasets.make_moons(200, noise=0.20)
# train_x = [X[:,0], X[:,1]]
# train_y = [y]
# test_x = None
# test_y = None


class NN:
    def __init__(self, S, activations):
        if isinstance(activations, str):
            activations = [activations] * len(S)

        def get_act(name):
            if name == 'sigm':
                def sigm(x):
                    return 1/(1+np.exp(-x))
                def dsigm(x):
                    f = 1/(1+np.exp(-x))
                    return np.multiply(f,(1-f))
                return (sigm, dsigm)
            elif name == 'tanh':
                def tanh(x):
                    return np.tanh(x)
                def dtanh(x):
                    f = np.tanh(x)
                    return 1-np.multiply(f,f)
                return (tanh, dtanh)
            elif name == 'relu':
                def relu(x):
                    r = np.copy(x)
                    r[r<0] = 0
                    return r
                def drelu(x):
                    r = np.copy(x)
                    r[r<0] = 0
                    r[r>0] = 1
                    return r
                return (relu, drelu)

        assert(len(S) == len(activations))
        activation_functions = [a for a in \
                zip(*[get_act(name) for name in activations])]
        self.act = activation_functions[0]
        self.dact = activation_functions[1]
        self.S = S
        self.depth = len(self.S)
        self.clear()

    def clear(self):
        self.W = []
        self.b = []
        for i in range(1, len(self.S)):
            self.b.append(np.random.normal(0, 1, size=self.S[i]))
            self.W.append(np.random.normal(0, 1, (self.S[i], self.S[i-1])))

    def derr(self, a, z, y):
        # cross entropy performed worse for everything but sigmoid...
        #d = a*(1-a)
        #return (a-y)*self.dact[-1](z) / (d if d != 0 else 0.001)
        return (a-y)

    def forward(self, inp):
        z = []
        cur = np.array(inp)
        a = [cur]

        # Forward
        for W, b, act in zip(self.W, self.b, self.act):
            cur = np.add(np.dot(W, cur), b)
            z.append(cur)
            cur = act(cur)
            a.append(cur)

        return (z,a)

    def backward(self, inp, y):
        z,a = self.forward(inp)

        # Backwards
        d = self.derr(a[-1],z[-1],y)
        dW = [np.dot(d, a[-2].reshape(1, len(a[-2])))]
        db = [d]

        for i in range(2, self.depth):
            zi = z[-i]
            df = self.dact[-i](zi)
            W = self.W[-i+1]
            s = np.dot(W.transpose(), d)
            d = np.multiply(s, df)
            ai = a[-i]
            dW.insert(0, np.dot(d, ai.transpose()))
            db.insert(0, d)

        return (dW, db)

    def predict(self, x1, x2):
        return self.forward([x1,x2])[1][-1][0]

    def predict_class(self, x1, x2):
        return self.predict(x1, x2) > 0.5

    def print_errs(self, i, iters, xs1, xs2, ys, test_x, test_y, wdiff):
        res = np.array([self.predict(x1,x2) for x1,x2 in zip(xs1, xs2)])
        err = np.sum(np.dot(res-ys, res-ys))
        miss = 0
        for a, y in zip(res, ys):
            if (y == 0 and a > 0.5) or (y == 1 and a <= 0.5):
                miss += 1
        if not test_x is None:
            tres = np.array([self.predict(x1,x2) \
                    for x1,x2 in zip(test_x[0], test_x[1])])
            terr = np.sum(np.dot(tres-test_y[0], tres-test_y[0]))
            tmiss = 0
            for a, y in zip(tres, test_y[0]):
                if (y == 0 and a > 0.5) or (y == 1 and a <= 0.5):
                    tmiss += 1
        else:
            terr = 0
            tmiss = 0

        print("%d / %d [%f] : %f (%d), %f (%d)" % \
                (i, iters, wdiff, err, miss, terr, tmiss))

    def plot(self, xs, ys, test_x, test_y, block=True):
        res = [self.predict_class(x1, x2) for x1,x2 in zip(train_x[0], train_x[1])]
        if not test_x is None:
            tres = [self.predict_class(x1,x2) for x1,x2 in zip(test_x[0], test_x[1])]
        if block:
            grid = 0.01
        else:
            grid = 0.2
        xmin1 = min([10e100 if test_x is None else test_x[0].min() - .5, train_x[0].min() - .5])
        xmin2 = min([10e100 if test_x is None else test_x[1].min() - .5, train_x[1].min() - .5])
        xmax1 = max([-10e100 if test_x is None else test_x[0].max() + .5, train_x[0].max() + .5])
        xmax2 = max([-10e100 if test_x is None else test_x[1].max() + .5, train_x[1].max() + .5])
        gx1, gx2 = np.meshgrid(np.arange(xmin1, xmax1, grid), np.arange(xmin2, xmax2, grid))
        cl = np.array([self.predict_class(x1,x2) for x1,x2 in np.c_[gx1.ravel(), gx2.ravel()]])
        cl = cl.reshape(gx1.shape)
        plt.contourf(gx1, gx2, cl, cmap=plt.cm.Spectral)
        plt.scatter(train_x[0], train_x[1], \
                c=['orange' if y else 'green' for y in train_y[0]])
        if not test_x is None:
            plt.scatter(test_x[0], test_x[1], \
                    c=['yellow' if y else 'blue' for y in test_y[0]])
        if block:
            miss1 = []
            miss2 = []
            for x1, x2, a, y in zip(train_x[0], train_x[1], res, train_y[0]):
                if (y == 0 and a) or (y == 1 and not a):
                    miss1.append(x1)
                    miss2.append(x2)
            if not test_x is None:
                for x1, x2, a, y in zip(test_x[0], test_x[1], tres, test_y[0]):
                    if (y == 0 and a) or (y == 1 and not a):
                        miss1.append(x1)
                        miss2.append(x2)
            plt.scatter(miss1, miss2, s=80, facecolors='none', edgecolors='r')

            plt.show()
        else:
            plt.pause(0.001)

    def gd(self, xs, ys, test_x=None, test_y=None, ld = 0.4, alpha = 0.4, iters = 10000, plot=False) :
        print("SG with ld=%f, alpa=%f" %(ld, alpha))
        N = len(ys[0])
        xs1 = xs[0]
        xs2 = xs[1]
        ys = ys[0]
        stuck = 0
        for i in range(iters):
            dW, db = self.backward([xs1[0],xs2[0]], ys[0])
            for x1,x2,y in zip(xs1[1:], xs2[1:], ys[1:]):
                ddW, ddb = self.backward([x1,x2], y)
                dW = [dW+ddW for dW,ddW in zip(dW, ddW)]
                db = [db+ddb for db,ddb in zip(db, ddb)]

            wdiff = 0
            for j in range(0, len(self.S)-1):
                self.W[j] -= (alpha/N) * (ld * self.W[j] + dW[j])
                self.b[j] -= (alpha/N) * db[j]
                wdiff += np.dot(dW[j].transpose(), dW[j])
                wdiff += np.dot(db[j].transpose(), db[j])

            if i % 100 == 0:
                if plot:
                    self.plot(xs, ys, test_x, test_y, block=False)
                self.print_errs(i, iters, xs1, xs2, ys, test_x, test_y, wdiff)
            if wdiff < np.sum(self.S)*3:
                stuck += 1
                if stuck > N*4:
                    break
            elif stuck > 0:
                stuck -= 1
        self.print_errs(i, iters, xs1, xs2, ys, test_x, test_y, 0)

#import cProfile
#cProfile.run('nn.gd(train_x, train_y, iters=200, test_x=test_x, test_y=test_y)')

assert(len(sys.argv)>2)

S = [int(float(s)) for s in str.split(sys.argv[1],',')]
a = str.split(sys.argv[2],",")
if len(a) == 1:
    a = a[0]

nn = NN(S, a)
#nn = NN([2,10,1], 'sigm')
#nn = NN([2,5,2,1], 'tanh')
#nn = NN([2,10,1], 'relu')

#for ld in [0, 0.1, 0.2, 0.5, 1, 1.5, 2]:
#    print("------ %f -------------" % ld)
#    nn.clear()
#    nn.gd(train_x, train_y, ld=ld, iters=2000, test_x=test_x, test_y=test_y)

if len(sys.argv)>4:
    nn.gd(train_x, train_y, test_x, test_y, alpha=float(sys.argv[3]), ld=float(sys.argv[4]), plot=True)
elif len(sys.argv)>3:
    nn.gd(train_x, train_y, test_x, test_y, alpha=float(sys.argv[3]), plot=True)
else:
    nn.gd(train_x, train_y, test_x, test_y, plot=True)

s = 0
for W, b in zip (nn.W, nn.b):
    s += 1
    print("W%d" %s)
    print(W)
    print("b%d" %s)
    print(b)

nn.plot(train_x, train_y, test_x, test_y)
