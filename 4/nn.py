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

mat = scipy.io.loadmat("dataset.mat")

train_x = mat['X_train']
train_y = mat['Y_train']
test_x = mat['X_test']
test_y = mat['Y_test']

np.random.seed(12345)
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

    def print_errs(self, i, iters, xs1, xs2, ys, test_x, test_y):
        res = np.array(  \
                [self.forward([x1,x2])[1][-1][0]  \
                    for x1,x2 in zip(xs1, xs2)])
        err = np.sum(np.dot(res-ys, res-ys))
        miss = 0
        for a, y in zip(res, ys):
            if (y == 0 and a > 0.5) or (y == 1 and a <= 0.5):
                miss += 1
        if not test_x is None:
            tres = np.array(  \
                    [self.forward([x1,x2])[1][-1][0]  \
                        for x1,x2 in zip(test_x[0], test_x[1])])
            terr = np.sum(np.dot(tres-test_y[0], tres-test_y[0]))
            tmiss = 0
            for a, y in zip(tres, test_y[0]):
                if (y == 0 and a > 0.5) or (y == 1 and a <= 0.5):
                    tmiss += 1

        print("%d / %d : %f (%d), %f (%d)" % \
                (i, iters, err, miss, terr, tmiss))
        return miss+tmiss


    def gd(self, xs, ys, test_x, test_y, ld = 0.5, alpha = 0.5, iters = 2000) :
        N = len(ys[0])
        xs1 = xs[0]
        xs2 = xs[1]
        ys = ys[0]
        for i in range(iters):
            if i % 100 == 0:
                miss = self.print_errs(i, iters, xs1, xs2, ys, \
                        test_x, test_y)
                if miss == 0:
                    return

            dW, db = self.backward([xs1[0],xs2[0]], ys[0])
            for x1,x2,y in zip(xs1[1:], xs2[1:], ys[1:]):
                ddW, ddb = self.backward([x1,x2], y)
                dW = [dW+ddW for dW,ddW in zip(dW, ddW)]
                db = [db+ddb for db,ddb in zip(db, ddb)]

            wdiff = 0
            for i in range(0, len(self.S)-1):
                self.W[i] -= (alpha/N) * (ld * self.W[i] + dW[i])
                self.b[i] -= (alpha/N) * db[i]
                wdiff += np.sum(np.abs(dW[i]))
                wdiff += np.sum(np.abs(db[i]))
        self.print_errs(iters, iters, xs1, xs2, ys, test_x, test_y)

#import cProfile
#cProfile.run('nn.gd(train_x, train_y, iters=200, test_x=test_x, test_y=test_y)')


nn = NN([2,10,1], ['tanh', 'tanh', 'sigm'])
#nn = NN([2,10,1], 'sigm')
#nn = NN([2,5,2,1], 'tanh')
#nn = NN([2,10,1], 'relu')

#for ld in [0, 0.1, 0.2, 0.5, 1, 1.5, 2]:
#    print("------ %f -------------" % ld)
#    nn.clear()
#    nn.gd(train_x, train_y, ld=ld, iters=2000, test_x=test_x, test_y=test_y)

nn.gd(train_x, train_y, test_x, test_y)

s = 0
for W, b in zip (nn.W, nn.b):
    s += 1
    print("W%d" %s)
    print(W)
    print("b%d" %s)
    print(b)

res = [nn.forward([x1,x2])[1][-1][0] for x1,x2 in zip(train_x[0], train_x[1])]
tres = [nn.forward([x1,x2])[1][-1][0] for x1,x2 in zip(test_x[0], test_x[1])]

plt.scatter(train_x[0], train_x[1], \
        c=['orange' if y>0.5 else 'green' for y in res])
plt.scatter(test_x[0], test_x[1], \
        c=['yellow' if y>0.5 else 'blue' for y in tres])

miss1 = []
miss2 = []
for x1, x2, a, y in zip(train_x[0], train_x[1], res, train_y[0]):
    if (y == 0 and a > 0.5) or (y == 1 and a <= 0.5):
        miss1.append(x1)
        miss2.append(x2)
for x1, x2, a, y in zip(test_x[0], test_x[1], tres, test_y[0]):
    if (y == 0 and a > 0.5) or (y == 1 and a <= 0.5):
        miss1.append(x1)
        miss2.append(x2)

plt.scatter(miss1, miss2, s=80, facecolors='none', edgecolors='r')

plt.show()
