#!/usr/bin/env python3

import scipy.io
import numpy as np

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def get(tt, xs):
    return np.dot(tt, xs)

def get_n(tt, normalization, x):
    xs = [1.0] + \
            [(x**(n+1) - normalization[n][0]) / \
             (normalization[n][1] - normalization[n][0]) \
             for n in range(len(normalization))]
    return np.dot(tt, xs)

def error(res, target):
    return target-res

def learn(tt, ttup, x, target, rate):
    pred = get(tt, x)
    err = error(pred, target)

    for i,v in enumerate(ttup):
        ttup[i] += x[i]*err

    return err*err

def update(tt, ttup, rate):
    for i,v in enumerate(tt):
        tt[i] += rate*ttup[i]
        ttup[i] = 0.0

def create_inputs(xs, poly):
    normalization = []

    res = []
    for n in range(1, poly+1):
        po = [x**n for x in xs]
        mi = np.min(po)
        ma = np.max(po)
        po = [(x-mi)/(ma-mi) for x in po]
        res.append(po)
        normalization.append([mi, ma])

    res2 = []
    for idx in range(len(xs)):
        res2.append([1.0] + [res[n][idx] for n in range(poly)])

    return [res2, normalization]

def sg(xs, ys):
    poly = 3

    inputs, normalization = create_inputs(xs, poly)

    tt = [0.0 for i in range(poly+1)]
    ttup = [0.0 for i in range(poly+1)]

    batch = 60
    rate = 0.01

    error_rate = []

    limit = 0.9999

    while (len(error_rate) < 10 or \
            (error_rate[-1] / error_rate[-2]) < limit):
        err_sum = 0.0
        for i,v in enumerate(xs):
            x = inputs[i]
            y = ys[i]
            err_sum += learn(tt, ttup, x, y, rate)
            if i % batch == (batch - 1):
                update(tt, ttup, rate)
        if (len(xs)%batch != 0):
            update(tt, ttup, rate)
        error_rate.append(err_sum)

    return [tt, normalization, error_rate]

def plot(data, tt, normalization, error_rate):
    fig, axes = plt.subplots(nrows=len(data)+1)

    error_rate = [np.log(x) for x in error_rate]
    axes[0].plot(range(len(error_rate)), error_rate, '-')

    for i, [xs, ys] in enumerate(data):
        axes[i+1].plot(xs, ys, 'o')

        sorted_xs = sorted(xs)
        py = [get_n(tt, normalization, x) for x in sorted_xs]
        axes[i+1].plot(sorted_xs, py, '-')

    print(tt)
    plt.show()


def main():
    mat = scipy.io.loadmat("dataset1.mat")

    train_x = mat['X_trn']
    train_y = mat['Y_trn']
    txs = [x for [x] in train_x]
    tys = [y for [y] in train_y]

    tt, normalization, error_rate = sg(txs, tys)

    test_x = mat['X_tst']
    test_y = mat['Y_tst']
    xs = [x for [x] in test_x]
    ys = [y for [y] in test_y]

    plot([[txs, tys],[xs, ys]], tt, normalization, error_rate)

if __name__ == "__main__":
    main()
