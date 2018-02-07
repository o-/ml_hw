#!/usr/bin/env python3

import scipy.io
import numpy as np

import time

import matplotlib
import matplotlib.pyplot as plt

import argparse

verbose = False

def log(msg):
    if verbose:
        print(msg)


def tt_to_s(tt):
    return "[" + ", ".join(["%.02f" % c for c in tt]) + "]"


class NormalizedInputs:
    def __init__(self, xs, model):
        normalization = []

        model_size = len(model.at(1))
        mins = [float("Inf") for i in range(model_size)]
        maxs = [float("-Inf") for i in range(model_size)]

        for x in xs:
            row = model.at(x)
            for n in range(model_size):
                if mins[n] > row[n]:
                    mins[n] = row[n]
                if maxs[n] < row[n]:
                    maxs[n] = row[n]

        self.rng = [maxs[i] - mins[i] for i in range(model_size)]
        self.mins = mins
        self.model = model
        self.model_size = model_size

        inputs = [self.normalize(self.model.at(x)) for x in xs]

        self.inputs = inputs

    def normalize(self, x):
        return [1.0] + \
                [(x[i]-self.mins[i])/self.rng[i] \
                for i in range(self.model_size)]

    def __getitem__(self, i):
        return self.inputs[i]

    def row_len(self):
        return self.model_size+1

    def len(self):
        return len(self.inputs)

    def asNp(self):
        return np.array(self.inputs)


class SG:
    def __init__(self, xs, ys, ridge=0.0):
        self.xs = xs
        self.ys = ys
        self.ridge = ridge

    def get_predictor(self):
        def prd(x):
            x = self.xs.normalize(self.xs.model.at(x))
            return np.dot(x, self.tt)
        return prd

    def run(self, batch = 60, initial_rate = .005, decay = 0.995, precision=0.01, limit=5000):
        def error(res, target):
            return res-target

        def learn(tt, ttup, x, target, rate):
            pred = np.dot(x, tt)
            err = error(pred, target)
            for i in range(len(ttup)):
                ttup[i] += x[i]*err
            return err*err

        def update(tt, ttup, rate):
            ridge_term = np.multiply(self.ridge/2, tt)
            ttup = np.add(ttup, ridge_term)
            step = np.multiply(rate, ttup)
            stepsize = np.sqrt(np.sum(np.multiply(step, step)))
            newtt = np.subtract(tt, step)
            return (newtt, stepsize)

        tt = np.zeros(self.xs.row_len())
        ttup = np.zeros(self.xs.row_len())

        error_rate = []

        epoch = 0
        steps = 0
        err_sum = 0.0
        stepsize = float("Inf")
        rate = initial_rate

        while epoch < limit and stepsize > precision:
            rate *= decay
            prev_err_sum = err_sum
            err_sum = 0.0
            epoch += 1
            for i in range(self.xs.len()):
                steps += 1
                err_sum += learn(tt, ttup, self.xs[i], self.ys[i], rate)
                if steps % batch == 0:
                    tt, stepsize = update(tt, ttup, rate)
                    ttup = np.zeros(self.xs.row_len())

            error_rate.append(err_sum)
            if epoch % 100 == 0:
                log("      %d\t%d\t%s, err %.02f, stepsize %0.2f" % \
                        (epoch, steps, tt_to_s(tt), err_sum, stepsize))

        self.error_rate = error_rate
        self.tt = tt
        return epoch

class ClosedFormSolution:
    def __init__(self, xs, ys, delta=0.0):
        self.xs = xs
        self.ys = ys
        n = len(xs[0])
        self.ridge = np.zeros((n,n))
        for i in range(n):
            self.ridge[i][i] = delta

    def get_predictor(self):
        def prd(x):
            x = self.xs.normalize(self.xs.model.at(x))
            return np.dot(x, self.tt)
        return prd

    def run(self):
        a = self.xs.asNp()
        at = a.transpose()

        inv = np.linalg.inv(np.add(np.matmul(at, a), self.ridge))

        self.tt = np.matmul( \
                np.matmul(inv, at), \
            self.ys)

def plot(data, predictors, error_rates):
    fig, axes = plt.subplots(nrows=len(data)+1)

    legends = []
    for i,e in enumerate(error_rates):
        error_rate = [np.log(y) for y in e]
        p = axes[0].plot(range(len(e)), error_rate, '-')
        legends.append(predictors[i][0])
    axes[0].legend(legends)
    axes[0].set_title("Log Squared Error Rate")

    for i, [name, xs, ys] in enumerate(data):
        legends = ["y"]
        axes[i+1].plot(xs, ys, 'o')

        sorted_xs = sorted(xs)
        for p in predictors:
            l = p[0]
            p = p[1]
            py = [p(x) for x in sorted_xs]
            axes[i+1].plot(sorted_xs, py, '-', label=l)
            legends.append(l)

        axes[i+1].legend(legends)
        axes[i+1].set_title("Dataset "+name)

    plt.show()

class PolyModel:
    def __init__(self, n):
        self.n = n

    # 1.0 at the beginning is implicit...
    def at(self, x):
        return [x**i for i in range(1, self.n+1)]

    def __repr__(self):
        return "[1, " + \
                ", ".join(["x^"+str(i) for i in range(1, self.n+1)]) + "]"

def getSquaredError(xs, ys, predict):
    err = 0
    for i,x in enumerate(xs):
        e = ys[i] - predict(x)
        err += e*e
    return err/len(xs)

def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-i', help="input matrix filename", required=True)
    parser.add_argument('--cf', help="calculate closed form solution", action='store_true', default=False)
    parser.add_argument('--gd', help="calculate stochastic gradient descent solution", action='store_true', default=False)
    parser.add_argument('--ridge', help="list of ridge regression lamdas to try", nargs="+", type=float, default=[])
    parser.add_argument('--batch', help="list of batch sizes for gd", nargs="+", type=int, default=[60])
    parser.add_argument('--plot', help="plot results", action='store_true', default=False)
    parser.add_argument('--poly', help="polynom model max n", nargs="+", type=int, required=True)
    parser.add_argument('-v', help="verbose", action='store_true', default=False)
    parser.add_argument('--error', help="evaluate mean square error of the solutions", action='store_true', default=False)

    args = parser.parse_args()

    if not args.gd and not args.cf:
        print("Please use at least one of --cf or --gd")
        return

    global verbose
    verbose = args.v

    mat = scipy.io.loadmat(args.i)

    train_x = mat['X_trn']
    train_y = mat['Y_trn']
    txs = [x for [x] in train_x]
    tys = [y for [y] in train_y]

    polys = args.poly
    model = [PolyModel(n) for n in polys]

    inputs = [NormalizedInputs(txs, m) for m in model]

    batches = args.batch

    predictors = []
    error_rates = []

    run_ridges   = True

    run_gd = args.gd
    run_cf = args.cf

    lambdas = args.ridge
    no_ridge = False
    if lambdas == []:
        lambdas = [0.0]
        no_ridge = True

    def print_res(name, tt, tm, batches, iters, err):
        print ("%s \t tt: %s %s took: %5dms, batchsz: %4d, iters, %5d, err: %.02f" % \
                (name, tt_to_s(tt), (5-p)*"\t", tm*1000, batches, iters, err))

    if run_cf:
        for l in lambdas:
            for i,p in enumerate(polys):
                if no_ridge:
                    name = "CF_"+str(p)+"p"
                    log("Computing closed form solution for poly(%d) model (%s)" % (p, name))
                else:
                    name = "CF_"+str(p)+"p_"+str(l)+"l"
                    log("Computing closed form solution for poly(%d) model and ridge factor %.2f (%s)" % (p, l, name))
                cf = ClosedFormSolution(inputs[i], tys, l)
                t1 = time.time()
                cf.run()
                t2 = time.time()
                predictors.append([name, cf.get_predictor()])
                print ("%s \t\t tt: %s %s took:   %.02fms" % (name, tt_to_s(cf.tt), (5-p)*"\t", (t2-t1)*1000))


    if run_gd:
        for l in lambdas:
            for i,p in enumerate(polys):
                sg = SG(inputs[i], tys, l)
                for batch in batches:
                    if no_ridge:
                        name = "SG_"+str(batch)+"m_"+str(p)+"p"
                        log("Running stochastic gradient with a poly(%d) model, batch size %d (%s)" % (p, batch, l))
                    else:
                        name = "SG_"+str(batch)+"m_"+str(p)+"p_"+str(l)+"l"
                        log("Running stochastic gradient with a poly(%d) model, batch size %d and ridge factor %.2f (%s)" % (p, batch, l, name))
                    t1 = time.time()
                    iters = sg.run(batch)
                    t2 = time.time()
                    predictors.append([name, sg.get_predictor()])
                    error_rates.append(sg.error_rate)
                    print_res(name, sg.tt, t2-t1, batch, iters, sg.error_rate[-1])


    test_x = mat['X_tst']
    test_y = mat['Y_tst']
    vxs = [x for [x] in test_x]
    vys = [y for [y] in test_y]

    if (args.error):
        if len(predictors) > 0:
            log("Testing on train and validation DS")
        for p in predictors:
            e1 = getSquaredError(txs, tys, p[1])
            e2 = getSquaredError(vxs, vys, p[1])
            print("%s \t mean squar err: %.02f (train) and %.02f (test)" % (p[0], e1, e2))

    if args.plot:
        plot([["train", txs, tys], ["test", vxs, vys]], predictors, error_rates);

if __name__ == "__main__":
    main()
