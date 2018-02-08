#!/usr/bin/env python3

import scipy.io
import numpy as np

import time

import matplotlib
import matplotlib.pyplot as plt

import argparse

import sys

import random

verbose = False
very_verbose = False

def log(msg):
    if verbose or very_verbose:
        print(msg)

def tt_to_s(tt):
    res =  "[" + ", ".join(["%5.02f" % c for c in tt]) + "]"
    res += " "*(50-len(res))
    return res

def frange(x, y, jump):
    while x <= y:
        yield x
        x += jump


class NormalizedInputs:
    def __init__(self, xs, model, include_bias=True):
        normalization = []

        self.include_bias = include_bias
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
        return ([1.0] if self.include_bias else []) + \
                [((x[i]-self.mins[i])/self.rng[i]) \
                for i in range(self.model_size)]
    def __getitem__(self, i):
        return self.inputs[i]
    def row_len(self):
        return self.model_size + (1 if self.include_bias else 0)
    def len(self):
        return len(self.inputs)
    def asNp(self):
        return np.array(self.inputs)

class Inputs:
    def __init__(self, xs, model, include_bias=True):
        self.include_bias = include_bias
        model_size = len(model.at(1))
        self.model = model
        self.model_size = model_size
        self.inputs = [([1.0] if include_bias else []) \
                + self.model.at(x) for x in xs]
    def normalize(self, x):
        if self.include_bias:
            return [1.0] + x
        else:
            return x
    def __getitem__(self, i):
        return self.inputs[i]
    def row_len(self):
        return self.model_size + (1 if self.include_bias else 0)
    def len(self):
        return len(self.inputs)
    def asNp(self):
        return np.array(self.inputs)



class SG:
    def __init__(self, batch, ridge, initial_rate, decay, precision, limit, randomize):
        self.randomize = randomize
        self.batch = batch
        self.ridge = ridge*ridge
        self.initial_rate = initial_rate
        self.decay = decay
        self.precision = precision
        self.limit = limit

    def get_predictor(self, inputs):
        def prd(x):
            x = inputs.normalize(inputs.model.at(x))
            return np.dot(self.tt, x)
        return prd

    def run(self, xs, ys):
        def error(res, target):
            return res-target

        def learn(tt, ttup, x, target, rate):
            # predict using current value of theta (tt)
            pred = np.dot(tt, x)
            err = error(pred, target)
            # update ttup, which is our accumulator for the current batch
            for i in range(len(ttup)):
                ttup[i] += 2.0*x[i]*rate*err
            return err*err

        def update(tt, ttup, rate):
            # this function applies the accumulated step in ttup to tt
            if self.ridge != 0:
                # apply the regularization term
                penalize = [2.0* self.ridge * rate * t for t in tt]
                penalize[0] = 0.0
                ttup = np.add(ttup, penalize)
            stepsize = np.sqrt(np.sum(np.multiply(ttup, ttup)))
            newtt = np.subtract(tt, ttup)
            return (newtt, stepsize)

        tt = np.zeros(xs.row_len())
        ttup = np.zeros(xs.row_len())

        error_rate = []

        epoch = 0
        steps = 0
        err_sum = 0.0
        stepsize = float("Inf")
        rate = self.initial_rate

        global very_verbose
        while epoch < self.limit and stepsize > self.precision:
            rate *= self.decay
            prev_err_sum = err_sum
            err_sum = 0.0
            epoch += 1
            r = list(range(xs.len()))
            if self.randomize:
                random.shuffle(r)
            for i in r:
                steps += 1
                # add the gradient to the current batch update ttup
                err = learn(tt, ttup, xs[i], ys[i], rate)
                err_sum += err
                if very_verbose:
                    log("   cur: %s, %s, err %.02f" % \
                            (tt_to_s(ttup), tt_to_s(tt), err))
                if steps % self.batch == 0:
                    # apply the accumulated update to tt (theta)
                    tt, stepsize = update(tt, ttup, rate)
                    ttup = np.zeros(xs.row_len())
                    if very_verbose:
                        log("   step: %s" % (tt_to_s(tt)))

            error_rate.append(err_sum / xs.len())
            if very_verbose or epoch % 100 == 0:
                log("      %d\t%d\t%s, err %.02f, stepsize %0.2f" % \
                        (epoch, steps, tt_to_s(tt), err_sum, stepsize))

        self.error_rate = error_rate
        self.tt = tt
        return epoch

class ClosedFormSolution:
    def __init__(self, n=0, delta=0.0):
        self.ridge = np.zeros((n,n))
        for i in range(n):
            self.ridge[i][i] = delta*delta

    def get_predictor(self, inputs):
        def prd(x):
            x = inputs.normalize(inputs.model.at(x))
            return np.dot(self.tt, x)
        return prd

    def run(self, xs, ys):
        a = xs.asNp()
        at = a.transpose()

        inv = np.linalg.inv(np.add(np.matmul(at, a), self.ridge))

        self.tt = np.matmul( \
                np.matmul(inv, at), \
            ys)
        return 0

def plot(data, predictors, error_rates):
    fig, axes = plt.subplots(nrows=len(data)+1)

    legends = []
    for i,err in enumerate(error_rates):
        error_rate = [np.log(y) for y in err[1]]
        p = axes[0].plot(range(len(err[1])), error_rate, '-')
        legends.append(err[0])
    axes[0].legend(legends)
    axes[0].set_title("Log Squared Error Rate")

    for i, [name, xs, ys] in enumerate(data):
        legends = ["y"]
        axes[i+1].plot(xs, ys, 'o')

        l = np.min(xs)
        h = np.max(xs)
        if l < 0:
            l *= 1.1
        else:
            l *= 0.9
        if h > 0:
            h *= 1.1
        else:
            h *= 0.9
        xs = [x for x in frange(l, h, 0.1)]
        for p in predictors:
            l = p[0]
            p = p[1]
            py = [p(x) for x in xs]
            axes[i+1].plot(xs, py, '-', label=l)
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
    parser.add_argument('--no-normalize', help="don't normalize inputs", action='store_true', default=False)
    parser.add_argument('--cf', help="calculate closed form solution", action='store_true', default=False)
    parser.add_argument('--gd', help="calculate stochastic gradient descent solution", action='store_true', default=False)
    parser.add_argument('--batch', help="list of batch sizes for gd", nargs="+", type=int, default=[60])
    parser.add_argument('--plot', help="plot results", action='store_true', default=False)
    parser.add_argument('--no-randomize', help="don't randomize sample order", action='store_true', default=False)
    parser.add_argument('--poly', help="polynom model max n", nargs="+", type=int, required=True)
    parser.add_argument('-v', help="verbose", action='store_true', default=False)
    parser.add_argument('-vv', help="very verbose", action='store_true', default=False)
    parser.add_argument('--no-print-res', help="don't print result values", action='store_true', default=False)
    parser.add_argument('--error', help="evaluate mean square error of the solutions", action='store_true', default=False)
    parser.add_argument('--ridge', help="ridge regression lamdas", nargs="+", type=float, default=[])
    parser.add_argument('--ridge-seq', help="ridge regression lamdas to try, give: 'min max step'", nargs=3, type=float, default=[])
    parser.add_argument('--kcross', help="k-cross validations", nargs="+", type=int, default=[])

    parser.add_argument('--gd-init-rate', help="initial gradient descent learn rate", type=float, default=None)
    parser.add_argument('--gd-rate-decay', help="gd decay of learn rate per epoch", type=float, default=None)
    parser.add_argument('--gd-precision', help="gd convergence criterion: minimal step size", type=float, default=None)
    parser.add_argument('--gd-limit', help="gd number of epochs limit", type=float, default=5000)

    args = parser.parse_args()

    # Some sane default hyperparameters, need to be more conservative if not normalized
    if args.no_normalize:
        if args.gd and args.gd_init_rate is None:
            print("Please specify initial learn rate with --gd-init-rate")
            sys.exit(1)
        gd_init_rate = args.gd_init_rate
        gd_precision   = 0.00001   if args.gd_precision  is None else args.gd_precision
        gd_rate_decay  = 0.9999    if args.gd_rate_decay is None else args.gd_rate_decay
    else:
        gd_init_rate   = 0.005     if args.gd_init_rate  is None else args.gd_init_rate
        gd_rate_decay  = 0.999     if args.gd_rate_decay is None else args.gd_rate_decay
        gd_precision   = 0.001     if args.gd_precision  is None else args.gd_precision

    lambdas = []
    if args.ridge_seq != []:
        if len(args.ridge_seq) == 3:
            lambdas = [x for x in frange(args.ridge_seq[0], args.ridge_seq[1], args.ridge_seq[2])]
        else:
            print("Please give 3 values for --ridge-seq: start end step")
            sys.exit(1)
    if len(args.ridge) > 0:
        lambdas += args.ridge

    if not args.gd and not args.cf:
        print("Please use at least one of --cf or --gd")
        sys.exit(1)

    global verbose
    global very_verbose
    verbose = args.v
    very_verbose = args.vv

    mat = scipy.io.loadmat(args.i)

    train_x = mat['X_trn']
    train_y = mat['Y_trn']
    txs = [x for [x] in train_x]
    tys = [y for [y] in train_y]
    train = [txs, tys]

    test_x = mat['X_tst']
    test_y = mat['Y_tst']
    vxs = [x for [x] in test_x]
    vys = [y for [y] in test_y]
    validation = [vxs, vys]

    log("Loaded a dataset with %d train samples and %d validation samples" % \
            (len(train_x), len(test_x)))
    polys = args.poly

    batches = args.batch

    predictors = []
    error_rates = []

    run_ridges   = True

    run_gd = args.gd
    run_cf = args.cf

    no_ridge = False
    if lambdas == []:
        lambdas = [0.0]
        no_ridge = True

    def print_res(name, tt, p, tm, epoch, err):
        res = tt_to_s(tt)
        print ("%s θ=%s took: %7.02fms (%d)\terr: %6.02f" % \
                (name, res, tm*1000, epoch, err))

    def run_experiment(name, poly, algo, train, validation, output=True):
        if args.no_normalize:
            inputs = Inputs(train[0], PolyModel(poly))
        else:
            inputs = NormalizedInputs(train[0], PolyModel(poly))

        t1 = time.time()
        epoch = algo.run(inputs, train[1])
        t2 = time.time()

        p = algo.get_predictor(inputs)

        predictors.append([name, p])
        if hasattr(algo, "error_rate"):
            error_rates.append([name, algo.error_rate])

        name += (16-len(name))*" "
        e1 = getSquaredError(train[0], train[1], p)
        if not args.no_print_res and output:
            print_res(name, algo.tt, poly, t2-t1, epoch, e1)

        e2 = getSquaredError(validation[0], validation[1], p)
        if args.error and output:
            print("%s mean square err: %8.02f (train) and %8.02f (test)" % \
                    (name, e1, e2))
        return (e2, algo.get_predictor(inputs), algo.tt)

    def k_cross_validate(ks, todo):
        for k in ks:
            minimal_e1 = ["", float("inf"), float("inf"), float("inf")]
            minimal_e2 = ["", float("inf"), float("inf"), float("inf")]
            minimal_e3 = ["", float("inf"), float("inf"), float("inf")]

            k_train_x = []
            k_train_y = []
            k_test_x = []
            k_test_y = []

            for i in range(len(train[0])):
                if (i+1) % k != 0:
                    k_train_x.append(train[0][i])
                    k_train_y.append(train[1][i])
                else:
                    k_test_x.append(train[0][i])
                    k_test_y.append(train[1][i])

            for v in todo:
                e1, pred, tt = run_experiment(v[0], v[1], v[2], [k_train_x, k_train_y], [k_test_x, k_test_y])
                e2 = getSquaredError(train[0], train[1], pred)
                e3 = getSquaredError(validation[0], validation[1], pred)
                log("\tcross: %5.2f, train: %5.2f, test: %5.2f" % (e1, e2, e3))

                if e1 < minimal_e1[1]:
                    minimal_e1 = [v[0], e1, e2, e3, tt_to_s(tt)]
                if e2 < minimal_e2[2]:
                    minimal_e2 = [v[0], e1, e2, e3, tt_to_s(tt)]
                if e3 < minimal_e3[3]:
                    minimal_e3 = [v[0], e1, e2, e3, tt_to_s(tt)]

            print("best train      err in %s with %8.2f train, %8.2f cross val, and %8.2f validation error\n    θ=%s" % tuple(minimal_e1))
            print("best cross val. err in %s with %8.2f train, %8.2f cross val, and %8.2f validation error\n    θ=%s" % tuple(minimal_e2))
            print("best validation err in %s with %8.2f train, %8.2f cross val, and %8.2f validation error\n    θ=%s" % tuple(minimal_e3))

    todo = []

    if run_gd:
        for l in lambdas:
            for i,p in enumerate(polys):
                for batch in batches:
                    if no_ridge:
                        name = "SG(x^%d, m=%2d)" %(p, batch)
                        log("Running stochastic gradient with a poly(%d) model, batch size %d (%s)" % (p, batch, l))
                    else:
                        name = "SG(x^%d, λ=%6.2f, m=%2d)" %(p, l, batch)
                        log("Running stochastic gradient with a poly(%d) model, batch size %d and ridge factor %.2f (%s)" % (p, batch, l, name))

                    sg = SG(batch, l, gd_init_rate, gd_rate_decay, gd_precision, args.gd_limit, not args.no_randomize)
                    todo.append([name, p, sg])

    if run_cf:
        for l in lambdas:
            for i,p in enumerate(polys):
                if no_ridge:
                    name = "CF(x^%d)" % p
                    log("Computing closed form solution for poly(%d) model (%s)" % (p, name))
                else:
                    name = "CF(x^%d, λ=%6.2f)" % (p, l)
                    log("Computing closed form solution for poly(%d) model and ridge factor %.2f (%s)" % (p, l, name))
                cf = ClosedFormSolution(p+1, l)
                todo.append([name, p, cf])


    if len(args.kcross) > 0:
        k_cross_validate(args.kcross, todo)
    else:
        for v in todo:
            run_experiment(v[0], v[1], v[2], train, validation)

    if args.plot:
        plot([["train", txs, tys], ["test", vxs, vys]], predictors, error_rates);

if __name__ == "__main__":
    main()
