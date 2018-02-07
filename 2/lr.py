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
    def __init__(self, batch=60, ridge=0.0):
        self.batch = batch
        self.ridge = ridge

    def get_predictor(self, inputs):
        def prd(x):
            x = inputs.normalize(inputs.model.at(x))
            return np.dot(x, self.tt)
        return prd

    def run(self, xs, ys, initial_rate = .005, decay = 0.995, precision=0.01, limit=5000):
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

        tt = np.zeros(xs.row_len())
        ttup = np.zeros(xs.row_len())

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
            for i in range(xs.len()):
                steps += 1
                err_sum += learn(tt, ttup, xs[i], ys[i], rate)
                if steps % self.batch == 0:
                    tt, stepsize = update(tt, ttup, rate)
                    ttup = np.zeros(xs.row_len())

            error_rate.append(err_sum / xs.len())
            if epoch % 100 == 0:
                log("      %d\t%d\t%s, err %.02f, stepsize %0.2f" % \
                        (epoch, steps, tt_to_s(tt), err_sum, stepsize))

        self.error_rate = error_rate
        self.tt = tt
        return epoch

class ClosedFormSolution:
    def __init__(self, n=0, delta=0.0):
        self.ridge = np.zeros((n,n))
        for i in range(n):
            self.ridge[i][i] = delta

    def get_predictor(self, inputs):
        def prd(x):
            x = inputs.normalize(inputs.model.at(x))
            return np.dot(x, self.tt)
        return prd

    def run(self, xs, ys):
        a = xs.asNp()
        at = a.transpose()

        inv = np.linalg.inv(np.add(np.matmul(at, a), self.ridge))

        self.tt = np.matmul( \
                np.matmul(inv, at), \
            ys)

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
    parser.add_argument('--batch', help="list of batch sizes for gd", nargs="+", type=int, default=[60])
    parser.add_argument('--plot', help="plot results", action='store_true', default=False)
    parser.add_argument('--poly', help="polynom model max n", nargs="+", type=int, required=True)
    parser.add_argument('-v', help="verbose", action='store_true', default=False)
    parser.add_argument('--error', help="evaluate mean square error of the solutions", action='store_true', default=False)
    parser.add_argument('--ridge', help="list of ridge regression lamdas to try", nargs="+", type=float, default=[])
    parser.add_argument('--kcross', help="k-cross validations", nargs="+", type=int, default=[])

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
    train = [txs, tys]

    test_x = mat['X_tst']
    test_y = mat['Y_tst']
    vxs = [x for [x] in test_x]
    vys = [y for [y] in test_y]
    validation = [vxs, vys]

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

    def print_res(name, tt, p, tm, err):
        print ("%s \t tt: %s %s took: %7.02fms, err: %6.02f" % \
                (name, tt_to_s(tt), (5-p)*"\t", tm*1000, err))

    def run_experiment(name, poly, algo, train, validation, output=True):
        inputs = NormalizedInputs(train[0], PolyModel(poly))

        t1 = time.time()
        algo.run(inputs, train[1])
        t2 = time.time()

        p = algo.get_predictor(inputs)

        predictors.append([name, p])
        if hasattr(algo, "error_rate"):
            error_rates.append(algo.error_rate)

        if len(name) < 10:
            name += (10-len(name))*" "

        e1 = getSquaredError(train[0], train[1], p)
        if output:
            print_res(name, algo.tt, poly, t2-t1, e1)

        e2 = getSquaredError(validation[0], validation[1], p)
        if args.error and output:
            print("%s \t mean square err: %.02f (train) and %.02f (test)" % \
                    (name, e1, e2))
        return e2

    def k_cross_validate(ks, todo):
        for k in ks:
            print ("-----== %d cross validation ==-----------" % (k))
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
                err = run_experiment(v[0], v[1], v[2], [k_train_x, k_train_y], [k_test_x, k_test_y])
                print ("error : %5.2f" % (err))


    todo = []

    if run_cf:
        for l in lambdas:
            for i,p in enumerate(polys):
                if no_ridge:
                    name = "CF_"+str(p)+"p"
                    log("Computing closed form solution for poly(%d) model (%s)" % (p, name))
                else:
                    name = "CF_"+str(p)+"p_"+str(l)+"l"
                    log("Computing closed form solution for poly(%d) model and ridge factor %.2f (%s)" % (p, l, name))
                cf = ClosedFormSolution(p+1, l)
                todo.append([name, p, cf])

    if run_gd:
        for l in lambdas:
            for i,p in enumerate(polys):
                for batch in batches:
                    if no_ridge:
                        name = "SG_"+str(batch)+"m_"+str(p)+"p"
                        log("Running stochastic gradient with a poly(%d) model, batch size %d (%s)" % (p, batch, l))
                    else:
                        name = "SG_"+str(batch)+"m_"+str(p)+"p_"+str(l)+"l"
                        log("Running stochastic gradient with a poly(%d) model, batch size %d and ridge factor %.2f (%s)" % (p, batch, l, name))
                    sg = SG(batch, l)
                    todo.append([name, p, sg])

    if len(args.kcross) > 0:
        k_cross_validate(args.kcross, todo)
    else:
        for v in todo:
            run_experiment(v[0], v[1], v[2], train, validation)

    if args.plot:
        plot([["train", txs, tys], ["test", vxs, vys]], predictors, error_rates);

if __name__ == "__main__":
    main()
