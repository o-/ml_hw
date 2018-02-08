#!/bin/sh

echo
echo "* closed form solution θ for n=2,3,5"
./lr.py -i dataset1.mat --poly 2 3 5 --cf --no-normalize

echo
echo "* mean square errors for closed form solutions"
./lr.py -i dataset1.mat --poly 2 3 5 --cf --err --no-print-res --no-normalize

echo
echo "* sgd solution θ for n=2"
./lr.py -i dataset1.mat --poly 2 --gd --gd-init-rate 0.0003 --gd-precision 0.03 --err --no-normalize

echo
echo "* sgd solution θ for n=3"
./lr.py -i dataset1.mat --poly 3 --gd --gd-init-rate 0.000028 --gd-precision 0.001 --no-normalize

# n = 5 overflows

echo
echo
echo "--= normalizing input values makes regression much more stable! =--"
echo "                from now on normalize to [0..1]"

echo
echo "* closed form solution θ and mean square error for n=2,3,5"
./lr.py -i dataset1.mat --poly 2 3 5 --cf

echo
echo "* sgd solution θ and mean square error for n=2,3,5"
./lr.py -i dataset1.mat --poly 2 3 5 --gd --err

echo
echo
echo "--= error for n=5 is horrible, because the train set does not cover x < -4 =--"
echo "            let's look at the data"

./lr.py -i dataset1.mat --poly 5 --gd-rate-decay 0.995 --cf --gd --no-print-res  --plot

echo
echo "--= larger minibatches gives us faster execution time =--"
echo "        no effect on error rate or convergence rate"
echo

# We set precision to 0, to force all of those iterations to take the same number of epochs
./lr.py -i dataset1.mat --poly 5 --gd --gd-rate-decay 0.995 --batch 1 2 5 10 20 30 60 90 --gd-precision 0 --gd-limit 1000 --err --plot
