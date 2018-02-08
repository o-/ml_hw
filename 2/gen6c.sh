#!/bin/sh


ridge="-1 1 0.01"

echo "--== Searching for good ridge regression parameters between -1 and 1 ==--"
echo
for n in 2 3 5; do
  echo "****** n = $n"
  for k in 2 10 100; do
    echo "********* k = $k"
    echo "* Serching for best ridge factors for closed form solution and n=$n and k=$k"
    ./lr.py -i dataset2.mat --poly $n --cf --ridge-seq $ridge --kcross $k --no-print-res
    echo "* Searching for best ridge factors for sgd solution and n=$n and k=$k"
    ./lr.py -i dataset2.mat --poly $n --gd --ridge-seq $ridge --kcross $k --no-print-res
    echo
  done
done

echo
echo "--== How does the test error change with different n ==--"
echo "      A: n == 3 fits best"
echo
./lr.py -i dataset2.mat --poly 2 3 5 --gd --err

echo
echo "--== How does the test error change with λ ==--"
echo "    A: there seems to be a globally optimal value"
echo
echo "****** n = 3"
./lr.py -i dataset2.mat --poly 3 --ridge-seq $ridge --gd --err
echo "****** n = 5"
./lr.py -i dataset2.mat --poly 5 --ridge-seq $ridge --gd --err
