#!/bin/sh


ridge="-1 1 0.01"

echo "--== Searching for good ridge regression parameters between -1 and 1 ==--"
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
