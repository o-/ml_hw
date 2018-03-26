# Solutions to Question 7 & 8

The solution to (7) Logistic Regression is in the folder `lg`, the (8) SVM Implementation in `smo`.

## Logistic Regression

### Question A

See `lg/lg.py`. The script expects the path of the dataset as argument.

It outputs the learned parameters of the regression model and the confusion matrix for test and train data points.
Additionally, it plots the datapoints (using different colors for train and test data) and the classification boundary to a png.

### Question B

Obtained by running `./lg/lg.py data1.mat`.

Learned the parameters θ=[ 31.29378289  -9.54228945] in 98.07ms.
Test and training data are both separated without any classification error.
The plot is in `lg/data1.png`, train samples are purple/yellow, test samples are green/red.

### Question C

Obtained by running `./lg/lg.py data2.mat`.
Learned the parameters θ=[ 7.45795584  0.05800465] in 39.25ms.
Test data are separated without any classification error.
In the train dataset there are 3 wrongly classified yellow samples and 2 wrongly labeled purple ones.
The plot is in `lg/data2.png`, train samples are purple/yellow, test samples are green/red.

## SVM

The SVM implementation is in `smo/smo.py`. The arguments for the script are: `./smo.py data.mat start end step`, where `start`, `end`, `step` define the list of regularization parameters `C` that should be tried.

For every `C`, we use the train samples to train a linear SVM model using sklearn, and a second model using our SMO implementation.
For both of those models and every `C`, we report classification error (combined positive negative and negative positive) for the train and test data.

### Question A

We tried values of `C` from 0.05 to 1.45, with a step size of 0.05.

Regardless of `C`, both our implementation and sklearn, did not have any classification errors.
See `smo/data1.log` for the detailed results.

### Question B

We tried values of `C` from 0.05 to 1.45, with a step size of 0.05.

For most `C`, both our implementation and sklearn achieve a combined classification error of 3.17% on the train data and 7.14% on the test data.
For any `C`, the classification error of sklearn is 3.17% on the train data and exactly 7.14% on the test data.
For any `C`, the classification error of our implementation is 3.23% +/- 0.8% on the train data and exactly 7.14% on the test data.
See `smo/data2.log` for the detailed results.

As expected, the regularization term does not have an influence on the classification.

For very small values of `C` (C <= 0.001) the SMO algorithm can get unstable.
