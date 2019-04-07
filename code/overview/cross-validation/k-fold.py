# An example of K-Fold Cross Validation split

import numpy
from sklearn.model_selection import KFold

# Configurable constants
NUM_SPLITS = 3

# Create some data to perform K-Fold CV on
data = numpy.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])

# Perform a K-Fold split and print results
kfold = KFold(n_splits=NUM_SPLITS)
split_data = kfold.split(data)

print("""\
The K-Fold method works by splitting off 'folds' of test data until every point has been used for testing.

The following output shows the result of splitting some sample data.
A bar displaying the current train-test split as well as the actual data points are displayed for each split.
In the bar, "-" is a training point and "T" is a test point.
""")

print("Data:\n{}\n".format(data))
print('K-Fold split (with n_splits = {}):\n'.format(NUM_SPLITS))

for train, test in split_data:
    output_train = ''
    output_test = ''

    bar = ["-"] * (len(train) + len(test))

    # Build our output for display from the resulting split
    for i in train:
        output_train = "{}({}: {}) ".format(output_train, i, data[i])

    for i in test:
        bar[i] = "T"
        output_test = "{}({}: {}) ".format(output_test, i, data[i])

    print("[ {} ]".format(" ".join(bar)))
    print("Train: {}".format(output_train))
    print("Test:  {}\n".format(output_test))
