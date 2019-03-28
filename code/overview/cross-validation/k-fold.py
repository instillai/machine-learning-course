# An example of K-Fold Cross Validation split

import numpy
from sklearn.model_selection import KFold

# Configurable constants
NUM_SPLITS = 3


def print_result(kfold):
    """
    Prints the result of a K-Fold split

    Args:
        kfold (KFold): The KFold object from sklearn
    """
    # Perform a KFold split and print the result
    for train, test in kfold.split(data):
        output_train = ''
        output_test = ''

        # Build our output for display from the resulting split
        for i in train:
            output_train = "{}({}: {}) ".format(output_train, i, data[i])

        for i in test:
            output_test = "{}({}: {}) ".format(output_test, i, data[i])

        print("Train: {}\tTest: {}".format(output_train, output_test))

    print("") # prints a newline


# Create some data to perform K-Fold CV on
data = numpy.array([[1, 2], [3, 4], [5, 6], [7, 8]])

# Perform a K-Fold split and print results
kfold = KFold(n_splits=NUM_SPLITS)

print("Data:\n{}\n".format(data))

print('K-Fold split (with n_splits = {}):'.format(NUM_SPLITS))
print_result(kfold)
