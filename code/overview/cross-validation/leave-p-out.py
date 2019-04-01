# Example of LOOCV and LPOCV splitting

import numpy
from sklearn.model_selection import LeaveOneOut, LeavePOut

# Configurable constants
P_VAL = 2


def print_result(split_data):
    """
    Prints the result of either a LPOCV or LOOCV operation

    Args:
        split_data: The resulting (train, test) split data
    """
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


# Create some data to split with
data = numpy.array([[1, 2], [3, 4], [5, 6], [7, 8]])

# Our two methods
loocv = LeaveOneOut()
lpocv = LeavePOut(p=P_VAL)

split_loocv = loocv.split(data)
split_lpocv = lpocv.split(data)

print("""\
The Leave-P-Out method works by using every combination of P points as test data.

The following output shows the result of splitting some sample data by Leave-One-Out and Leave-P-Out methods.
A bar displaying the current train-test split as well as the actual data points are displayed for each split.
In the bar, "-" is a training point and "T" is a test point.
""")

print("Data:\n{}\n".format(data))

print("Leave-One-Out:\n")
print_result(split_loocv)

print("Leave-P-Out (where p = {}):\n".format(P_VAL))
print_result(split_lpocv)
