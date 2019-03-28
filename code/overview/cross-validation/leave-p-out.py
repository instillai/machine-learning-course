# Example of LOOCV and LPOCV splitting

import numpy
from sklearn.model_selection import LeaveOneOut, LeavePOut


def print_result(method):
    """
    Prints the result of either a LPOCV or LOOCV operation

    Args:
        method (LeaveOneOut|LeavePOut): The method to perform
    """
    # Perform the split of either LOOCV or LPOCV
    for train, test in method.split(data):
        output_train = ''
        output_test = ''

        # Build our output for display from the resulting split
        for i in train:
            output_train = "{}({}: {}) ".format(output_train, i, data[i])

        for i in test:
            output_test = "{}({}: {}) ".format(output_test, i, data[i])
            
        print("Train: {}\tTest: {}".format(output_train, output_test))

    print("") # prints a newline

# Create some data to split with
data = numpy.array([[1, 2], [3, 4], [5, 6], [7, 8]])

# Our two methods
loocv = LeaveOneOut()
lpocv = LeavePOut(p=2)

print("Data:\n{}\n".format(data))

print("Leave-One-Out:")
print_result(loocv)

print("Leave-P-Out (where p = 2):")
print_result(lpocv)
