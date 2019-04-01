# An example of the Holdout Cross-Validation split

import pandas
from sklearn import datasets
from sklearn.model_selection import train_test_split

# The percentage (as a decimal) of our data that will be training data
TRAIN_SPLIT = 0.7

# The diabetes dataset contains the following columns:
columns = [
    'age', 'sex', 'bmi', 'map', 'tc', 'ldl', 'hdl', 'tch', 'ltg', 'glu'
]

# Load the diabetes dataset
dataset = datasets.load_diabetes()

# Create a pandas DataFrame from the diabetes dataset
dataframe = pandas.DataFrame(dataset.data, columns=columns)

# Split via the holdout method
x_train, x_test, y_train, y_test = train_test_split(
    dataframe, dataset.target, train_size=TRAIN_SPLIT, test_size=1-TRAIN_SPLIT)

print("""\
The holdout method removes a certain portion of the training data and uses it as test data.
Ideally, the data points removed are random on each run.

The following output shows a set of sample diabetes data split into test and training data:
""")

# Print our test and training data
print("Total diabetes data points: {}".format(len(dataframe.index)))
print("# of training data points: {} (~{}%)".format(len(x_train), TRAIN_SPLIT*100))
print("# of test data points: {} (~{}%)\n".format(len(x_test), (1-TRAIN_SPLIT)*100))

print("If you'd like to see the actual data points, uncomment the print statements at the bottom of this script.")
# print("Training data:\n{}\n".format(x_train))
# print("Test data:\n{}".format(x_test))
