# An example of the Holdout Cross-Validation split

import pandas
from sklearn import datasets
from sklearn.model_selection import train_test_split

# The percentage (as a decimal) of our data that will be training data
TRAIN_SPLIT = 0.7

# The diabetes dataset contains the following columns:
columns = [
  'age',
  'sex',
  'bmi',
  'map',
  'tc',
  'ldl',
  'hdl',
  'tch',
  'ltg',
  'glu'
]

# Load the diabetes dataset
dataset = datasets.load_diabetes()

# Create a pandas DataFrame from the diabetes dataset
dataframe = pandas.DataFrame(dataset.data, columns=columns)

# Split via the holdout method
x_train, x_test, y_train, y_test = train_test_split(
  dataframe, dataset.target, train_size=TRAIN_SPLIT, test_size=1-TRAIN_SPLIT)

# Print our test and training data
print("X Test Data:")
print(x_test)

print("\n\n")

print("X Training Data:")
print(x_train)
