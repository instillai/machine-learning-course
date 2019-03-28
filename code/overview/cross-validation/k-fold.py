# An example of K-Fold Cross Validation split

import numpy
from sklearn.model_selection import KFold

# Create some data to perform K-Fold CV on
x = numpy.array([[1, 2], [3, 4], [5, 6], [7, 8]])
y = numpy.array([1, 2, 3, 4])

# Perform a K-Fold split
kfold = KFold(n_splits=3)
kfold.get_n_splits(x)

#Print split results
print('X Data:')
print(x)

print('Y Data:')
print(y)

print('Training splits:')
for train_data, test_data in kfold.split(x):
    print("Training data: {}\tTest data: {}".format(train_data, test_data))
