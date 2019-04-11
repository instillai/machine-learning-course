import numpy as np
from sklearn.naive_bayes import GaussianNB

# The features in X are broken down as follows:
# [Red %, Green %, Blue %]

# Some data is created to train with
X = np.array([[.5, 0, .5], [1, 1, 0], [0, 0, 0]])
# These are our target values (Classes: Purple, Yellow, or Black)
y = np.array(['Purple', 'Yellow', 'Black'])

# This is the code we need for the Gaussian model
clf = GaussianNB()
# We train the model on our data
clf.fit(X, y)

# Now we can make a prediction on what class new data belongs to
print("Our data set represents RGB triples and their associated colors.\n")
print("We have trained a Gaussian model on our data set.\n")
print("Let's consider a new input with 100% red, 0% green, and 100% blue.\n")
print("What color does our model think this should be?")
print("Answer: %s!" % clf.predict([[1, 0, 1]])[0])
