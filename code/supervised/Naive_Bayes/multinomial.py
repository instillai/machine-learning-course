import numpy as np
from sklearn.naive_bayes import MultinomialNB

# The features in X are broken down as follows:
# [Size, Weight, Color]
#
# Size: 0 = Small, 1 = Moderate, 2 = Large
# Weight: 0 = Light, 1 = Moderate, 2 = Heavy
# Color: 0 = Red, 1 = Blue, 2 = Brown

# Some data is created to train with
X = np.array([[1, 1, 0], [0, 0, 1], [2, 2, 2]])
# These are our target values (Classes: Apple, Blueberry, or Coconut)
y = np.array(['Apple', 'Blueberry', 'Coconut'])

# This is the code we need for the Multinomial model
clf = MultinomialNB()
# We train the model on our data
clf.fit(X, y)

# Now we can make a prediction on what class new data belongs to
print("Our data set represents fruits and their characteristics.\n")
print("We have trained a Multinomial model on our data set.\n")
print("Let's consider a new input that is moderately sized, heavy, and red.\n")
print("What fruit does our model think this should be?")
print("Answer: %s!" % clf.predict([[1, 2, 0]])[0])
