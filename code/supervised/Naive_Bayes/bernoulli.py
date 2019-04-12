import numpy as np
from sklearn.naive_bayes import BernoulliNB

# The features in X are broken down as follows:
# [Walks like a duck, Talks like a duck, Is small]
#
# Walks like a duck: 0 = False, 1 = True
# Talks like a duck: 0 = False, 1 = True
# Is small: 0 = False, 1 = True

# Some data is created to train with
X = np.array([[1, 1, 0], [0, 0, 1], [1, 0, 0]])
# These are our target values (Classes: Duck or Not a duck)
y = np.array(['Duck', 'Not a Duck', 'Not a Duck'])

# This is the code we need for the Bernoulli model
clf = BernoulliNB()
# We train the model on our data
clf.fit(X, y)

# Now we can make a prediction on what class new data belongs to
print("Our data set represents things that are and aren't ducks.\n")
print("We have trained a Bernoulli model on our data set.\n")
print(("Let's consider a new input that:\n"
       "   Walks like a duck\n"
       "   Talks like a duck\n"
       "   Is large\n"))
print("What does our model think this should be?")
print("Answer: %s!" % clf.predict([[1, 1, 1]])[0])
