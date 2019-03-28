#All the libraries we need for linear SVM
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
#This is used for our dataset
from sklearn.datasets import make_blobs


# =============================================================================
# We are using sklearn datasets to create the set of data point that is separable 
# n_samples is the number of data points. 
# centers is the number of cluster of data points so in our case it is 2.
# There are more optional parameters but we will leave them as default, 
# You can find them at https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html
# =============================================================================
X_train, Y_train = make_blobs(n_samples=50, centers=2)

# =============================================================================
# Creates the linear svm model and fits it to our data points
# The optional parameter will be default other than these two,
# You can find the other parameters at https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
# =============================================================================
model = svm.SVC(kernel = 'linear', C = 10000)
model.fit(X_train, Y_train)

#plots the points 
plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_train, s=30, cmap=plt.cm.prism)

# Creates the axis for the grid
axis = plt.gca()
x_limit = axis.get_xlim()
y_limit = axis.get_ylim()

# Creates a grid to evaluate model
x = np.linspace(x_limit[0], x_limit[1], 50)
y = np.linspace(y_limit[0], y_limit[1], 50)
X, Y = np.meshgrid(x, y)
xy = np.c_[X.ravel(), Y.ravel()]

#Creates the decision line for the data points, use model.predict if you are classifying more than two 
decision_line = model.decision_function(xy).reshape(Y.shape)


# Plot the decision line and the margins
axis.contour(X, Y,  decision_line, colors = 'k',  levels=[-1, 0, 1], alpha=0.5,
           linestyles=[':', '-', ':'])
# Shows the support vectors that determine the desision line
axis.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none', edgecolors='k')

#Shows the graph
plt.show()