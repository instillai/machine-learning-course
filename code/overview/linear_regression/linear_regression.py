import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

'''
Based on the linear regression example for scikit-learn by Jaques Grobler.
Available at:
https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html
'''

# Load the diabetes dataset from sklearn
diabetes = datasets.load_diabetes()

# Use only one feature from the dataset
diabetes_X = diabetes.data[:, np.newaxis, 2]

# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
# We reserve 20 data points for testing
diabetes_X_test = diabetes_X[-20:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes.target[:-20]
# We reserve 20 target points for testing
diabetes_y_test = diabetes.target[-20:]

# Create a linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training set
regr.fit(diabetes_X_train, diabetes_y_train)

# Make predictions using the testing set
diabetes_y_pred = regr.predict(diabetes_X_test)

# The coefficients of the model
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(diabetes_y_test, diabetes_y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(diabetes_y_test, diabetes_y_pred))

# Plot outputs
sns.set_style("darkgrid")
sns.regplot(x=diabetes_X_test, y=diabetes_y_test, fit_reg=False)
plt.plot(diabetes_X_test, diabetes_y_pred, color='black')

plt.xticks(())
plt.yticks(())

plt.tight_layout()
plt.show()
