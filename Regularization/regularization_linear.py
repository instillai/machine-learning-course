import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import numpy as np

# We will create some x-values and randomly choose some as data points
X = np.linspace(0, 10, 100)
# We are fixing the random number seed for consistency
rn = np.random.RandomState(0)
# Shuffle the data for variety
rn.shuffle(X)
# Grab the first 30 of our shuffled points and sort them for plotting
X = np.sort(X[:30])
# Our output will be a quadratic function
y = X**2
# We will add some variance to the data so that it's more interesting
y = y + (((np.random.rand(30) * 2) - 1) * 30)

# Pipeline lets us setup a fixed number of steps for our modeling
model = Pipeline([('poly', PolynomialFeatures(degree=1)), \
('linear', LinearRegression(fit_intercept=False))])
# Now we train on our data
model = model.fit(X[:, np.newaxis], y)
# Now we pridict
X_plot = np.linspace(0, 10, 100)
X_plot = X_plot[:, np.newaxis]
y_plot = model.predict(X_plot)

# Plot data
sns.set_style("darkgrid")
plt.plot(X_plot, y_plot, color='black')
plt.scatter(X, y, marker='o')
plt.xticks(())
plt.yticks(())
plt.tight_layout()
plt.show()
