import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
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

# Plot data
sns.set_style("darkgrid")
plt.scatter(X, y, marker='o')
plt.xticks(())
plt.yticks(())
plt.tight_layout()
plt.show()
