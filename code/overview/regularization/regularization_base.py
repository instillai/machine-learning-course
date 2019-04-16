import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.datasets import make_regression
import numpy as np

# Create a data set for analysis
x, y = make_regression(n_samples=100, n_features = 1, noise=15, random_state=0)
y = y ** 2

# Plot data
sns.set_style("darkgrid")
plt.scatter(x, y, marker='o')
plt.xticks(())
plt.yticks(())
plt.tight_layout()
plt.show()
