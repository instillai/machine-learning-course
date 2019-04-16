import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import numpy as np

# Create a data set for analysis
x, y = make_regression(n_samples=100, n_features = 1, noise=15, random_state=0)
y = y ** 2

# Pipeline lets us set the steps for our modeling
# We are using a simple linear model here
model = Pipeline([('poly', PolynomialFeatures(degree=1)), \
('linear', LinearRegression(fit_intercept=False))])

# Now we train on our data
model = model.fit(x, y)
# Now we pridict
y_predictions = model.predict(x)

# Plot data
sns.set_style("darkgrid")
plt.plot(x, y_predictions, color='black')
plt.scatter(x, y, marker='o')
plt.xticks(())
plt.yticks(())
plt.tight_layout()
plt.show()
