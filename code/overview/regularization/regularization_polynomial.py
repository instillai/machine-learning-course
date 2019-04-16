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
# We are using a polynomial model here (polynomial with degree 10)
model = Pipeline([('poly', PolynomialFeatures(degree=10)), \
('linear', LinearRegression(fit_intercept=False))])

# Now we train on our data
model = model.fit(x, y)
# Now we pridict
# The next two lines are used to model input for our prediction graph
x_plot = np.linspace(min(x)[0], max(x)[0], 100)
x_plot = x_plot[:, np.newaxis]
y_predictions = model.predict(x_plot)

# Plot data
sns.set_style("darkgrid")
plt.plot(x_plot, y_predictions, color='black')
plt.scatter(x, y, marker='o')
plt.xticks(())
plt.yticks(())
plt.tight_layout()
plt.show()
