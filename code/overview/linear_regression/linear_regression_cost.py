import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets, linear_model
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Create a data set for analysis
x, y = make_regression(n_samples=500, n_features = 1, noise=25, random_state=0)

# Split the data set into testing and training data
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)

# Create a linear regression object
regression = linear_model.LinearRegression()

# Train the model using the training set
regression.fit(x_train, y_train)

# Make predictions using the testing set
y_predictions = regression.predict(x_test)

# Grab a sample pair of points to analyze cost
point_number = 2
x_sample = [x_test[point_number].item(), x_test[point_number].item()]
y_sample = [y_test[point_number].item(), y_predictions[point_number].item()]

# Plot the data
sns.set_style("darkgrid")
sns.regplot(x_test, y_test, fit_reg=False)
plt.plot(x_test, y_predictions, color='black')
plt.plot(x_sample, y_sample, color='red', label="cost", marker='o')

# Add a legend
n = ['actual value', 'prediction']
for i, txt in enumerate(n):
    plt.annotate(txt, (x_sample[i], y_sample[i]), xytext=(10, -10),
                 textcoords='offset pixels', fontsize=20)
plt.legend(fontsize=20)

# Remove ticks from the plot
plt.xticks([])
plt.yticks([])

plt.tight_layout()
plt.show()
