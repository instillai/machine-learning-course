from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

#Loads the sample dataset iris from sklearn's datasets
X, y = load_iris(return_X_y=True)

#Creates the Logistic Regression object
clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial')
#Fit the model to match the data
clf = clf.fit(X, y)

#Now, we show the predictions for each row of the dataset.
print(clf.predict(X[:2, :]))
print(clf.predict_proba(X[:2, :]))

#Check the fit of the model.  Score returns the mean accuracy on the data.  
print(clf.score(X, y))
