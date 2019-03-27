from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
X, y = load_iris(return_X_y=True)
clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial')
clf = clf.fit(X, y)
clf.predict(X[:2, :])
clf.predict_proba(X[:2, :])
#Now, we check the fit of the model
clf.score(X, y)
