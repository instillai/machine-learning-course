Cross-Validation
================

It's easy to train a model against a particular dataset, but how does
this model perform when introduced with new data? How do you know which
machine learning model to use? Cross-validation answers these questions
by assuring a model is producing accurate results and comparing those
results against other models. Cross-validation goes beyond regular
validation, the process of analyzing how a model does on its own
training data, by evaluating how a model does on *new* data. Several
different methods of cross-validation are discussed in the following
sections:

Holdout Method
--------------

The holdout cross-validation method involves removing a certain portion
of the training data and using it as test data. The model is first
trained against the training set, then asked to predict output from the
testing set. This is the simplest form of cross-validation techniques,
and is useful if you have a large amount of data or need to implement
validation quickly and easily.

.. figure:: https://github.com/machinelearningmindset/machine-learning-for-everybody/blob/master/docs/source/content/overview/_img/holdout.png
   :scale: 50 %
   :alt: holdout method


Typically the holdout method involves splitting a dataset into 20-30%
test data and the rest as training data. These numbers can vary - a
larger percentage of test data will make your model more prone to errors
as it has less training experience, while a smaller percentage of test
data may give your model an unwanted bias towards the training data.
This lack of training or bias can lead to
[Underfitting/Overfitting](TODO: Link to under/overfitting) of our
model.


K-Fold Cross Validation
-----------------------

K-Fold Cross Validation helps remove these biases from your model by
repeating the holdout method on k subsets of your dataset. With K-Fold
Cross Validation, a dataset is broken up into several unique folds of
test and training data. The holdout method is performed using each
combination of data, and the results are averaged to find a total error
estimation.

.. figure:: https://github.com/machinelearningmindset/machine-learning-for-everybody/blob/master/docs/source/content/overview/_img/kfold.png
   :scale: 50 %
   :alt: kfold method

A "fold" here is a unique section of test data. For instance, if you
have 100 data points and use 10 folds, each fold contains 10 test
points. K-Fold Cross Validation is important because it allows you to
use your complete dataset for both training and testing. It's especially
useful when evaluating a model using small or limited datasets.

.. _leave-p-out--leave-one-out-cross-validation:

Leave-P-Out / Leave-One-Out Cross Validation
--------------------------------------------

Leave-P-Out Cross Validation (LPOCV) tests a model by using every
possible combination of P test data points on a model. As a simple
example, if you have 3 data points and use 2 test points, the model will
be trained and tested as follows:

::

   [P][P][T]
   [P][T][P]
   [T][P][P]

Where P is a test point, and T is a training point. Below is another
visualization of LPOCV:

.. figure:: https://github.com/machinelearningmindset/machine-learning-for-everybody/blob/master/docs/source/content/overview/_img/LPOCV.png
   :scale: 50 %
   :alt: kfold method

   Ref: http://www.ebc.cat/2017/01/31/cross-validation-strategies/

LPOCV can provide an extremely accurate error estimation, but can
quickly become exhaustive for large datasets. The amount of testing
iterations a model has to go through using LPOCV can be calculated using
a mathematical `combination`_ n C P, with n being our total number of
data points. We can see, for instance, that a LPOCV run using a dataset
of 10 points with 3 test points would require 10 C 3 = 120 iterations.

Because of this, Leave-One-Out Cross Validation (LOOCV) is a commonly
used cross-validation method. It is just a subset of LPOCV, with P being
1. This allows us to evaluate a model in the same number of steps as
there are data points. LOOCV can also be seen as K-Fold Cross
Validation, where the number of folds is equal to the number of data
points.

.. figure:: https://github.com/machinelearningmindset/machine-learning-for-everybody/blob/master/docs/source/content/overview/_img/LOOCV.png
   :scale: 50 %
   :alt: kfold method

   Ref: http://www.ebc.cat/2017/01/31/cross-validation-strategies/


Similar to K-Fold Cross Validation, LPOCV and LOOCV train a model using
the full dataset. They are particularly useful when you're working with
a small dataset, but incur performance tradeoffs.

.. _combination: https://en.wikipedia.org/wiki/Combination

.. |LPOCV| image:: http://www.ebc.cat/wp-content/uploads/2017/01/leave_p_out.png
.. |LOOCV| image:: http://www.ebc.cat/wp-content/uploads/2017/01/leave_one_out.png
