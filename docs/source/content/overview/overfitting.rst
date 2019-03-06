================================
Overfitting and Underfitting
================================

.. contents::
  :local:
  :depth: 3

----------------------------
Overview
----------------------------
When using machine learning, there are many ways to go wrong.  Some of the most common issues in machine learning are **overfitting** and underfitting**.  To understand these concepts, let's imagine a machine learning model that is trying to learn to classify numbers, and has access to a training set of data and a testing set of data.

----------------------------
Overfitting
----------------------------

A model suffers from **Overfitting** when it has learned too much from the training data, and does not perform well in practice as a result.  This is usually caused by the model having too much exposure to the training data.  For the number classification example, if the model is overfit in this way, it may be picking up on tiny details that are misleading, like stray marks as an indication of a specific number.  Think of overfitting as looking like this:

.. figure:: https://github.com/machinelearningmindset/machine-learning-for-everybody/blob/master/docs/source/content/overview/_img/Lagrange_Error.svg.png

The estimate looks pretty good when you look at the middle of the graph, but the edges have large error.  In practice, this error isn't always at edge cases and can pop up anywhere.  The noise in training can cause the error seen in the graph

----------------------------
Underfitting
----------------------------

A model suffers from **Underfitting** when it has not learned enough from the training data, and does not perform well in practice as a result.  As a direct contrast to the previous idea, this issue is caused by not letting the model learn enough from training data.  In the number classification example, if the training set is too small or the model has not had enough attempts to learn from it, then it will not be able to pick out key features of the numbers.  Think of underfitting as looking like this:

.. figure:: https://github.com/machinelearningmindset/machine-learning-for-everybody/blob/master/docs/source/content/overview/_img/Underfit_Graph.PNG

The issue with this estimate is clear to the human eye, the model should be nonlinear, and is instead just a simple line.  In machine learning, this could be a result of underfitting, the model has not had enough exposure to training data to adapt to it, and is currently in a simple state.

----------------------------
How to avoid overfitting
----------------------------
A key idea in avoiding overfitting issues in machine learning is to maintain a **validation set**.  This set is used for training purposes, but most importantly the model **does not learn from it**.  So, the model first learns from the **training set**, then checks its knowledge on a completely different **validation set**.  When it performs well enough on the **validation set**, we can be more confident that it is not overfit to the training data than if we just looked at training results.
