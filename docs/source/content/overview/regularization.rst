================
Regularization
================

**Note:** All code for data set creation and modeling is provided.

Consider the following scenario. You are making a peanut butter sandwich and are trying to adjust parameters so that it has the best taste.
You might consider the type of bread, type of peanut butter, or peanut butter to bread ratio in your decision making process. But would you
consider other factors like how warm it is in the room, what you had for breakfast, or what color socks you’re wearing? Probably not as these
things don’t have as much impact on the taste of your sandwich. You would focus more on the first few features for whatever model you end up
developing and avoid paying too much attention to the other ones.

In previous modules, we have seen prediction models trained on some sample set and scored against how close they are to a test set.
We obviously want our models to make predictions that are accurate but can they be too accurate? When we look at a set of data,
there are two main components: the underlying pattern and noise. We only want to match the pattern and not the noise. Consider
the figures below that represent quadratic data.

.. figure:: Regularization_Linear.png
.. figure:: Regularization_Quadratic.png
.. figure:: Regularization_Polynomial.png
