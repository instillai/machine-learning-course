====================
k-Nearest Neighbors
====================

K-Nearest Neighbors (KNN) is a basic classifier for machine learning. So we are trying to identify what class an object is in. To do this we look at the closest points (neighbors) to the object and the class with the majority of neighbors will be the class that we identify the object to be in. The k is the number of nearest neighbors to the object. So if k = 1 then the class the object would be in is the class of the closest neighbor. Let's look at an example. 

.. figure:: https://github.com/machinelearningmindset/machine-learning-for-everybody/blob/master/docs/source/content/overview/_img/knn.png
   :scale: 50 %
   :alt: KNN

   Ref: https://coxdocs.org

So in this example we are trying to classify the red star to be either a green square or a blue octagon. So first if we look at the inner circle where k = 3, we can see that there are 2 blue octagons and 1 green square. So there is a majority of blue octagons, so the red star would be classified as a blue octagon. Now we take a look at k = 5, the outer circle. In this one there is 2 blue octagons and 3 green squares. So the red star would be classified as a green square.

How does it work?
-----------------

We will look at two different ways to go about this. The two ways we will look at is the brute force method and the K-D tree method.

Brute Force Method
--------------------

This is the simpliest method. Basically it's just calculating the euclidean distance from the object being classified to each point in the set. You want to use this method when the dimensions are small or the number of points are small. 

K-D Tree Method
-----------------

This method tries to improve the running time by reducing the amount of times we calculate the euclidean distance. The idea behind this method is that if we know that two data points are close to each other and we calculate the euclidean distance to one of them and then we know that distance is roughly close to the other point. If you have a larger data set it is better to use this method. 

Choosing k
-----------

Choosing k typically depends on the dataset you are looking at. You never want to choose k = 2 because it has a very high chance that there won't be a majority class, so in the example above the there would be one of each so we wouldn't be able to classify the red star. Typically k you want the value of k to be small. As k goes to infinity all unidentified data points will always be classified to one class or the other depending on which class has more data points. So typically you don't want this to happen, so it is wise to choose a k that is relatively small. 




