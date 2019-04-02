==========================
Linear Support Vector Machines
==========================

A **Support Vector Machine** (SVM for short) is another machine learning algorithm that is used to classify data. The point of SVM's are to try and find a line or **hyperplane** to divide a dimensional space which best classifies the data points. If we were trying to divide two classes A and B we would try best separate the two classes with a line. So on one side of the line/hyperplane would be data from class A and on the otherside would be from class B. This alogorithm is very useful in classifying because we just have to caluclate the best line or hyperplane once and any new data points we can easily classify just by seeing which side of the line it falls on. Unlike KNN, where we would have to calculate each data points nearest neighbors. 

Hyperplane
----------
A **hyperplane** depends on the space it is in, but it divides the space into two disconnected parts. So 1-dimensional space this would just be a point, 2-d space a line, 3-d space a plane, and so on. 

How do we find the best hyperplane/line?
----------------------------------------

You might be wondering that there could be multiple lines that split the data well. In fact there is a infinite amount of lines that can divide two classes.  As you can see in the graph below every line splits the squares and the circles, so which one do we choose?

.. figure:: https://github.com/machinelearningmindset/machine-learning-for-everybody/blob/master/docs/source/content/overview/_img/Possible_hyperplane.PNG
   :scale: 50 %
   :alt: Possible_Hyperplane

   Ref: https://towardsdatascience.com/support-vector-machine-introduction-to-machine-learning-algorithms-934a444fca47 

So how does SVM find the ideal line to separate the two classes. It doesn't just pick a random one. The algorithm the line/hyperplane with the **maximum margin**. Maximizing the margin will give us the optimal line to classify the data. This is shown in the figure below.  

.. figure:: https://github.com/machinelearningmindset/machine-learning-for-everybody/blob/master/docs/source/content/overview/_img/Optimal_hyperplane.PNG
   :scale: 50 %
   :alt: Optimal_Hyperplane

   Ref: https://towardsdatascience.com/support-vector-machine-introduction-to-machine-learning-algorithms-934a444fca47 

How to maximize the margin?
---------------------------

The data that is closest to the line are what determmine the optimal line. These data points are called **support vectors**. They are shown as the filled in squares and circles above. The distance from these vectors to the hyperplane is called the **margin**. In general the further those points are from the hyperplane, the greater the probability of correctly classifying the data. There is a lot of complex math that goes into finding the support vectors and maximizing the margin. We won't go into that, we just want to get the basic idea behind SVMs. 

Ignore Outliers
---------------

Sometimes data classes will have **outliers**. This is data points that are clearly separated from the rest of it's class. Support Vector Machines will ignore these outliers. This is shown in the figure below. 

.. figure:: https://github.com/machinelearningmindset/machine-learning-for-everybody/blob/master/docs/source/content/overview/_img/SVM_Outliers.png
   :scale: 50 %
   :alt: Outliers

   Ref:  https://www.analyticsvidhya.com/blog/2017/09/understaing-support-vector-machine-example-code/

The star that is with the red circles is the outlier. So the SVM ignores the outlier and creates the best line to separate the two classes. 


Kernel SVM
-----------

There will be data classes that can't be separated with a simple line or hyperplane. This is called **non-linearly separable data**. Here is an example of that kind of data. 

.. figure:: https://github.com/machinelearningmindset/machine-learning-for-everybody/blob/master/docs/source/content/overview/_img/SVM_Kernal.png
   :scale: 50 %
   :alt: Outliers

   Ref:  https://www.analyticsvidhya.com/blog/2017/09/understaing-support-vector-machine-example-code/


There's is no clear way to separate the stars from the circles. SVMs will be able to classify non-linearly separable data by using a trick called the **kernel trick**. Basically what goes behind the scenes, is that it takes the points to a higher dimension, which turns non-linearly separable date to linear separable data. So the above figure would be classified with a circle that separates the data. 

There are three types of kernels:

- **Linear** Kernel
- **Polynomial** Kernel
- **Radial Basis Function (RBF)** kernel

You can see how these kernels change the outcome of the optimal hyperplane by changing the value of kernel in "model = svm.SVC(kernel = 'linear', C = 10000)" to either 'poly' or 'rbf'. This is in the linear_svm.py. 


Conclusion
-----------

A SVM is a great machine learning technique to classify data. Now that we know a little about SVM's we can show the advantages and disadvantages to using this classifier. 
The pros to SVM's:
- Effective in classifying higher dimesional space
- Saves space on memory because it only uses the support vectors to create the optimal line. 
- Best classifier when data points are separable

The cons to SVM's:
- Peforms poorly when there is a large data set, the training times are longer.
- Performs badly when the classes are overlapping, i.e. non-separable data points.   

Check out our code to learn how to implement a linear SVM using Python's scikit-learn library. 



