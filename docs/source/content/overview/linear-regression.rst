#################
Linear Regression
#################

.. contents::
  :local:
  :depth: 3

********
Overview
********
**Linear regression** is a technique used to analyze a **linear relationship** between **input/independent/x** variables and a single **output/dependent/y** variable. A **linear relationship** means that the data points tend to follow a straight line. **Simple linear regression** involves only a single input variable.

.. image:: https://github.com/machinelearningmindset/machine-learning-for-everybody/blob/master/docs/source/content/overview/_img/LR.png

Our goal is to find the line that best models the path of the data points called a line of best fit. The equation for this line looks like this: :math:`y=a_0+a_1x`.

.. image:: https://github.com/machinelearningmindset/machine-learning-for-everybody/blob/master/docs/source/content/overview/_img/LR_LOBF.png

Let’s break it down. We already know that x is the input value and y is our predicted output. :math:`a_0` and :math:`a_1` describe the shape of our line. :math:`a_0` is called the **bias** and :math:`a_1` is called a **weight**. Changing :math:`a_0` will move the line up or down on the plot and changing :math:`a_1` changes the slope of the line. Linear regression helps us pick appropriate values for :math:`a_0` and :math:`a_1`.

Note that we could have more than one input variable. In this case, we call it **multiple linear regression**. Let’s add another input variable called z.

.. image:: https://github.com/machinelearningmindset/machine-learning-for-everybody/blob/master/docs/source/content/overview/_img/MLR.png

Then the equation changes to :math:`y=a_0+a_1x+a_2z` which is a plane.

.. image:: https://github.com/machinelearningmindset/machine-learning-for-everybody/blob/master/docs/source/content/overview/_img/MLR_POBF.png

Adding extra input variables just means that we’ll need to find more weights. For this exercise, we will only consider a simple linear regression.

***********
When to Use
***********
Linear regression is a useful technique but isn’t always the right choice for your data. Linear regression is a good choice when there is a linear relationship between your independent and dependent variables and you are trying to predict continuous values.

.. image:: https://github.com/machinelearningmindset/machine-learning-for-everybody/blob/master/docs/source/content/overview/_img/LR.png

It is not a good choice when the relationship between independent and dependent variables is more complicated or when outputs are discrete values.

.. image:: https://github.com/machinelearningmindset/machine-learning-for-everybody/blob/master/docs/source/content/overview/_img/Not_Linear.png

It is worth noting that sometimes you can apply transformations to data so that it appears to be linear. For example, you could apply a logarithm to exponential data. Then you can use linear regression on the transformed data.

Here is an example of data that does not look linear.

.. image:: https://github.com/machinelearningmindset/machine-learning-for-everybody/blob/master/docs/source/content/overview/_img/Exponential.png

Here is the same data after transforming the output variable.

.. image:: https://github.com/machinelearningmindset/machine-learning-for-everybody/blob/master/docs/source/content/overview/_img/Exponential_Transformed.png

*************
Cost Function
*************
Once we have a prediction, we need some way to tell if it’s reasonable. A **cost function** helps us do this. The cost function compares all the predictions against their actual values and provides us with a single number that we can use to score the prediction function.

.. image:: https://github.com/machinelearningmindset/machine-learning-for-everybody/blob/master/docs/source/content/overview/_img/Cost.png

Two common terms that appear in cost functions are the **error** and **squared error**. The error is how far away from the actual value our prediction is and looks like this: :math:`E=(prediction – actual)`. Squaring this value gives us a useful expression for the general error distance: :math:`E^{2}=(prediction – actual)^{2}`. We know an error of 2 above the actual value and an error of 2 below the actual value should be about as bad as each other. The squared error makes this clear because :math:`2^{2}` and :math:`(-2)^{2}` are both 4. We will use the Mean Squared Error (MSE) function as our cost function. This function finds the average squared error value for all of our data points. It looks like this:

.. image:: https://github.com/machinelearningmindset/machine-learning-for-everybody/blob/master/docs/source/content/overview/_img/MSE_Function.png

Cost functions are important in machine learning because it’s the main way to measure success.  If we craft a model with a low cost, our program can make small changes, knowing that it is close to the best solution.  If the result of the cost function is high, the program can make a larger change to the model to more quickly approach the best solution.  Get comfortable with this idea, since you’ll see it show up in most of the modules for this course.

# *******
# Methods
# *******

# ======================
# Ordinary Least Squares
# ======================
