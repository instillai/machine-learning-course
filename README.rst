
###################################################
Machine Learning for Everybody
###################################################

.. image:: https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat
    :target: https://github.com/pyairesearch/machine-learning-for-everybody/pulls
.. image:: https://badges.frapsoft.com/os/v2/open-source.png?v=103
    :target: https://github.com/ellerbrock/open-source-badge/
.. image:: https://img.shields.io/badge/Made%20with-Python-1f425f.svg
      :target: https://www.python.org/
.. image:: https://img.shields.io/github/contributors/Naereen/StrapDown.js.svg
      :target: https://github.com/pyairesearch/machine-learning-for-everybody/graphs/contributors
.. image:: https://img.shields.io/twitter/follow/machinemindset.svg?label=Follow&style=social
      :target: https://twitter.com/machinemindset



##################
Table of Contents
##################
.. contents::
  :local:
  :depth: 4

***************
Introduction
***************

The purpose of this project is to provide a comperehensive and yet simple course in Machine Learning using Python.


============
Motivation
============

``Machine Learning``, as a tool for ``Artificial Intelligence``, is one of the most widely adopted
scientific fields. A considerable amount of literature has been published on Machine Learning.
The purpose of this project is to provide the most important aspects of ``Machine Learning`` by presenting a
series of simple and yet comprehensive tutorials using ``Python``. In this project, we built our
tutorials using many different well-known Machine Learning frameworks such as ``Scikit-learn``. In this project you will learn:

* What is the definition of Machine Learning?
* When it started and what is the trending evolution?
* What are the Machine Learning categories and sucategories?
* What are the mostly used Machine Learning algorithms and how to implement them?



=====================
Machine Learning
=====================



------------------------------------------------------------
Overview of machine learning
------------------------------------------------------------
.. figure:: _img/deeplearning.png
.. _lrtutorial: docs/source/content/overview/linear-regression.rst
.. _lrcode: code/overview/linear_regression

.. _overtutorial: docs/source/content/overview/overfitting.rst
.. _overcode: code/overview/overfitting

.. _regtutorial: docs/source/content/overview/regularization.rst
.. _regpython: code/overview/regularization

.. _crosstutorial: docs/source/content/overview/crossvalidation.rst
.. _crosspython: code/overview/cross-validation




+--------------------------------------------------------------------+-------------------------------+--------------------------------+
| Title                                                              |    Code                       |    Document                    |
+====================================================================+===============================+================================+
| Linear Regression                                                  |   `Python <lrcode_>`_         | `Tutorial <lrtutorial_>`_      |
+--------------------------------------------------------------------+-------------------------------+--------------------------------+
| overfitting                                                        |  `Python <overcode_>`_        | `Tutorial <overtutorial_>`_    |
+--------------------------------------------------------------------+-------------------------------+--------------------------------+
| regularization                                                     | `Python <regpython_>`_        | `Tutorial <regtutorial_>`_     |
+--------------------------------------------------------------------+-------------------------------+--------------------------------+
| cross-validation                                                   | `Python <crosspython_>`_      | `Tutorial <crosstutorial_>`_   |
+--------------------------------------------------------------------+-------------------------------+--------------------------------+


------------------------------------------------------------
Supervised learning
------------------------------------------------------------

.. figure:: _img/supervised.png

.. _dtdoc: docs/source/content/supervised/decisiontrees.rst
.. _dtcode: code/supervised/DecisionTree/decisiontrees.py

.. _knndoc: docs/source/content/supervised/knn.rst
.. _knncode: code/supervised/KNN/knn.py

.. _nbdoc: docs/source/content/supervised/bayes.rst
.. _nbcode: code/supervised/Naive_Bayes

.. _logisticrdoc: docs/source/content/supervised/logistic_regression.rst
.. _logisticrcode: supervised/Logistic_Regression/logistic_ex1.py


.. _linearsvmdoc: docs/source/content/supervised/linear_SVM.rst
.. _linearsvmcode: code/supervised/Linear_SVM/linear_svm.py



+--------------------------------------------------------------------+-------------------------------+------------------------------+
| Title                                                              |    Code                       |    Document                  |
+====================================================================+===============================+==============================+
| Decision trees                                                     | `Python <dtcode_>`_           | `Tutorial <dtdoc_>`_         |
+--------------------------------------------------------------------+-------------------------------+------------------------------+
| K-Nearest Neighbor                                                 |  `Python <knncode_>`_         | `Tutorial <knndoc_>`_        |
+--------------------------------------------------------------------+-------------------------------+------------------------------+
| Naive Bayes                                                        | `Python <nbcode_>`_           |  `Tutorial <nbdoc_>`_        |
+--------------------------------------------------------------------+-------------------------------+------------------------------+
| Logistic Regression                                                | `Python <logisticrcode_>`_    |  `Tutorial <logisticrdoc_>`_ |
+--------------------------------------------------------------------+-------------------------------+------------------------------+
| Support Vector Machines                                            | `Python <linearsvmcode_>`_    | `Tutorial <linearsvmdoc_>`_  |
+--------------------------------------------------------------------+-------------------------------+------------------------------+




------------------------------------------------------------
Unsupervised learning
------------------------------------------------------------

.. figure:: _img/unsupervised.png

.. _clusteringdoc: docs/source/content/unsupervised/clustering.rst
.. _clusteringcode: code/unsupervised/Clustering

+--------------------------------------------------------------------+-------------------------------+--------------------------------+
| Title                                                              |    Code                       |    Document                    |
+====================================================================+===============================+================================+
| clustering                                                         | `Python <clusteringcode_>`_   | `Tutorial <clusteringdoc_>`_   |
+--------------------------------------------------------------------+-------------------------------+--------------------------------+
| Principal Components Analysis                                      |                               |                                |
+--------------------------------------------------------------------+-------------------------------+--------------------------------+

.. ------------------------------------------------------------
.. Deep Learning
.. ------------------------------------------------------------
..
.. .. _conganpaper: https://arxiv.org/abs/1411.1784
.. .. _congancode: https://github.com/zhangqianhui/Conditional-GAN
..
.. .. _photorealpaper: https://arxiv.org/pdf/1609.04802.pdf
.. .. _photorealcode: https://github.com/tensorlayer/srgan
..
.. .. _im2impaper: https://arxiv.org/abs/1611.07004
.. .. _im2imcode: https://github.com/phillipi/pix2pix
..
.. .. _vismanpaper: https://arxiv.org/abs/1609.03552
.. .. _vismancode: https://github.com/junyanz/iGAN
..
..
..
..
.. +--------------------------------------------------------------------+-------------------------------+---------------------------+
.. | Title                                                              |    Text                       |    Software               |
.. +====================================================================+===============================+===========================+
.. | Neural Networks Overview                                           |                               |                           |
.. +--------------------------------------------------------------------+-------------------------------+---------------------------+
.. | Convolutional Neural Networks                                      |                               |                           |
.. +--------------------------------------------------------------------+-------------------------------+---------------------------+
.. | Recurrent Neural Networks                                          |                               |                           |
.. +--------------------------------------------------------------------+-------------------------------+---------------------------+
.. | Autoencoders                                                       |                               |                           |
.. +--------------------------------------------------------------------+-------------------------------+---------------------------+



========================
Pull Request Process
========================

Please consider the following criterions in order to help us in a better way:

1. The pull request is mainly expected to be a link suggestion.
2. Please make sure your suggested resources are not obsolete or broken.
3. Ensure any install or build dependencies are removed before the end of the layer when doing a
   build and creating a pull request.
4. Add comments with details of changes to the interface, this includes new environment
   variables, exposed ports, useful file locations and container parameters.
5. You may merge the Pull Request in once you have the sign-off of at least one other developer, or if you
   do not have permission to do that, you may request the owner to merge it for you if you believe all checks are passed.

========================
Final Note
========================

We are looking forward to your kind feedback. Please help us to improve this open source project and make our work better.
For contribution, please create a pull request and we will investigate it promptly. Once again, we appreciate
your kind feedback and support.
