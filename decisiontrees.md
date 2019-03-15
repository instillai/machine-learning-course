# Decision Trees

Decision trees are a classifier in machine learning that allow us to make predictions based on previous data.
They are like a series of sequential "if ... then" statements you feed new data into.

To demonstrate decision trees, let's take a look at an example.
Imagine we want to predict whether Mike is going to go grocery shopping on any given day.
We can look at previous factors that led Mike to go to the store:

![Dataset](/shopping_table.png)

Here we can see the amount of grocery supplies Mike had, the weather, and whether Mike worked each day.
Green rows are days he went to the store, and red days are those he didn't.
The goal of a decision tree is to try to understand *why* Mike goes to the store, and apply that to new data later on.

Let's divide the first attribute up into a tree.
Mike can either have a low, medium, or high amount of supplies:

![Tree 1](/decision_tree_1.png)

Here we can see that Mike never goes to the store if he has a high amount of supplies.
This is called a **pure subset**, a subset with only positive or only negative examples.
With decision trees, there is no need to break a pure subset down further.

Let's break the Med Supplies category into whether Mike worked that day:

![Tree 2](/decision_tree_2.png)

Here we can see we have two more pure subsets, so this tree is complete.
We can replace any pure subsets with their respective answer - in this case, yes or no.

Finally, let's split the Low Supplies category by the Weather attribute:

![Tree 3](/decision_tree_3.png)

Now that we have all pure subsets, we can create our final decision tree:

![Tree 4](/decision_tree_4.png)

## Classification Trees

TODO

## Regression Trees

TODO

## Splitting

TODO

### Recursive Binary

TODO

### Cost of Splitting

TODO

## Pruning

TODO

### Reduced Error

TODO

### Complexity / Weakest Link

