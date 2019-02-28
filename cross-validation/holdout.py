# An example of the Holdout Cross-Validation method

import pprint
import pandas
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split

test = datasets.load_boston()

pp = pprint.PrettyPrinter(indent=2)
pp.pprint("this is a test:")
pp.pprint(test)
