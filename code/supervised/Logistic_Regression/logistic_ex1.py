from sklearn.linear_model import LogisticRegression
import numpy as np
import random

#defines the classification for the training data.
def true_classifier(i):
    if i >= 700:
        return 1
    return 0

#Generate a random dataset which includes random scores from 0 to 1000.
x = np.array([ random.randint(0,1000) for i in range(0,1000) ])

#The model will expect a 2D array, so we must reshape
#For the model, the 2D array must have rows equal to the number of samples,
#and columns equal to the number of features.
#For this example, we have 1000 samples and 1 feature.
x = x.reshape((-1, 1))

#For each point, y is a pass/fail for the grade. The simple threshold is arbitrary,
#and can be changed as you would like. Classes are 1 for success and 0 for failure
y = [ true_classifier(x[i][0]) for i in range(0,1000) ]


#Again, we need a numpy array, so we convert.
y = np.array(y)

#Our goal will be to train a logistic regression model to do pass/fail to the same threshold.
model = LogisticRegression(solver='liblinear')

#The fit method actually fits the model to our training data
model = model.fit(x,y)

#Create 100 random samples to try against our model as test data
samples = [random.randint(0,1000) for i in range(0,100)]
#Once again, we need a 2d Numpy array
samples = np.array(samples)
samples = samples.reshape(-1, 1)

#Now we use our model against the samples.  output is the probability, and _class is the class.
_class = model.predict(samples)
proba = model.predict_proba(samples)

num_accurate = 0

#Finally, output the results, formatted for nicer viewing.
#The format is [<sample value>]: Class <class number>, probability [ <probability for class 0> <probability for class 1>]
#So, the probability array is the probability of failure, followed by the probability of passing.
#In an example run, [7]: Class 0, probability [  9.99966694e-01   3.33062825e-05]
#Means that for value 7, the class is 0 (failure) and the probability of failure is 99.9%
for i in range(0,100):
    if (true_classifier(samples[i])) == (_class[i] == 1):
        num_accurate = num_accurate + 1
    print("" + str(samples[i]) + ": Class " + str(_class[i]) + ", probability " + str(proba[i]))
#skip a line to separate overall result from sample output
print("")
print(str(num_accurate) +" out of 100 correct.")
