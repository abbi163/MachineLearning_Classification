import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import matplotlib.ticker as ticker
from sklearn import preprocessing


df = pd.read_csv('E:\Pythoncode\Coursera\Classification_Algorithms\KNN Algorithms\CustomerCategory/teleCust1000t.csv')
#print(df.head())
#print(df.columns)

# To use scikit-learn library, we have to convert the Pandas data frame to a Numpy array:

X = df[['region', 'tenure', 'age', 'marital', 'address', 'income', 'ed', 'employ', 'retire', 'gender', 'reside']].values

#print(X[0:5])

Y = df[['custcat']]

#print(Y[0:5])

# Normalization Data
# Data Standardization give data zero mean and unit variance, it is good practice, especially for algorithms such as KNN which is based on distance of cases:
# Formulae for Standarized data is : Y_standarized =[(Y_i - mean(Y))/variance(Y)]
# Normalization is done so that any one of the data don't have very high effect once calculating squared Euclidian distance.

X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
# print(X[0:5])

# Splitting the dataset into mutually exclusive train and test dataset.


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size=0.2, random_state=4)
# print ('Train set:', X_train.shape,  Y_train.shape)
# print ('Test set:', X_test.shape,  Y_test.shape)

# K nearest neighbor (KNN)

from sklearn.neighbors import KNeighborsClassifier

k = 4
#Train Model and Predict  
neigh = KNeighborsClassifier(n_neighbors=k).fit(X,Y)

Yhat = neigh.predict(X_test)
# print(yhat[0:5])

# In multilabel classification, accuracy classification score is a function that computes subset accuracy.
# This function is equal to the jaccard_similarity_score function i.e J(A,B) = ( |A∩B| / |A∪B|)
# Essentially,it calculates how closely the actual labels and predicted labels are matched in the test set.
from sklearn import metrics
print("Train set Accuracy: ", metrics.accuracy_score(Y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(Y_test, Yhat))

