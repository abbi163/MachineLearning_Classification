# -*- coding: utf-8 -*-
"""
@author: abbi163
"""
import pandas as pd

df = pd.read_csv('E:\Pythoncode\Coursera\Classification_Algorithms\Decision_Tree\Drug_Effect\drug200.csv', index_col = None) 
# print(df.count())

# Dividing the data into Feature vector X & Response Vector Y

X = df[['Age','Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
Y = df[['Drug']]

# Now data has been stored in the array
# print(X[0:5])
# print(Y[0:5])
# Now , converting categorical data to Numerical Value by labeling them as Sklearn don't handle categorical dataset

# Converting sex, BP, Cholestrol level from Categorical dataset to Numerical Dataset 
from sklearn import preprocessing

le_sex = preprocessing.LabelEncoder()
le_sex.fit(['M','F'])
X[:,1]=le_sex.transform(X[:,1])

le_bp = preprocessing.LabelEncoder()
le_bp.fit(['HIGH','NORMAL','LOW'])
X[:,2]=le_bp.transform(X[:,2])

le_chol = preprocessing.LabelEncoder()
le_chol.fit(['NORMAL','HIGH'])
X[:,3] = le_chol.transform(X[:,3])

#print(X[0:5])

# Splitting data into Test and Train DataSet

from sklearn.model_selection import train_test_split

# train_test_split will return 4 different parameters. We will name them: X_trainset, X_testset, y_trainset, y_testset 
# four parameters for the function train_test_split are X, Y, test_size, random_state
X_trainset, X_testset, Y_trainset, Y_testset = train_test_split(X,Y,test_size = 0.25, random_state = 1)

# print(len(X_trainset))
# print(len(X_testset))

# shape of the X_trainset
#print (X_trainset.shape)
 
from sklearn.tree import DecisionTreeClassifier

drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
drugTree # it shows the default parameters

drugTree.fit(X_trainset,Y_trainset)

# making prediction for the testset

predTree = drugTree.predict(X_testset)

# print (predTree [0:5])
# print (Y_testset [0:5])

# Evaluation of the metrics

from sklearn import metrics
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(Y_testset, predTree))







