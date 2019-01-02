# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

# importing alll the necessary packages to use the various classification algorithms
from sklearn.linear_model import LogisticRegression  # for Logistic Regression algorithm
from sklearn.cross_validation import train_test_split #to split the dataset for training and testing
from sklearn.neighbors import KNeighborsClassifier  # for K nearest neighbours
from sklearn import svm  #for Support Vector Machine (SVM) Algorithm
from sklearn import metrics #for checking the model accuracy
import seaborn as sns
#importing the dataset
dataset = pd.read_csv('train.csv')
dataset.drop('id',axis=1,inplace=True) 
df = pd.read_csv('test.csv')
df.drop('id',axis=1,inplace=True)

X = dataset.iloc[: , :-1].values
y = dataset.iloc[ :, 7].values


#visualization
g = sns.pairplot(dataset, hue='category', markers='+')
plt.show()
g = sns.violinplot(y='category', x='w1', data=dataset, inner='quartile')
plt.show()
g = sns.violinplot(y='category', x='w2', data=dataset, inner='quartile')
plt.show()
g = sns.violinplot(y='category', x='w3', data=dataset, inner='quartile')
plt.show()
g = sns.violinplot(y='category', x='w4', data=dataset, inner='quartile')
plt.show()

#split the dataset into test and train set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


#X_test = df.iloc[: , :].values

#MLPC Algorithm
clf = MLPClassifier(hidden_layer_sizes=(50,50,50,50), max_iter=50, alpha=0.0001,solver='sgd', verbose=10,
                                                random_state=0,tol=0.000000001)

clf.fit(X_train, y_train)

#predicting the test set
y_pred = clf.predict(X_test)

#accuracy
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)

np.savetxt('y_pred.csv',y_pred ,delimiter=',')
