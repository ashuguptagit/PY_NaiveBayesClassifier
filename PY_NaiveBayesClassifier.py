# -*- coding: utf-8 -*-
"""
Created on Tue May 21 12:43:22 2019

@author: asgupta
"""
#Import scikit-learn dataset library
from sklearn import datasets
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

#Load dataset
wine = datasets.load_wine()

# print the names of the 13 features
print("Features: ", wine.feature_names)

# print the label type of wine(class_0, class_1, class_2)
print("Labels: ", wine.target_names)

# print data(feature)shape
wine.data.shape

# print the wine data features (top 5 records)
print(wine.data[0:5])

# print the wine labels (0:Class_0, 1:class_2, 2:class_2)
print(wine.target)

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3,random_state=109)

#Create a Gaussian Classifier
gnb = GaussianNB()

#Train the model using the training sets
gnb = gnb.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = gnb.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

