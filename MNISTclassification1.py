#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 12:54:33 2024

@author: mehdy
"""

import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn import datasets
from sklearn import model_selection
X, y = datasets.fetch_openml('mnist_784', version=1, return_X_y=True)
X = np.asarray(X)
Y = np.asarray(y)
print('X shape', X.shape)
print('Y shape', y.shape)

def show_digit(x_, y_):
    X_reshape = x_.reshape(28, 28) # reshape it to have 28*28
    plt.imshow(X_reshape, 'gray')
    plt.title('Label is: ' + y_)
    plt.show()

show_digit(X[0],y[0])

test_percentage = 0.2
X_train, X_test, y_train, y_test = model_selection.train_test_split(X,Y, test_size= test_percentage, random_state=12) # split the data to have 20% test set and 80% training set
X_train = X_train / np.max(X_train)  # Normalize X_train based on the maximum value of X_train
X_test = X_test / np.max(X_train) #  Normalize X_test based on the maximum value of X_train

from sklearn.neighbors import KNeighborsClassifier
R_train_KNN = []
R_test_KNN = []
Neighbours = np.arange(1,100,5)

print("start KNN:\n")
for k in Neighbours:
  model = KNeighborsClassifier(n_neighbors=k,weights='distance')  # define KNN model
  model.fit(X_train,y_train)  # fit the data
  y_res_train = model.predict(X_train) # Output for training set
  y_res_test = model.predict(X_test) # Output for test set
  R_train_KNN.append(sklearn.metrics.accuracy_score(y_train, y_res_train))
  R_test_KNN.append(sklearn.metrics.accuracy_score(y_test, y_res_test))
print("KNN finished:\n")
print("start sklearn:\n")

from sklearn.ensemble import RandomForestClassifier
R_train_RF = []
R_test_RF = []
Min_sample = np.arange(5,51,5)
for s in Min_sample:
  model_RF = RandomForestClassifier(min_samples_leaf=s) # define Random forest model
  model_RF.fit(X_train,y_train) # fit the data
  y_res_train = model_RF.predict(X_train) # Output for train
  y_res_test = model_RF.predict(X_test) 
  R_train_RF.append(sklearn.metrics.accuracy_score(y_train, y_res_train))
  R_test_RF.append(sklearn.metrics.accuracy_score(y_test, y_res_test))

#print("Now we start drawing graph of KNN:\n")
plt.figure(figsize=(12, 6))
#plot knn
plt.subplot(1, 2, 1)
plt.plot(Neighbours, R_train_KNN, label='Training Accuracy')
plt.plot(Neighbours, R_test_KNN, label='Testing Accuracy')
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.title('KNN Accuracy with Different Neighbors')
plt.legend()

#plot random forest
#print("Now we start drawing graph of randon forest:\n")
plt.subplot(1, 2, 2)
plt.plot(Min_sample, R_train_RF, label='Training Accuracy')
plt.plot(Min_sample, R_test_RF, label='Testing Accuracy')
plt.xlabel('Min Samples Leaf')
plt.ylabel('Accuracy')
plt.title('Random Forest Accuracy with Different Min Samples Leaf')
plt.legend()

plt.tight_layout()  # 调整子图间距
plt.show()

# Find the index of the highest testing accuracy for KNN
best_index_KNN = np.argmax(R_test_KNN)
# Get the corresponding best number of neighbors
best_neighbour = Neighbours[best_index_KNN]

# Print the best hyperparameter for KNN
print(f'Best number of neighbours for KNN: {best_neighbour}')
print(f'Best testing accuracy for KNN: {R_test_KNN[best_index_KNN]}')

# Find the index of the highest testing accuracy for Random Forest
best_index_RF = np.argmax(R_test_RF)
# Get the corresponding best min_samples_leaf
best_min_samples_leaf = Min_sample[best_index_RF]

# Print the best hyperparameter for Random Forest
print(f'Best min_samples_leaf for Random Forest: {best_min_samples_leaf}')
print(f'Best testing accuracy for Random Forest: {R_test_RF[best_index_RF]}')
