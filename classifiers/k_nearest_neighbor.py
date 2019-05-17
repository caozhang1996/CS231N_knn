#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 20:05:14 2018

@author: caozhang
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

class K_Nearest_Neighbor(object):
    """
    a kNN classifier with L2 distance
    """
    def __init__(self):
        pass
    
    
    def train(self, X, Y):
        """
         Train the classifier. For k-nearest neighbors this is just 
         memorizing the training data.

         Inputs:
            - X: A numpy array of shape (num_train, D) containing the training data
            consisting of num_train samples each of dimension D.
            - y: A numpy array of shape (N,) containing the training labels, where
            y[i] is the label for X[i].
        """
        self.X_train = X
        self.Y_train = Y
      
        
    def distances_with_two_loops(self, X):
        """
         Compute the distance between each test point in X and each training point
         in self.X_train using a nested loop over both the training data and the 
         test data.

         Inputs:
            - X: A numpy array of shape (num_test, D) containing test data.

         Returns:
             - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
             is the Euclidean distance between the ith test point and the jth training
             point.
        """
        num_train = self.X_train.shape[0]
        num_test = X.shape[0]
        dists = np.zeros((num_test, num_train))
        
        for i in xrange(num_test):
            for j in xrange(num_train):
                dists[i, j] = np.sqrt(np.sum(np.square(X[i, :] - self.X_train[j, :])))
        
        return dists
    
    
    def distances_with_one_loop(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a single loop over the test data.

        Input / Output: Same as compute_distances_two_loops
        """
        num_train = self.X_train.shape[0]
        num_test = X.shape[0]
        dists = np.zeros([num_test, num_train])
        
        for i in xrange(num_test):
            dists[i, :] = np.sqrt(np.sum(np.square(X[i, :] - self.X_train), axis=1))  # 第二维求和
            
        return dists
    
    
     # 完全向量化方式运行
    def distances_with_no_loops(self, X):
        """
         Compute the distance between each test point in X and each training point
         in self.X_train using no explicit loops.

         Input / Output: Same as compute_distances_two_loops
        """
        num_train = self.X_train.shape[0]
        num_test = X.shape[0]
        dists = np.zeros([num_test, num_train])
        
        dists = np.multiply(np.dot(X, self.X_train.T), -2)
        sq1 = np.sum(np.square(X), axis=1, keepdims=True)    # keepdims保证矩阵的二维性, 得到列向量 
        sq2 = np.sum(np.square(self.X_train), axis=1)
        dists = np.add(dists, sq1)
        dists = np.add(dists, sq2)
        dists = np.sqrt(dists)
        
        return dists
    
    
    def predict_labels(self, dists, k=1):
        """
         Given a matrix of distances between test points and training points,
         predict a label for each test point.
    
         Inputs:
         - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
           gives the distance betwen the ith test point and the jth training point.
    
         Returns:
         - y: A numpy array of shape (num_test,) containing predicted labels for the
           test data, where y[i] is the predicted label for the test point X[i].  
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in xrange(num_test):
            # A list of length k storing the labels of the k nearest neighbors to
            # the ith test point.
            closest_y = []
            # Use the distance matrix to find the k nearest neighbors of the ith    
            # testing point, and use self.Y_train to find the labels of these       
            # neighbors. Store these labels in closest_y.
            # Hint: Look up the function numpy.argsort.
            closest_y  = self.Y_train[np.argsort(dists[i, :])[:k]]    # argsort函数返回的是数组值从小到大的索引值
            y_pred[i] = np.argmax(np.bincount(closest_y))             # np.bincount(): 统计closest_y列表中各个数出现的次数
            
        return y_pred
        
    
    def predict(self, X, k=1, num_loops=0):
        if num_loops == 0:
            dists = self.distances_with_no_loops(X)
        elif num_loops == 1:
            dists = self.distances_with_one_loop(X)
        elif num_loops == 2:
            dists = self.distances_with_two_loops(X)
        else:
            raise ValueError('Invalid value %d for num_loops' % num_loops)
            
        return self.predict_labels(dists, k=k)
    