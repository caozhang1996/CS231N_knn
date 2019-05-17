#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 21:49:43 2018

@author: caozhang
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
from random import shuffle


def svm_loss(W, X, Y, reg):
    """
    SVM loss function, using loops
    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.(eg.(3072, 10))
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - Y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength
    """
    dW = np.zeros(W.shape)
    num_train = X.shape[0]
    num_classes = W.shape[1]
    loss = 0.0
    
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[Y[i]]
        for j in range(num_classes):
            if j == Y[i]:
                continue
            margin = scores[j] - correct_class_score + 1
            if margin > 0:
                loss += margin
            dW[: Y[i]] += -X[i, :].T
            dW[:, j] += X[i, :].T 
    loss /= num_train
    loss += reg * np.sum(W * W)   # Add L2 regularization to the loss.(将W数组中的数平方后全部相加)
    
    dW /= num_train
    dW += reg *W
    return loss, dW


    
    
            