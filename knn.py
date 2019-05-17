#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 20:57:58 2018

@author: caozhang
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

import data_utils
from classifiers.k_nearest_neighbor import K_Nearest_Neighbor

dataset_dir = './cifar-10-batches-py'
X_train, Y_train, X_test, Y_test = data_utils.load_CIFAR10(root_dir=dataset_dir)
print('Train data shape: ', X_train.shape)
print('Train labels shape: ', Y_train.shape)
print('test data shape: ', X_test.shape)
print('test labels shape: ', Y_test.shape)


def show_some_samples(cifar10_dir):
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    num_classes = len(classes)
    samples_per_class = 8
    
    for y, cls in enumerate(classes):
        idxs = np.flatnonzero(Y_train == y)     # 找到标签中y类的位置
        idxs = np.random.choice(idxs, samples_per_class, replace=False)
        for i, idx in enumerate(idxs):  
            plt_idx = i * num_classes + y + 1
            plt.subplot(samples_per_class, num_classes, plt_idx)
            plt.imshow(X_train[idx].astype('uint8'))
            plt.axis('off')                     # 不显示坐标尺寸
            if i == 0:
                plt.title(cls)
    plt.show()    


def knn():
    # for efficiently running data, get the subset of image dataset.
    num_training = 5000
    mask = list(range(num_training))
    X_sub_train = X_train[mask]
    Y_sub_train = Y_train[mask]
    
    num_test = 500
    mask = list(range(num_test))
    X_sub_test = X_test[mask]
    Y_sub_test = Y_test[mask]
    
    #  flatten the image data into 2-D
    X_sub_train = np.reshape(X_sub_train, (X_sub_train.shape[0], -1))
    X_sub_test = np.reshape(X_sub_test, (X_sub_test.shape[0], -1))
    print ('Input shape of train: ', X_sub_train.shape)
    print ('Input shape of test: ', X_sub_test.shape)
    
    # create a instance of K Nearest Neighbor
    classifier = K_Nearest_Neighbor()
    classifier.train(X_sub_train, Y_sub_train)
    
    dists_one = classifier.distances_with_no_loops(X_sub_test)
    dists_two = classifier.distances_with_one_loop(X_sub_test)
    Y_test_pred = classifier.predict_labels(dists=dists_one, k=1)
    # 计算准确率
    num_correct = 0
    for i in range(num_test):
        if Y_test_pred[i] == Y_test[i]:
            num_correct = num_correct + 1
    accuracy = float(num_correct) / num_test 
    print ('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))
    
    
    def compare_dists(dist_1, dist_2):
        difference = np.linalg.norm(dist_1 - dist_2, ord='fro') # ????查函数
        print ('Difference is: %f' % difference)
        if difference < 0.001:
            print ('The distance matrices are the same')
        else:
            print ('The distance matrices are different')
    compare_dists(dists_one, dists_two)
    

def cross_validation():
    """
    Args:
        X: Train data
        Y: Labels of train data
    """
    num_training = 20000             # 原始数据量太大,只能取一部分进行计算
    mask = range(num_training)
    X = X_train[mask]
    Y = Y_train[mask]
    # reshape the original data
    X = np.reshape(X, (X.shape[0], -1))
    num_folds = 5
    k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]
    
    
    k_accuracies = {}
    classifier = K_Nearest_Neighbor()
    for k in k_choices:
        accuracies = np.zeros(num_folds)
        for fold in range(num_folds):
            X_train_folds = np.array_split(X, num_folds)
            Y_train_folds = np.array_split(Y, num_folds)       
            X_validate_fold = X_train_folds.pop(fold)
            Y_validate_fold = Y_train_folds.pop(fold)   
           
            # 剩下的四折
            temp_X = np.array([y for x in X_train_folds for y in x])
            temp_Y = np.array([y for x in Y_train_folds for y in x])
            classifier.train(temp_X, temp_Y)
            
            Y_pred = classifier.predict(X_validate_fold, k=k, num_loops=0)
            # 计算准确率
            num_correct = 0
            for i in range(X_validate_fold.shape[0]):
                if Y_pred[i] == Y_validate_fold[i]:
                    num_correct = num_correct + 1
            accuracy = float(num_correct) / X_validate_fold.shape[0]
            accuracies[fold] = accuracy
        
        k_accuracies[k] = accuracies
    
    # 输出准确率
    for k in sorted(k_accuracies):
        for accuracy in k_accuracies[k]:
            print ('k = %d, accuracy = %f' % (k, accuracy))
    
    # 画出准确率图形
    for k in k_choices:
        accuracies = k_accuracies[k]
        plt.scatter([k] * len(accuracies), accuracies, c='b')
    
    accuracies_mean = np.array([np.mean(v) for k, v in sorted(k_accuracies.items())])
    accuracies_std = np.array([np.std(v) for k, v in sorted(k_accuracies.items())])
    plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
    plt.title('Cross-validation on k')
    plt.xlabel('k')
    plt.ylabel('Cross-validation accuracy')
    plt.show()

          
def show_accuracies():
    k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]
    k_to_accuracies = {1: [0.526000, 0.514000, 0.528000, 0.556000, 0.532000],
                       3: [0.478000, 0.498000, 0.480000, 0.532000, 0.508000],
                       5: [0.496000, 0.532000, 0.560000, 0.584000, 0.560000],
                       8: [0.524000, 0.564000, 0.546000, 0.580000, 0.546000],
                       10: [0.530000, 0.592000, 0.552000, 0.568000, 0.560000],
                       12: [0.520000, 0.590000, 0.558000, 0.566000, 0.560000],
                       15: [0.504000, 0.578000, 0.556000, 0.564000, 0.548000],
                       20: [0.540000, 0.558000, 0.558000, 0.564000, 0.570000],
                       50: [0.542000, 0.576000, 0.556000, 0.538000, 0.532000],
                       100: [0.512000, 0.540000, 0.526000, 0.512000, 0.526000]}
    for k in k_choices:
        accuraices = k_to_accuracies[k]
        plt.scatter([k] * len(accuraices), accuraices, c='b')    # 画离散点
        
    # plot the trend line with error bars that correspond to standard deviation
    # sorted(k_to_accuracies.items())的值:[(1, [0.526, 0.514, 0.528, 0.556, 0.532]), (3, [0.478, 0.498, 0.48, 0.532, 0.508])....]
    accuracies_mean = np.array([np.mean(v) for k,v in sorted(k_to_accuracies.items())]) 
    accuracies_std = np.array([np.std(v) for k,v in sorted(k_to_accuracies.items())])   # np.std(): 计算矩阵标准差
    plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
    plt.title('Cross-validation on k')
    plt.xlabel('k')
    plt.ylabel('Cross-validation accuracy')
    plt.show()


#if __name__ == '__main__':
#    cross_validation()
    
