#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 20:56:09 2018

@author: caozhang
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import numpy as np
import platform                # platform模块提供了很多方法去获取操作系统的信息
from scipy.misc import imread
import pickle                  # pkl文件是python里面保存文件的一种格式, 使用cPickle打开


def load_pickle(f):
    """
    Args:
        f: dataset file name
    """
    version = platform.python_version_tuple()
    if version[0] == '2':
        return pickle.load(f)
    elif version[0] == '3':
        return pickle.load(f, encoding='latin1')
    raise ValueError('Invalid python version: {}'.format(version))
    

def load_CIFAR_batch(file_name):
    """
    load single cifar batch
    """
    with open(file_name, 'rb') as f:
        data_dict = load_pickle(f)
        X = data_dict['data']
        Y = data_dict['labels']
        # 以float类型读取便于数值计算
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype('float') # data_dict['data']对应的值是10000张图像的平铺
        Y = np.array(Y)
        return X, Y


def load_CIFAR10(root_dir):
    """
    load all images data
    """
    Xtr = []
    Ytr = []
    for i in range(1, 6):
        file_name = os.path.join(root_dir, 'data_batch_%d' % i)
        X, Y = load_CIFAR_batch(file_name)
        Xtr.append(X)           # shape of Xtr: [5, 10000, 32, 32, 3]
        Ytr.append(Y)
        
    Xtr = np.concatenate(Xtr)   #shape of Xtr: [50000, 32, 32, 3]
    Ytr = np.concatenate(Ytr)
    Xte, Yte = load_CIFAR_batch(os.path.join(root_dir, 'test_batch'))
    return Xtr, Ytr, Xte, Yte


    
        
    
    
    
    
    
