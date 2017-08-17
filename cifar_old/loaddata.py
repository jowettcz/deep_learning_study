#coding=utf-8
import os
import numpy as np
import tensorflow as tf


#区分为training，validation和test
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

cwd = os.getcwd()

def get_training_data():
    dict = unpickle(cwd + '/cifar10/cifar10-batches-py/data_batch_' + str(1))
    images = dict[b'data']
    labels = dict[b'labels']
    filenames = dict[b'filenames']

    for i in range(2,5):
        idict = unpickle(cwd + '/cifar10/cifar10-batches-py/data_batch_' + str(i));
        dict = np.row_stack((dict,idict))

        iimages = idict[b'data']
        images =  np.row_stack((images,iimages))

        ilabels = idict[b'labels']
        labels = np.column_stack((labels,ilabels))

        ifilenames = idict[b'filenames']
        filenames = np.row_stack((filenames,ifilenames))

    return {b'batch_label':'training batch,40000*3072',b'data':images,b'labels':labels,b'filenames':filenames}

def get_validation_data():
    dict = unpickle(cwd + '/cifar10/cifar10-batches-py/data_batch_' + str(5))
    dict[b'batch_label']='validation data,size is 10000*3072'
    return dict

def get_test_data():
    dict = unpickle(cwd + '/cifar10/cifar10-batches-py/test_batch')

    return dict

test_data = get_test_data()
print(test_data)
validation_data = get_validation_data()

training_data = get_training_data()
print(training_data)