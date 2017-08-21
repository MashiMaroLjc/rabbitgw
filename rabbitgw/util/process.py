#coding:utf-8

import numpy as np

# the process usually need to do

def standardlize(x,mean=None,std=None):
    """
    standardlize the data
    :param x: the data you need to process
    :param mean: the mean of your data.it will be auto compute if the values is None.Also you can set a certain value
    :param std:  the std of your data.it will be auto compute if the values is None.Also you can set a certain value
    :return: 
    """
    if mean is None or std is None:
        mean = np.mean(x)
        std = np.std(x)

    return (x-mean)/std


def inverse_standardlize(x,mean,std):
    """
    inverse standardlize data
    :param x:
    :param mean:
    :param std:
    :return:
    """
    return x*std + mean