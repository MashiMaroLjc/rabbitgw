# coding:utf-8
from ...configure import BACKEND
import numpy as np
tf = BACKEND["tf"]


def leakyRelu(x, alpha=0.2):
    output = tf.maximum(alpha * x, x)
    return output

def dense(inputs, shape, name, act_fun=None):
    """
    the dense layer
    :param inputs:  the input tensor
    :param shape:  the shape of W matrix
    :param name: the name of layer
    :param act_fun: activation function of tensorflow
    :return:
    """
    W = tf.get_variable(name + ".w", initializer=tf.random_normal(shape=shape) / np.sqrt(shape[0] / 2))
    b = tf.get_variable(name + ".b", initializer=(tf.zeros((1, shape[-1])) + 0.1))
    y = tf.add(tf.matmul(inputs, W), b)
    if act_fun is not None:
        y = act_fun(y)
    return y


def batch_normalization(inputs, out_size, name, axes=0):
    """
    the batch normalization layer
    :param inputs: the input tensor
    :param out_size:the size of output tensor
    :param name:the name of layer
    :param axes:the axes of the matrix calculating the mean and var.
    :return:
    """
    mean, var = tf.nn.moments(inputs, axes=[axes])
    scale = tf.get_variable(name=name + ".scale", initializer=tf.ones([out_size]))
    offset = tf.get_variable(name=name + ".shift", initializer=tf.zeros([out_size]))
    epsilon = 0.001
    return tf.nn.batch_normalization(inputs, mean, var, offset, scale, epsilon, name=name + ".bn")

def max_pooling(inputs, ksize=(2, 2), stride=(2, 2), padding="SAME"):
    """

    :param inputs:
    :param ksize:
    :param stride:
    :param padding:
    :return:
    """
    return tf.nn.max_pool(inputs,
                          ksize=[1, ksize[0], ksize[1], 1],
                          strides=[1, stride[0], stride[1], 1], padding=padding)


def flatten(inputs):
    """
    flatten layer
    :param inputs:
    :return:
    """
    dim = tf.reduce_prod(tf.shape(inputs)[1:])
    return tf.reshape(inputs, [-1, dim])


def decov2d(inputs, filter_shape, output_shape, name, stride=(1, 1),
            padding="SAME", baies=True, act_fun=None):
    """

    :param inputs:
    :param filter_shape:
    :param output_shape:
    :param name:
    :param stride:
    :param padding:
    :param baies:
    :param act_fun:
    :return:
    """
    w = tf.get_variable(name + ".w", shape=[filter_shape[0], filter_shape[1], output_shape[-1], inputs.get_shape()[-1]],
                        initializer=tf.truncated_normal_initializer(stddev=0.02))
    if baies:
        b = tf.get_variable(name + ".b", [output_shape[-1]], initializer=tf.constant_initializer(0.01))
        convt = tf.nn.conv2d_transpose(inputs, w, output_shape=output_shape,
                                       strides=[1, stride[0], stride[1], 1], padding=padding) + b
    else:
        convt = tf.nn.conv2d_transpose(inputs, w, output_shape=output_shape,
                                       strides=[1, stride[0], stride[1], 1], padding=padding)
    if act_fun:
        convt = act_fun(convt)
    return convt


def conv2d(inputs, filter_shape, name, stride=(1, 1), padding="SAME", baies=True, act_fun=None):
    """

    :param inputs:
    :param filter_shape:
    :param name:
    :param stride:
    :param padding:
    :param baies:
    :param act_fun:
    :return:
    """
    filter_w = tf.get_variable(name + ".w", shape=filter_shape,
                               initializer=tf.truncated_normal_initializer(stddev=0.02))
    if baies:
        filter_b = tf.get_variable(name + ".b", initializer=tf.zeros((1, filter_shape[-1])) + 0.1)
        feature_map = tf.nn.conv2d(inputs, filter_w, strides=[1, stride[0], stride[1], 1], padding=padding) + filter_b
    else:
        feature_map = tf.nn.conv2d(inputs, filter_w, strides=[1, stride[0], stride[1], 1], padding=padding)
    if act_fun:
        feature_map = act_fun(feature_map)
    return feature_map

