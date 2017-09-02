# coding:utf-8

# the net may be you need to use
import numpy as np
from ._tf_layers import *

def mlpD_tf(input_dim, hidden_unit, output_dim, output_act_fun=None):
    """
    use tensorflow to build a Multilayer perceptron
    :param input_dim: your data dimension
    :param hidden_unit: the number of hidden unit in your nn model
    :param output_dim: output data dimension
    :param output_act_fun: the output activate function
    :return: a function use for building D or G
    """

    def mlp(inputs):
        l1 = dense(inputs, [input_dim, hidden_unit], name="relu1", act_fun=tf.nn.relu)
        l2 = dense(l1, [hidden_unit, hidden_unit], name="relu2", act_fun=tf.nn.relu)
        l3 = dense(l2, [hidden_unit, hidden_unit], name="relu3", act_fun=tf.nn.relu)
        y = dense(l3, [hidden_unit, output_dim], name="output")
        if output_act_fun is not None:
            y = output_act_fun(y)
        return y

    return mlp


def mlpG_tf(input_dim, hidden_unit, output_dim, output_act_fun=None):
    """
    use tensorflow to build a Multilayer perceptron
    :param input_dim: your data dimension
    :param hidden_unit: the number of hidden unit in your nn model
    :param output_dim: output data dimension
    :param output_act_fun: the output activate function
    :return: a function use for building D or G
    """

    def mlp(inputs):
        l1 = dense(inputs, [input_dim, hidden_unit], name="relu1", act_fun=leakyRelu)
        l2 = dense(l1, [hidden_unit, hidden_unit], name="relu2", act_fun=leakyRelu)
        l3 = dense(l2, [hidden_unit, hidden_unit], name="relu3", act_fun=leakyRelu)
        y = dense(l3, [hidden_unit, output_dim], name="output")
        if output_act_fun is not None:
            y = output_act_fun(y)
        return y

    return mlp

def ConvD_tf(image_size,fillter_size, output_act_fun=None):
    if len(image_size) == 2:
        W,H, = image_size
        channel = 1
    elif len(image_size) == 3:
        W,H,channel = image_size
    else:
        raise ValueError("Can't support this image size for now")
    fw,fh,hidden = fillter_size
    def convd(inputs):
        l1 = conv2d(tf.reshape(inputs, [-1, W, H, channel]), [fw, fh, channel,hidden], name="cov1",act_fun=tf.nn.relu)
        l2 = conv2d(l1, [fw, fh, hidden, hidden], name="cov2", act_fun=tf.nn.relu)
        l3 = conv2d(l2, [fw, fh, hidden, hidden], name="cov3", act_fun=tf.nn.relu)
        y = dense(flatten(l3),[W*H*hidden,1],name="conv_output",act_fun=output_act_fun)
        return y
    return convd

def deconvG_tf(z_size,image_size,fillter_size, batch_size,output_act_fun=None):
    if len(image_size) == 2:
        W,H, = image_size
        channel = 1
    elif len(image_size) == 3:
        W,H,channel = image_size
    else:
        raise ValueError("Can't support this image size for now")
    fw,fh,hidden = fillter_size
    def deconvd(inputs):
        l1= dense(inputs,[z_size,W*H*hidden],name="de_linear",act_fun=leakyRelu)
        l2 = decov2d(tf.reshape(l1, [-1, W, H, hidden]), filter_shape=[fw, fh], output_shape=[batch_size, W, H, hidden],
                name="deconv",act_fun=leakyRelu)
        l3 = decov2d(l2, filter_shape=[fw, fh], output_shape=[batch_size, W, H, hidden],
                     name="deconv2", act_fun=leakyRelu)
        l4 = decov2d(l3, filter_shape=[fw, fh], output_shape=[batch_size, W, H, channel],
                     name="deconv_output", act_fun=output_act_fun)
        l4 = flatten(l4)
        return l4
    return deconvd