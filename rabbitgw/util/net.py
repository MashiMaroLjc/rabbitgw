#coding:utf-8

# the net may be you need to use
import numpy as np

def mlp_tf(input_dim,hidden_unit,output_dim,output_act_fun = None):
    """
    use tensorflow to build a Multilayer perceptron
    :param input_dim: your data dimension
    :param hidden_unit: the number of hidden unit in your nn model
    :param output_dim: output data dimension
    :param output_act_fun: the output activate function
    :return: a function use for building D or G
    """
    import tensorflow as tf
    def dense(inputs, shape, name,act_fun=None):
        W = tf.get_variable(name + ".w", initializer=tf.random_normal(shape=shape) / np.sqrt(shape[0] / 2))
        b = tf.get_variable(name + ".b", initializer=(tf.zeros((1, shape[-1])) + 0.1))
        y = tf.add(tf.matmul(inputs, W), b)
        if act_fun is not None:
            y = act_fun(y)
        return y
    def mlp(inputs):
        l1 = dense(inputs, [input_dim, hidden_unit], name="relu1", act_fun=tf.nn.relu)
        l2 = dense(l1, [hidden_unit, hidden_unit], name="relu2", act_fun=tf.nn.relu)
        l3 = dense(l2, [hidden_unit, hidden_unit], name="relu3", act_fun=tf.nn.relu)
        y = dense(l3, [hidden_unit, output_dim], name="output")
        if output_act_fun is not None:
            y = output_act_fun(y)
        return y
    return mlp
