# coding:utf-8
# the global configure

BACKEND = dict()

try:
    import tensorflow as tf
    BACKEND["tf"] = tf
except ImportError:
    pass