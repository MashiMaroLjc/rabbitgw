# coding:utf-8
# this is a example of WGAN
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image
from rabbitgw.nn import WGAN_TF,WGAN_GP_TF
from rabbitgw.util.net import mlp_tf
from rabbitgw.util.process import inverse_standardlize,standardlize

Mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
X,_ = Mnist.train.next_batch(50000)
X = standardlize(X,0,1)
mlp_d = mlp_tf(28*28,256,1)
mlp_g = mlp_tf(100,256,28*28,output_act_fun=tf.nn.sigmoid)
wgan =  WGAN_GP_TF(x_size=28*28,z_size=100,net_G=mlp_g,net_D=mlp_d)

def my_callback(context):
    ep = context.get("ep")
    predict = context.get("predict")
    image = predict(1)
    image = inverse_standardlize(image,0,1).reshape([28,28])
    image = (image * 255).astype("uint8")
    Image.fromarray(image).save("image/ep%d.jpg"%(ep))

wgan.open_session()
wgan.fit(X,epoch=500,callbacks=[my_callback])
wgan.save_model("model/gan.pkct")
wgan.close()
