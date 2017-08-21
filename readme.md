#rabbitgw
## what is rabbitgw?

*rabbitgw* is a GAN wrapper aim to provide a steady train method for GAN model like wgan and wang-gp.It also support different backend like tensorflow and pytorch.But only support tensorflow for now.

## What can I benefit from it?

The GAN(Generative Adversarial Networks) model is a great idea of deep learning.Howerver the process of training ,control the Generator as well as Discriminator but Discriminator always better than Generator so that the Gradient disappeared ,is so hard.Thus,people come up with some way to slove it.Among them,WGAN and WGAN-GP usually be better,which change the goal(loss function) what the model optimize.


For the easy task,like generate image from the noise,you can use WGAN and WGAN-GP the *rabbitgw* implemented to obtain a acceptable and steady result directly.If you meet a complex task like dual learning,you can view *rabbitgw* as a wrapper,which save your time for bulid the loss function.You can get the loss function built by *rabbitgw*,then you can use it to solve your complex task.

## When can I use it?

*rabbitgw* is a toy project,you can use it when you project is a toy. :P

## How to use it?

There are example using rabbitgw to train on minst

```python
# coding:utf-8

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image

# The WGAN_GP,WGAN wrapper in nn moudl 
from rabbitgw.nn import WGAN_GP_TF
# rabbitgw.util provide the tool you usually need,like the mlp(Multilayer perceptron) model or some process for the data.
from rabbitgw.util.net import mlp_tf
from rabbitgw.util.process import inverse_standardlize,standardlize

Mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
X,_ = Mnist.train.next_batch(50000)
X = standardlize(X,0,1)
mlp_d = mlp_tf(28*28,256,1)
mlp_g = mlp_tf(100,256,28*28,output_act_fun=tf.nn.sigmoid)
wgan =  WGAN_GP_TF(x_size=28*28,z_size=100,net_G=mlp_g,net_D=mlp_d)

# The callback function will be called after every epoch,it accept a dict.
# include following key:
# ep - the epoch number 
# d_loss - the value of loss function of D net
# g_loss - the value of loss function of G net
# wd - W distance(only WGAN and WGAN-GP)
# some key will be add in funture.
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
wgan.close_session()


```

The result 

![](/iamge/1.jpg)
![](/iamge/2.jpg)