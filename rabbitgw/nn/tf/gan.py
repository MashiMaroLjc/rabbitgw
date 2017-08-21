#coding:utf-8

# GANs model

import random
import numpy as np


class WGAN_TF:

    def __init__(self,x_size,z_size,net_G,net_D,lr=5e-5,clip=0.01,d_loss_addition=0,g_loss_addition=0):
        """

        :param x_size: the size of every input sample.
        :param z_size:  the size of every noise sample.
        :param net_G: a function ,build the G
        :param net_D: a function build the D
        :param lr: learning rate
        :param clip: the value use for cliping the weight
        :param d_loss_addition:  a number of graph of tensorflow will add with loss function of d
        :param g_loss_addition:  a number of graph of tensorflow will add with loss function of g

        """
        import tensorflow
        self.tf = tensorflow
        self.z_size=  z_size
        self.lr = lr
        self.clip = clip
        self.d_loss_addition= d_loss_addition
        self. g_loss_addition = g_loss_addition
        self.x = self.tf.placeholder("float32", [None, x_size])
        self.z = self.tf.placeholder("float32", [None, z_size])
        with self.tf.variable_scope("G"):
            self.G = net_G(self.z)
        with self.tf.variable_scope("D"):
            self.fake_D = net_D(self.G)
        with self.tf.variable_scope("D",reuse=True):
            self.real_D = net_D(self.x)

        self.session = None
        self._netD =net_D


    def _build_loss_and_opt(self):
        vars = self.tf.trainable_variables()

        D_PARAMS = [var for var in vars if var.name.startswith("D")]
        G_PARAMS = [var for var in vars if var.name.startswith("G")]

        d_clip = [self.tf.assign(var, self.tf.clip_by_value(var, -self.clip, self.clip)) for var in D_PARAMS]
        self.d_clip = self.tf.group(*d_clip)

        self.wd = self.tf.reduce_mean(self.real_D) - self.tf.reduce_mean(self.fake_D)
        self.d_loss = self.tf.reduce_mean(self.fake_D) - self.tf.reduce_mean(self.real_D)+ self.d_loss_addition
        self.g_loss = self.tf.reduce_mean(-self.fake_D) + self.g_loss_addition

        self.d_opt = self.tf.train.RMSPropOptimizer(self.lr).minimize(
            self.d_loss,
            global_step=self.tf.Variable(0),
            var_list=D_PARAMS
        )

        self.g_opt = self.tf.train.RMSPropOptimizer(self.lr).minimize(
            self.g_loss,
            global_step=self.tf.Variable(0),
            var_list=G_PARAMS
        )

    def open_session(self):
        if self.session is None:
            self.session = self.tf.Session()

    def save_model(self,path):
        """
        save the model in your path
        :param path: your path
        :return:
        """
        if self.session is None:
            raise ValueError("session is None")
        saver = self.tf.train.Saver()
        saver.save(self.session, path)


    def load_model(self,path):
        """
        load the model from your path
        :param path: your path
        :return:
        """
        if self.session is None:
            raise ValueError("session is None")
        saver = self.tf.train.Saver()
        saver.restore(self.session, path)

    def close_session(self):
        self.session.close()
        self.session = None

    def _get_train_dlist(self):
        return [self.wd,self.d_loss,self.d_opt,self.d_clip]

    def fit(self,x,epoch,batch_size=32,visual=True,callbacks =None):
        """

        :param x: your input
        :param epoch: the train epoch of this model
        :param batch_size: the batch size of this model
        :param visual: Will print the progress of train if the value is True,otherwise it will be silent
        :param callbacks: a function list called after a epoch
        :return:
        """
        self._build_loss_and_opt()
        dlist = self._get_train_dlist()
        if self.session is None:
            raise ValueError("session is None")
        self.session.run(self.tf.global_variables_initializer())
        def predict(n):
            return self.predict(n)
        length = len(x)
        for ep in range(epoch):
            random.shuffle(x)
            step = 0
            i = 0
            j = i+batch_size
            g_loss = np.inf
            d_loss = np.inf
            wd = np.inf
            while step < length:
                for _ in range(5):
                    noise = np.random.normal(size=(batch_size, self.z_size))
                    if abs(j-i) < batch_size:
                        i=i-(batch_size-(j-i))
                    wd,d_loss,_,_ = self.session.run(dlist,feed_dict={
                        self.x:x[i:j],
                        self.z:noise
                    })
                    i+=batch_size
                    j+=batch_size
                    if j > length:
                        j = length
                    step += batch_size
                    if visual:
                        print("\rep:%d/%d    %d/%d    d_loss:%.4f   g_loss:%.4f    wd:%.4f"%(
                            ep+1,epoch,step,length,d_loss,g_loss,wd),end="")
                for _ in range(1):
                    noise = np.random.normal(size=(batch_size, self.z_size))
                    g_loss,_ = self.session.run([self.g_loss,self.g_opt],feed_dict={
                        self.z:noise
                    })
                    if visual:
                        print("\rep:%d/%d    %d/%d    d_loss:%.4f   g_loss:%.4f    wd:%.4f" % (
                        ep + 1,epoch, step, length, d_loss, g_loss, wd), end="")
            print()
            context = {
                "predict":predict,
                "ep":ep,
                "g_loss":g_loss,
                "wd":wd,
                "d_loss":d_loss
            }
            if callbacks is not None:
                for callback in callbacks:
                    callback(context)



    def predict(self,n):
        """
        generate the sample with noise
        :param n: the number of sample.
        :return:
        """
        if self.session is None:
            raise ValueError("session is None")
        noise = np.random.normal(size=(n, self.z_size))
        gen = self.session.run(self.G,feed_dict={
            self.z:noise
        })
        return gen

    def get_d_loss(self):
        return self.d_loss

    def get_g_loss(self):
        return self.g_loss

    def get_d_opt(self):
        return self.d_opt

    def get_g_opt(self):
        return self.g_opt


class WGAN_GP_TF(WGAN_TF):

    def __init__(self,x_size,z_size,net_G,net_D,lr=1e-4,d_loss_addition=0,g_loss_addition=0,LAMBDA=1,K=1):
        """

        :param x_size:
        :param z_size:
        :param net_G:
        :param net_D:
        :param lr:
        :param d_loss_addition:
        :param g_loss_addition:
        :param LAMBDA:
        :param K:
        """
        super(WGAN_GP_TF,self).__init__(x_size,z_size,net_G,net_D,lr=lr,
                                        d_loss_addition=d_loss_addition,g_loss_addition=g_loss_addition,clip=None)
        self.K = K
        self.LAMBDA = LAMBDA

    def _build_loss_and_opt(self):
        vars = self.tf.trainable_variables()

        D_PARAMS = [var for var in vars if var.name.startswith("D")]
        G_PARAMS = [var for var in vars if var.name.startswith("G")]

        self.wd = self.tf.reduce_mean(self.real_D) - self.tf.reduce_mean(self.fake_D)
        self.d_loss = self.tf.reduce_mean(self.fake_D) - self.tf.reduce_mean(self.real_D) +self.d_loss_addition
        self.g_loss = self.tf.reduce_mean(-self.fake_D) + self.g_loss_addition


        #######GP-METHOD
        alpha = self.tf.random_uniform(
            shape=[self.batch_size, 1],
            minval=0.,
            maxval=1.
        )  # 采样
        insert_value = self.G - alpha * (self.x - self.G)
        with self.tf.variable_scope("D", reuse=True):
            gradients = self.tf.gradients(self._netD(insert_value), [insert_value])[0]
        slopes = self.tf.sqrt(self.tf.reduce_sum(self.tf.square(gradients), reduction_indices=[1]))  # 求范数
        gradient_penalty = self.tf.reduce_mean((slopes -self.K) ** 2)  # 最少化这个会使梯度集中在K值附近
        #######
        self.d_loss += self.LAMBDA * gradient_penalty
        self.d_opt = self.tf.train.AdamOptimizer(self.lr, beta1=0.4, beta2=0.9).minimize(
            self.d_loss,
            global_step=self.tf.Variable(0),
            var_list=D_PARAMS
        )

        self.g_opt = self.tf.train.AdamOptimizer(self.lr, beta1=0.4, beta2=0.9).minimize(
            self.g_loss,
            global_step=self.tf.Variable(0),
            var_list=G_PARAMS
        )

    def _get_train_dlist(self):
        return [self.wd,self.d_loss,self.d_opt,self.tf.Variable(0)]

    def fit(self,x,epoch,batch_size=32,visual=True,callbacks =None):
        self.batch_size =batch_size
        super().fit(x,epoch,batch_size,visual,callbacks)