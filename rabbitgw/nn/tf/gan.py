# coding:utf-8

# GANs model

import random
import numpy as np
from ...configure import BACKEND


class WGAN_TF:
    def __init__(self, x_size, z_size, net_G, net_D, lr=5e-5, clip=0.01, d_loss_addition=0, g_loss_addition=0,
                 batch_size=32):
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

        self.tf = BACKEND["tf"]
        self._z_size = z_size
        self._lr = lr
        self._clip = clip
        self._d_loss_addition = d_loss_addition
        self._g_loss_addition = g_loss_addition
        self._x = self.tf.placeholder("float32", [None, x_size])
        self._z = self.tf.placeholder("float32", [None, z_size])
        with self.tf.variable_scope("G"):
            self._G = net_G(self.z)
        with self.tf.variable_scope("D"):
            self._fake_D = net_D(self.G)
        with self.tf.variable_scope("D", reuse=True):
            self._real_D = net_D(self.x)
        self._batch_size = batch_size
        self._session = None
        self._netG = net_G
        self._netD = net_D
        self._build_loss_and_opt()
        self._dlist = self._get_train_dlist()

    @property
    def x(self):
        return self._x

    @property
    def z(self):
        return self._z

    @property
    def G(self):
        return self._G

    @property
    def fake_D(self):
        return self._fake_D

    @property
    def real_D(self):
        return self._real_D

    @property
    def d_clip(self):
        return self._d_clip

    @property
    def wd(self):
        return self._wd

    @property
    def d_loss(self):
        return self._d_loss

    @property
    def g_loss(self):
        return self._g_loss

    def _build_loss_and_opt(self):
        vars = self.tf.trainable_variables()

        D_PARAMS = [var for var in vars if var.name.startswith("D")]
        G_PARAMS = [var for var in vars if var.name.startswith("G")]

        d_clip = [self.tf.assign(var, self.tf.clip_by_value(var, -self._clip, self._clip)) for var in D_PARAMS]
        self._d_clip = self.tf.group(*d_clip)

        self._wd = self.tf.reduce_mean(self.real_D) - self.tf.reduce_mean(self.fake_D)
        self._d_loss = self.tf.reduce_mean(self.fake_D) - self.tf.reduce_mean(self.real_D) + self._d_loss_addition
        self._g_loss = self.tf.reduce_mean(-self.fake_D) + self._g_loss_addition

        self.d_opt = self.tf.train.RMSPropOptimizer(self._lr).minimize(
            self.d_loss,
            global_step=self.tf.Variable(0),
            var_list=D_PARAMS
        )

        self.g_opt = self.tf.train.RMSPropOptimizer(self._lr).minimize(
            self.g_loss,
            global_step=self.tf.Variable(0),
            var_list=G_PARAMS
        )

    def open_session(self,path=None,is_restore=False):

        if self._session is None:
            self._session = self.tf.Session()
        self._session.run(self.tf.global_variables_initializer())
        if is_restore:
            saver = self.tf.train.Saver()
            saver.restore(self._session, path)


    def save_model(self, path):
        """
        save the model in your path
        :param path: your path
        :return:
        """
        if self._session is None:
            raise ValueError("session is None")
        saver = self.tf.train.Saver()
        saver.save(self._session, path)

    def close_session(self):
        self._session.close()
        self._session = None

    def _get_train_dlist(self):
        return [self.wd, self.d_loss, self.d_opt, self.d_clip]

    def fit(self, x, epoch, visual=True, callbacks=None):
        """

        :param x: your input
        :param epoch: the train epoch of this model
        :param batch_size: the batch size of this model
        :param visual: Will print the progress of train if the value is True,otherwise it will be silent
        :param callbacks: a function list called after a epoch
        :return:
        """

        if self._session is None:
            raise ValueError("session is None")


        def predict(n):
            return self.predict(n)

        length = len(x)
        batch_size = self._batch_size
        for ep in range(epoch):
            index = [_ for _ in range(length)] # will be repeated if shuffle x directly
            random.shuffle(index)
            train_image = [x[i] for i in index]
            # random.shuffle(x)
            step = 0
            i = 0
            j = i + batch_size
            g_loss = np.inf
            d_loss = np.inf
            wd = np.inf
            while step < length:
                for _ in range(5):
                    noise = np.random.normal(size=(batch_size, self._z_size))
                    if abs(j - i) < batch_size:
                        i = i - (batch_size - (j - i))
                    wd, d_loss, _, _ = self._session.run(self._dlist, feed_dict={
                        self.x: train_image[i:j],
                        self.z: noise
                    })
                    i = j
                    j += batch_size
                    if j > length:
                        j = length

                    step += batch_size
                    if i == length:
                        break
                    if visual:
                        print("\rep:%d/%d    %d/%d    d_loss:%.4f   g_loss:%.4f    wd:%.4f" % (
                            ep + 1, epoch, step, length, d_loss, g_loss, wd), end="")
                for _ in range(1):
                    noise = np.random.normal(size=(batch_size, self._z_size))
                    g_loss, _ = self._session.run([self.g_loss, self.g_opt], feed_dict={
                        self.z: noise
                    })
                    if visual:
                        print("\rep:%d/%d    %d/%d    d_loss:%.4f   g_loss:%.4f    wd:%.4f" % (
                            ep + 1, epoch, step, length, d_loss, g_loss, wd), end="")
            if visual:
                print()
            context = {
                "predict": predict,
                "ep": ep,
                "g_loss": g_loss,
                "wd": wd,
                "d_loss": d_loss
            }
            if callbacks is not None:
                for callback in callbacks:
                    callback(context)

    def predict(self, n):
        """
        generate the sample with noise
        :param n: the number of sample.
        :return:
        """
        if self._session is None:
            raise ValueError("session is None")
        noise = np.random.normal(size=(n, self._z_size))
        gen = self._session.run(self.G, feed_dict={
            self.z: noise
        })
        return gen


class WGAN_GP_TF(WGAN_TF):
    def __init__(self, x_size, z_size, net_G, net_D, lr=1e-4, d_loss_addition=0, g_loss_addition=0,batch_size=32, LAMBDA=1, K=1):
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
        self._K = K
        self._LAMBDA = LAMBDA
        super(WGAN_GP_TF, self).__init__(x_size, z_size, net_G, net_D, lr=lr,
                                         d_loss_addition=d_loss_addition,
                                         g_loss_addition=g_loss_addition, clip=np.inf,
                                         batch_size=batch_size)



    def _build_loss_and_opt(self):
        vars = self.tf.trainable_variables()

        D_PARAMS = [var for var in vars if var.name.startswith("D")]
        G_PARAMS = [var for var in vars if var.name.startswith("G")]

        self._wd = self.tf.reduce_mean(self.real_D) - self.tf.reduce_mean(self.fake_D)
        self._d_loss = self.tf.reduce_mean(self.fake_D) - self.tf.reduce_mean(self.real_D) + self._d_loss_addition
        self._g_loss = self.tf.reduce_mean(-self.fake_D) + self._g_loss_addition

        #######GP-METHOD
        alpha = self.tf.random_uniform(
            shape=[self._batch_size, 1],
            minval=0.,
            maxval=1.
        )  # sampling
        insert_value = self.G - alpha * (self.x - self.G)
        with self.tf.variable_scope("D", reuse=True):
            gradients = self.tf.gradients(self._netD(insert_value), [insert_value])[0]
        slopes = self.tf.sqrt(self.tf.reduce_sum(self.tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = self.tf.reduce_mean((slopes - self._K) ** 2)
        #######
        self._d_loss += self._LAMBDA * gradient_penalty
        self.d_opt = self.tf.train.AdamOptimizer(self._lr, beta1=0.4, beta2=0.9).minimize(
            self._d_loss,
            global_step=self.tf.Variable(0),
            var_list=D_PARAMS
        )

        self.g_opt = self.tf.train.AdamOptimizer(self._lr, beta1=0.4, beta2=0.9).minimize(
            self._g_loss,
            global_step=self.tf.Variable(0),
            var_list=G_PARAMS
        )

    def _get_train_dlist(self):
        return [self.wd, self.d_loss, self.d_opt, self.tf.Variable(0)]


    @property
    def x(self):
        return self._x

    @property
    def z(self):
        return self._z

    @property
    def G(self):
        return self._G

    @property
    def fake_D(self):
        return self._fake_D

    @property
    def real_D(self):
        return self._real_D

    @property
    def d_clip(self):
        return self._d_clip

    @property
    def wd(self):
        return self._wd

    @property
    def d_loss(self):
        return self._d_loss

    @property
    def g_loss(self):
        return self._g_loss
