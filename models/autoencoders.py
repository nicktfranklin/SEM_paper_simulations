import edward as ed
import tensorflow as tf
from edward.models import Normal, MultivariateNormalDiag


class AutoEncoder(object):

    def __init__(self, n_hidden, n_sample=5, n_iter=5000):

        n_hidden = 18

        X_ph = tf.placeholder(tf.float32, [None, D])

        def encoder(X, W, b):
            y = tf.nn.tanh(tf.matmul(X, W) + b)
            return y

        def decoder(X, W, b):
            y = tf.matmul(X, W) + b
            return y

        W_0 = Normal(loc=tf.zeros([D, n_hidden]), scale=tf.ones([D, n_hidden]), name="W_0")
        W_1 = Normal(loc=tf.zeros([n_hidden, D]), scale=tf.ones([n_hidden, D]), name="W_1")

        b_0 = Normal(loc=tf.zeros(n_hidden), scale=tf.ones(n_hidden), name="b_0")
        b_1 = Normal(loc=tf.zeros(D), scale=tf.ones(D), name="b_1")

        qW_0 = Normal(loc=tf.Variable(tf.random_normal([D, n_hidden])),
                      scale=tf.nn.softplus(tf.Variable(tf.random_normal([D, n_hidden]))))
        qW_1 = Normal(loc=tf.Variable(tf.random_normal([n_hidden, D])),
                      scale=tf.nn.softplus(tf.Variable(tf.random_normal([n_hidden, D]))))

        qb_0 = Normal(loc=tf.Variable(tf.random_normal([n_hidden])),
                      scale=tf.nn.softplus(tf.Variable(tf.random_normal([n_hidden]))))
        qb_1 = Normal(loc=tf.Variable(tf.random_normal([D])),
                      scale=tf.nn.softplus(tf.Variable(tf.random_normal([D]))))

        scale = tf.nn.softplus(tf.Variable(tf.random_normal([1])))

        encoded = encoder(X_ph, W_0, b_0,)
        decoded = decoder(encoded, W_1, b_1)

        y = MultivariateNormalDiag(loc=decoded, scale_diag = tf.ones(D))

        inference = ed.KLqp({W_0: qW_0, W_1: qW_1,
                             b_0: qb_0, b_1: qb_1,},
                           data={X_ph: X_train, y: y_train})
        inference.run(n_samples=5, n_iter=5000)