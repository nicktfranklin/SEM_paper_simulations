import tensorflow as tf
import edward as ed
from keras.models import Sequential
from keras.layers import Dense, Activation, SimpleRNN
from keras import optimizers
from tensorflow.contrib import slim, rnn
from scipy.stats import multivariate_normal
import numpy as np


def unroll_data(X, t=1):
    """
    unrolls a data_set for with time-steps, truncated for t time-steps
    appends t-1 D-dimensional zero vectors at the beginning.

    Parameters:
        X: array, shape (N, D) or shape (D,)

        t: int
            time-steps to truncate the unroll

    output
    ------

        X_unrolled: array, shape (N-1, t, D)

    """
    if np.ndim(X) == 2:
        N, D = np.shape(X)
    elif np.ndim(X):
        N, D = 1, np.shape(X)[0]
        X = np.reshape(X, (1, D))

    X_unrolled = np.zeros((N, t, D))

    # append a t-1 blank (zero) input patterns to the beginning
    data_set = np.concatenate([np.zeros((t - 1, D)), X])

    for ii in range(N):
        X_unrolled[ii, :, :] = data_set[ii: ii + t, :]

    return X_unrolled


class EventModel(object):

    def __init__(self, D):

        self.D = D

    def update(self, X, Y):
        pass

    def predict(self, X):
        return np.copy(X)

    def log_likelihood(self, X, Y, Sigma):
        return multivariate_normal.pdf(X - Y, mean=np.zeros(self.D), cov=Sigma)

    def close(self):
        pass


class KerasLDS(EventModel):

    def __init__(self, D, sgd_kwargs=None):
        EventModel.__init__(self, D)
        self.x_train = np.zeros((0, self.D))  # initialize empty arrays
        self.y_train = np.zeros((0, self.D))
        self.is_initialized = False
        if sgd_kwargs is None:
            sgd_kwargs = dict(lr=0.01, momentum=0.9, nesterov=True)

        self.compile_opts = dict(optimizer=optimizers.SGD(**sgd_kwargs), loss='mean_squared_error')

    def _estimate(self):
        self.sess = tf.Session()

        N, D = self.x_train.shape

        self.model = Sequential([
            Dense(D, input_shape=(D, )),
            Activation('linear')
        ])

        self.model.compile(**self.compile_opts)

        self.model.fit(self.x_train, self.y_train, verbose=0)

    def update(self, X, Y, estimate=True):
        if np.ndim(X) == 1:
            N = 1
        else:
            N, _ = np.shape(X)
        self.x_train = np.concatenate([self.x_train, np.reshape(X, newshape=(N, self.D))])
        self.y_train = np.concatenate([self.y_train, np.reshape(Y, newshape=(N, self.D))])

        if estimate:
            self._estimate()
            self.is_initialized = True

    def predict(self, X):
        if self.is_initialized:
            if np.ndim(X) == 1:
                N = 1
            else:
                N, _ = np.shape(X)
            return self.model.predict(np.reshape(X, newshape=(N, self.D)))
        else:
            return X

    def log_likelihood(self, X, Y, Sigma):

        Y_hat = self.predict(X)
        LL = np.log(multivariate_normal.pdf(Y - Y_hat, mean=np.zeros(self.D), cov=Sigma))

        return LL


class KerasMultiLayerNN(KerasLDS):

    def __init__(self, D, n_hidden=None, hidden_act = 'tanh', sgd_kwargs=None):
        KerasLDS.__init__(self, D, sgd_kwargs=sgd_kwargs)
        if n_hidden is None:
            n_hidden = D
        self.n_hidden = n_hidden
        self.hidden_act = hidden_act

    def _estimate(self):

        N, D = self.x_train.shape

        self.model = Sequential()
        self.model.add(Dense(self.n_hidden, input_shape=(D,), activation=self.hidden_act))
        self.model.add(Dense(D, activation='linear'))

        self.model.compile(**self.compile_opts)
        self.model.fit(self.x_train, self.y_train, verbose=0)


class KerasRNN(KerasLDS):

    def __init__(self, D, t=5, n_hidden1=10, n_hidden2=10, hidden_act1='tanh', hidden_act2='tanh', sgd_kwargs=None):
        KerasLDS.__init__(self, D, sgd_kwargs=sgd_kwargs)
        self.t = t
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2
        self.hidden_act1 = hidden_act1
        self.hidden_act2 = hidden_act2

    def _estimate(self):

        x_train0 = unroll_data(self.x_train, self.t)

        self.model = Sequential()
        self.model.add(SimpleRNN(self.D, input_shape=(self.t, self.D), activation=self.hidden_act1))
        self.model.add(Dense(self.n_hidden2, activation=self.hidden_act2))
        self.model.add(Dense(self.D, activation=None))

        self.model.compile(**self.compile_opts)
        self.model.fit(x_train0, self.y_train, verbose=0)

    def predict(self, X):
        if self.is_initialized:
            x_test0 = unroll_data(X, self.t)
            return self.model.predict(x_test0)
        else:
            return X


class KerasSimpleRNN(KerasLDS):

    def __init__(self, D, t=5, n_hidden=10, hidden_act=None, sgd_kwargs=None):
        KerasLDS.__init__(self, D, sgd_kwargs=sgd_kwargs)
        self.t = t
        self.n_hidden = n_hidden
        self.hidden_act = hidden_act

    def _estimate(self):

        x_train0 = unroll_data(self.x_train, self.t)

        self.model = Sequential()
        self.model.add(SimpleRNN(self.n_hidden, input_shape=(self.t, self.D), activation=self.hidden_act))
        self.model.add(Dense(self.D, activation=None))

        self.model.compile(**self.compile_opts)
        self.model.fit(x_train0, self.y_train, verbose=0)

    def predict(self, X):
        if self.is_initialized:
            x_test0 = unroll_data(X, self.t)
            return self.model.predict(x_test0)
        else:
            return X


class KerasSRN_batch(KerasLDS):

    def __init__(self, D, t=5, sgd_kwargs=None, n_batch=100, aug_noise=0.01):
        KerasLDS.__init__(self, D, sgd_kwargs=None)
        self.t = t
        self.n_batch = n_batch
        self.noise = aug_noise

    def _estimate(self):

        self.model = Sequential()
        self.model.add(SimpleRNN(self.D, input_shape=(self.t, self.D), activation=None))
        self.model.add(Dense(self.D*self.D, activation='tanh'))
        self.model.add(Dense(self.D, activation=None))

        self.model.compile(**self.compile_opts)


        N, D = np.shape(self.x_train)
        for _ in range(self.n_batch):
            noise = np.reshape(np.random.randn(N, D) * self.noise, np.shape(self.x_train))
            X0 = self.x_train + noise
            y0 = self.y_train + noise
            self.model.train_on_batch(unroll_data(X0, self.t), y0)

    def predict(self, X):
        if self.is_initialized:
            x_test0 = unroll_data(X, self.t)
            return self.model.predict(x_test0)
        else:
            return X



class EdwardLDS(EventModel):
    def __init__(self, D, n_samples=100):
        """
        Parameters:
        -----------

        D: int
            dimensions of the data set

        n_samples: int
            number of samples to draw of the posterior

        """
        EventModel.__init__(self, D)
        self.x_train = np.zeros((0, self.D))  # initialize empty arrays
        self.y_train = np.zeros((0, self.D))

        # initialize samples, for untrained (new) models, predict X'_hat = X
        self.n_samples = n_samples
        self.w_samples = np.reshape(np.eye(self.D), (1, self.D, self.D))
        self.b_samples = np.zeros((1, self.D))
        self.initialized = False

    def _initialize_model(self, N):
        """
        Parameters:
        -----------
        N: int
            training sample size
        """

        self.W_0 = ed.models.Normal(loc=tf.zeros([self.D, self.D]), scale=tf.ones([self.D, self.D]), name="W_0")
        self.b_0 = ed.models.Normal(loc=tf.zeros(self.D), scale=tf.ones(self.D), name="b_0")

        self.X = tf.placeholder(tf.float32, [None, self.D], name="X")

        self.y = ed.models.MultivariateNormalDiag(
            loc=tf.matmul(self.X, self.W_0) + self.b_0,
            scale_diag=tf.ones(self.D),
            name="y"
        )

        self.qW_0 = ed.models.Normal(loc=tf.Variable(tf.random_normal([self.D, self.D]), name="loc"),
                           scale=tf.nn.softplus(tf.Variable(tf.random_normal([self.D, self.D]), name="scale")))

        self.qb = ed.models.Normal(loc=tf.Variable(tf.random_normal([self.D])),
                         scale=tf.nn.softplus(tf.Variable(tf.random_normal([self.D]))))

    def _train(self, x_train, y_train, n_samples=3):
        """
        Parameters
        ----------

        x_train: NxD array
            N exemplars of D-dimensional states

        y_train: NxD array
            N exemplars of D-dimensional successor states

        n_samples: int
            number of samples to use in inference
        """
        inference = ed.KLqp({self.W_0: self.qW_0, self.b_0: self.qb},
                            data={self.X: x_train, self.y: y_train})
        # inference.run(n_samples=5, n_iter=1000)
        inference.run(n_samples=n_samples)

    def update(self, X, Y, estimate=True):
        """
        Parameters
        ----------

        X: np.array, length D
            observed state vector

        Y: np.array, length D
            observed successor state vector

        estimate: boolean
            estimate the network parameters?
        """
        self.x_train = np.concatenate([self.x_train, np.reshape(X, newshape=(1, 2))])
        self.y_train = np.concatenate([self.y_train, np.reshape(Y, newshape=(1, 2))])

        if estimate:
            # initialize and train the edward model
            N, D = self.x_train.shape
            if not self.initialized:
                self._initialize_model(N)
                self.initialized = True
            self._train(self.x_train, self.y_train)

            # cache parameter samples for prediction

            # posterior predictive distribution
            y_post = ed.copy(self.y, {self.W_0: self.qW_0, self.b_0: self.qb})

            # try sampling from the posterior
            self.w_samples = self.W_0.sample(self.n_samples).eval()
            self.b_samples = self.b_0.sample(self.n_samples).eval()

    def log_likelihood(self, X, Y, Sigma):
        """
        Calculate a normal likelihood

        Parameters
        ----------
        X: np.array length D
            starting state vector

        Y: np.array length D
            sucessor state vector

        Sigma: DxD array
            covariance matrix for normal likelihood

        Returns
        -------

        ll: float
            log likelihood
        """

        Y_hat = self.predict(X)

        n = Y_hat.shape[0]  # number of samples
        LL = 0  # initialize log likelihood at zero

        for ii in range(n):
            # print multivariate_normal.pdf(Y - Y_hat[ii, :], mean=np.zeros(self.D), cov=Sigma)

            LL += np.log(multivariate_normal.pdf(Y - Y_hat[ii, :], mean=np.zeros(self.D), cov=Sigma))

        return LL / n

    def predict(self, X):
        """
        Parameters
        ----------

        X: np.array, length D
            state vector

        Returns
        -------

        Y_hat: np.array, n_samples x D
            sample of predicted successors states
        """

        # reshape the scene vector
        n = self.w_samples.shape[0]  # number of samples stored
        X_test = np.tile(X, (n, 1))
        X_test = np.reshape(X_test, newshape=(n, 1, self.D))
        Y_hat = np.reshape(np.matmul(X_test, self.w_samples), newshape=(n, self.D)) + self.b_samples

        return np.reshape(Y_hat, newshape=(n, 1, self.D))


class EdwardModel(EventModel):
    def __init__(self, D):
        """
        Parameters:
        -----------

        D: int
            dimensions of the data set

        n_samples: int
            number of samples to draw of the posterior

        """
        EventModel.__init__(self, D)
        self.x_train = np.zeros((0, self.D))  # initialize empty arrays
        self.y_train = np.zeros((0, self.D))

        # initialize samples, for untrained (new) models, predict X'_hat = X
        self.isInitialized = False

    def _estimate(self, x_train, y_train):
        """
        Parameters
        ----------

        x_train: NxD array
            N exemplars of D-dimensional states

        y_train: NxD array
            N exemplars of D-dimensional successor states

        n_samples: int
            number of samples to use in inference
        """
        N, D = x_train.shape

        # Define the neural network
        self.X_ph = tf.placeholder(tf.float32, [None, D])

        def neural_network(X):
            """ Neural network with one hidden layer, outputs location and scale parameter
            for the event model:
            mean, location = neural_network(X)
            """
            hidden = slim.fully_connected(X, D * D, activation_fn=None)
            loc = slim.fully_connected(hidden, D, activation_fn=None)
            scale = slim.fully_connected(hidden, D, activation_fn=tf.nn.softplus)
            return loc, scale

        loc, scale = neural_network(self.X_ph)
        self.Y = ed.models.MultivariateNormalDiag(loc=loc, scale_diag=scale)

        # inference = ed.KLqp(data={self.X_ph: x_train, self.Y: y_train})  # variational inference
        inference = ed.MAP(data={self.X_ph: x_train, self.Y: y_train})  # MAP approximation
        inference.run()

        self.sess = ed.get_session()

    def update(self, X, Y, estimate=True):
        """
        Parameters
        ----------

        X: np.array, length D
            observed state vector

        Y: np.array, length D
            observed successor state vector

        estimate: boolean
            estimate the network parameters?
        """
        self.x_train = np.concatenate([self.x_train, np.reshape(X, newshape=(1, 2))])
        self.y_train = np.concatenate([self.y_train, np.reshape(Y, newshape=(1, 2))])

        if estimate:
            # initialize and train the edward model

            self._estimate(self.x_train, self.y_train)
            self.isInitialized = True

    def log_likelihood(self, X, Y, Sigma):
        """
        Calculate a normal likelihood

        Parameters
        ----------
        X: np.array length D
            starting state vector

        Y: np.array length D
            sucessor state vector

        Sigma: DxD array
            covariance matrix for normal likelihood

        Returns
        -------

        ll: float
            log likelihood
        """
        Y_hat = self.predict(X)
        LL = np.log(multivariate_normal.pdf(Y - Y_hat, mean=np.zeros(self.D), cov=Sigma))

        return LL

    def predict(self, X):
        """
        Parameters
        ----------

        X: np.array, length D
            state vector

        Returns
        -------

        Y_hat: np.array, n_samples x D
            sample of predicted successors states
        """
        if np.array(X).ndim == 1:
            N = 1
            D = np.shape(X)[0]
        else:
            N, D = np.shape(X)

        if self.isInitialized:
            Y_hat = self.sess.run(self.Y, feed_dict={self.X_ph: np.reshape(X, newshape=(N, D))})

            return np.reshape(Y_hat, newshape=(N, 1, self.D))
        else:
            return X

    def close_session(self):
        """
        Close out the tensorflow sessions
        :return:
        """
        self.sess.close()


class EdwardNN(EdwardModel):

    def _estimate(self, x_train, y_train):
        """
        Parameters
        ----------

        x_train: NxD array
            N exemplars of D-dimensional states

        y_train: NxD array
            N exemplars of D-dimensional successor states

        n_samples: int
            number of samples to use in inference
        """
        N, D = x_train.shape

        # Define the neural network
        self.X_ph = tf.placeholder(tf.float32, [None, D])

        def neural_network(X):
            """ Neural network with one hidden layer, outputs location and scale parameter
            for the event model:
            mean, location = neural_network(X)
            """
            hidden = slim.fully_connected(X, D * D, activation_fn=None)
            loc = slim.fully_connected(hidden, D, activation_fn=None)
            scale = slim.fully_connected(hidden, D, activation_fn=tf.nn.softplus)
            return loc, scale

        loc, scale = neural_network(self.X_ph)
        self.Y = ed.models.MultivariateNormalDiag(loc=loc, scale_diag=scale)

        # inference = ed.KLqp(data={self.X_ph: x_train, self.Y: y_train})  # variational inference
        inference = ed.MAP(data={self.X_ph: x_train, self.Y: y_train})  # MAP approximation
        inference.run()

        self.sess = ed.get_session()


class EdwardNN0(EdwardModel):

    def _estimate(self, x_train, y_train):
        """
        Parameters
        ----------

        x_train: NxD array
            N exemplars of D-dimensional states

        y_train: NxD array
            N exemplars of D-dimensional successor states

        n_samples: int
            number of samples to use in inference
        """
        N, D = x_train.shape

        # Define the neural network
        self.X_ph = tf.placeholder(tf.float32, [None, D])

        def neural_network(X):
            """ Neural network with one hidden layer, outputs location and scale parameter
            for the event model:
            mean, location = neural_network(X)
            """
            loc = slim.fully_connected(X, D, activation_fn=None)
            scale = slim.fully_connected(X, D, activation_fn=tf.nn.softplus)
            return loc, scale

        loc, scale = neural_network(self.X_ph)
        self.Y = ed.models.MultivariateNormalDiag(loc=loc, scale_diag=scale)

        # inference = ed.KLqp(data={self.X_ph: x_train, self.Y: y_train})  # variational inference
        inference = ed.MAP(data={self.X_ph: x_train, self.Y: y_train})  # MAP approximation
        inference.run()

        self.sess = ed.get_session()


class EdwardRNN(EdwardModel):

    def _estimate(self, x_train, y_train):
        """
        Parameters
        ----------

        x_train: NxD array
            N exemplars of D-dimensional states

        y_train: NxD array
            N exemplars of D-dimensional successor states

        n_samples: int
            number of samples to use in inference
        """
        N, D = x_train.shape

        # Define the neural network
        self.X_ph = tf.placeholder(tf.float32, [None, D])

        def neural_network(X):
            """ Neural network with one hidden layer, outputs location and scale parameter
            for the event model:
            mean, location = neural_network(X)
            """
            h = rnn.BasicRNNCell(X, D * D, activation_fn=None)
            loc = slim.fully_connected(h, D, activation_fn=None)
            scale = slim.fully_connected(h, D, activation_fn=tf.nn.softplus)
            return loc, scale

        loc, scale = neural_network(self.X_ph)
        self.Y = ed.models.MultivariateNormalDiag(loc=loc, scale_diag=scale)

        # inference = ed.KLqp(data={self.X_ph: x_train, self.Y: y_train})  # variational inference
        inference = ed.MAP(data={self.X_ph: x_train, self.Y: y_train})  # MAP approximation
        inference.run()

        self.sess = ed.get_session()

class EdwardNN0wPrior(EdwardModel):

    def _estimate(self, x_train, y_train):
        """
        Parameters
        ----------

        x_train: NxD array
            N exemplars of D-dimensional states

        y_train: NxD array
            N exemplars of D-dimensional successor states

        n_samples: int
            number of samples to use in inference
        """
        N, D = x_train.shape

        # Define the neural network
        self.X_ph = tf.placeholder(tf.float32, [None, D])

        # define the parameters of the neural network
        W_0 = ed.models.Normal(loc=tf.zeros([D, D]), scale=tf.ones([D, D]), name="W_0")
        b_0 = ed.models.Normal(loc=tf.zeros(D), scale=tf.ones(D), name="b_0")

        W_1 = ed.models.Normal(loc=tf.zeros([D, D]), scale=tf.ones([D, D]), name="W_1")
        b_1 = ed.models.Normal(loc=tf.zeros(D), scale=tf.ones(D), name="b_1")

        # # used for inference
        qW_0 = ed.models.Normal(loc=tf.Variable(tf.random_normal([D, D]), name="loc"),
                           scale=tf.nn.softplus(tf.Variable(tf.random_normal([D, D]), name="scale")))

        qb_0 = ed.models.Normal(loc=tf.Variable(tf.random_normal([D])),
                         scale=tf.nn.softplus(tf.Variable(tf.random_normal([D]))))

        qW_1 = ed.models.Normal(loc=tf.Variable(tf.random_normal([D, D]), name="loc1"),
                           scale=tf.nn.softplus(tf.Variable(tf.random_normal([D, D]), name="scale1")))

        qb_1 = ed.models.Normal(loc=tf.Variable(tf.random_normal([D])),
                         scale=tf.nn.softplus(tf.Variable(tf.random_normal([D]))))

        def neural_network(X):
            """ Neural network with one hidden layer, outputs location and scale parameter
            for the event model:
            mean, location = neural_network(X)
            """
            loc = tf.matmul(X, W_0) + b_0
            scale = tf.nn.softplus(tf.matmul(X, W_1) + b_1)
            return loc, scale

        loc, scale = neural_network(self.X_ph)
        self.Y = ed.models.MultivariateNormalDiag(loc=loc, scale_diag=scale)

        # inference = ed.KLqp(data={self.X_ph: x_train, self.Y: y_train})  # variational inference
        inference = ed.KLqp(
            {W_0: qW_0, b_0: qb_0, W_1: qW_1, b_1: qb_1},
            data={self.X_ph: x_train, self.Y: y_train}
        )
        # inference.run(n_samples=5, n_iter=1000)
        # inference.run(n_samples=n_samples)
        # inference = ed.MAP(data={self.X_ph: x_train, self.Y: y_train})  # MAP approximation
        inference.run()

        self.sess = ed.get_session()


class LinearDynamicSystem(EventModel):
    def __init__(self, D, eta=0.01):
        """
        Parameters
        ----------
        D: int
            dimensions of the vector space of interest

        eta: float
            learning rate
        """
        EventModel.__init__(self, D)
        self.beta = np.zeros(D).flatten()
        self.W = np.eye(D).flatten()
        self.eta = eta

    def predict(self, X):
        """
        Parameters
        ----------
        X: np.array of length D
            vector at time t

        Returns
        -------
        Y_hat: np.array of length D
            prediction of vector at time t+1
        """
        Y_hat = self.beta + np.matmul(X, np.reshape(self.W, (self.D, self.D)))
        return Y_hat

    def log_likelihood(self, X, Y, Sigma):
        """
        Calculate a normal likelihood

        Parameters
        ----------
        X: np.array length D
            starting state vector

        Y: np.array length D
            sucessor state vector

        Sigma: DxD array
            covariance matrix for normal likelihood

        Returns
        -------

        ll: float
            log likelihood
        """

        Y_hat = self.predict(X)
        return np.log(multivariate_normal.pdf(Y - Y_hat, mean=np.zeros(self.D), cov=Sigma))

    def update(self, X, Y):
        """
        Parameters
        ----------

        X: np.array, length D
            observed state vector

        Y: np.array, length D
            observed sucessor state vector
        """
        Y_hat = self.predict(X)

        # needed for updating logic
        dXdb = np.eye(self.D)
        dXdW = np.tile((np.tile(X, (1, self.D))), (self.D, 1))
        g = np.concatenate([dXdb, dXdW], axis=1)

        # vectorize the parameters
        theta = np.concatenate([self.beta, self.W.flatten()])
        theta += self.eta * np.matmul(Y - Y_hat, g)

        # store the updated parameters
        self.beta = theta[:np.shape(X)[0]]
        self.W = theta[np.shape(X)[0]:]
