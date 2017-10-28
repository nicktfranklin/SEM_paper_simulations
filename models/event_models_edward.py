import edward as ed
import tensorflow as tf
from event_models import EventModel
from tensorflow.contrib import slim, rnn
from scipy.stats import multivariate_normal
import numpy as np


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

        Y_hat = self.predict_next(X)

        n = Y_hat.shape[0]  # number of samples
        LL = 0  # initialize log likelihood at zero

        for ii in range(n):
            # print multivariate_normal.pdf(Y - Y_hat[ii, :], mean=np.zeros(self.D), cov=Sigma)

            LL += np.log(multivariate_normal.pdf(Y - Y_hat[ii, :], mean=np.zeros(self.D), cov=Sigma))

        return LL / n

    def predict_next(self, X):
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
        Y_hat = self.predict_next(X)
        LL = np.log(multivariate_normal.pdf(Y - Y_hat, mean=np.zeros(self.D), cov=Sigma))

        return LL

    def predict_next(self, X):
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

    def predict_next(self, X):
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

        Y_hat = self.predict_next(X)
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
        Y_hat = self.predict_next(X)

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
