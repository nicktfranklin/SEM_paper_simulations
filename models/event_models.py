import tensorflow as tf
import numpy as np
from utils import unroll_data
from keras.models import Sequential
from keras.layers import Dense, Activation, SimpleRNN
from keras import optimizers
from scipy.stats import multivariate_normal



class EventModel(object):

    """ this is the base clase of the event model """

    def __init__(self, D):
        self.D = D

    def update(self, X, Y):
        """
        Parameters
        ----------
        X: NxD array-like data of inputs

        y: NxD array-like data of outputs

        Returns
        -------
        None

        """
        pass

    def predict(self, X):
        """
        Parameters
        ----------
        X: 1xD array-like data of inputs

        Returns
        -------
        y: 1xD array of prediction vectors

        """
        return np.copy(X)

    def log_likelihood(self, X, Y, Sigma):
        """
        Parameters
        ----------
        X: NxD array-like
            observed inputs

        y: NxD array-like
            observed outputs

        Sigma: float
            variance of diagonal normal distribution


        Returns
        -------

        LL: float
            log likelihood of observed data N(Y|X, sigma * I)


        """
        return np.log(multivariate_normal.pdf(X - Y, mean=np.zeros(self.D), cov=Sigma))

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
        """
        Parameters
        ----------
        X: 1xD array-like data of inputs

        Returns
        -------
        y: 1xD array of prediction vectors

        """

        if self.is_initialized:
            if np.ndim(X) == 1:
                N = 1
            else:
                N, _ = np.shape(X)
            return self.model.predict(np.reshape(X, newshape=(N, self.D)))
        else:
            return X

    def log_likelihood(self, X, Y, Sigma):

        """
        Parameters
        ----------
        X: NxD array-like
            observed inputs

        y: NxD array-like
            observed outputs

        Sigma: float
            variance of diagonal normal distribution


        Returns
        -------

        LL: float
            log likelihood of observed data N(Y|X, sigma * I)


        """

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

    def __init__(self, D, t=5, n_hidden1=10, n_hidden2=10, hidden_act1=None, hidden_act2='tanh', sgd_kwargs=None,
                 n_batch=100, aug_noise=0.01):
        KerasLDS.__init__(self, D, sgd_kwargs=sgd_kwargs)
        self.t = t
        self.n_batch = n_batch
        self.noise = aug_noise
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2
        self.hidden_act1 = hidden_act1
        self.hidden_act2 = hidden_act2

    def _estimate(self):

        self.model = Sequential()
        self.model.add(SimpleRNN(self.n_hidden1, input_shape=(self.t, self.n_hidden1), activation=self.hidden_act1))
        self.model.add(Dense(self.n_hidden2, activation=self.hidden_act2))
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



