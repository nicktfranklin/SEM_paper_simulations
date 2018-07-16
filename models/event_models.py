import tensorflow as tf
import numpy as np
from utils import unroll_data
from keras.models import Sequential
from keras.layers import Dense, Activation, SimpleRNN, GRU, Dropout, LSTM
from keras import optimizers
from keras import regularizers
from scipy.stats import multivariate_normal as mvnormal


class EventModel(object):
    """ this is the base clase of the event model """

    def __init__(self, D, beta=None):
        self.D = D
        self.f_is_trained = False
        self.f0_is_trained = False
        if beta is None:
            self.beta = 0.1 * D
        else:
            self.beta = beta

        # initilaize the covariance matrix
        self.Sigma = np.eye(D) * self.beta

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

    def predict_next(self, X):
        """
        wrapper for the prediction function that changes the prediction to the identity function
        for untrained models (this is an initialization technique)

        """
        if not self.f_is_trained:
            return np.copy(X)

        return self._predict_next(X)

    def likelihood_f0(self, Y):

        # predict the inital point
        Y_hat = self.predict_f0()

        # return the probability
        return mvnormal.logpdf(Y - Y_hat, mean=np.zeros(self.D), cov=self.Sigma)

    def likelihood_next(self, X, Y):
        Y_hat = self.predict_next(X)
        return mvnormal.logpdf(Y - Y_hat, mean=np.zeros(self.D), cov=self.Sigma)


    # def lik

    def _predict_next(self, X):
        """
        Internal function

        Parameters
        ----------
        X: 1xD array-like data of inputs

        Returns
        -------
        y: 1xD array of prediction vectors

        """
        return np.copy(X)

    def predict_f0(self):
        """
        wrapper for the prediction function that changes the prediction to the identity function
        for untrained models (this is an initialization technique)

        """
        if not self.f0_is_trained:
            return np.zeros(self.D)

        return self._predict_f0()

    def _predict_f0(self):
        return np.zeros(self.D)

    def update_f0(self, Y):
        pass

    def close(self):
        pass

    def new_token(self):
        pass

    def predict_next_generative(self, X):
        X0 = np.reshape(unroll_data(X, self.t)[-1, :, :], (1, self.t, self.D))
        return self._predict_next(X0)

    def run_generative(self, n_steps, initial_point=None):
        if initial_point is None:
            x_gen = self._predict_f0()
        else:
            x_gen = np.reshape(initial_point, (1, self.D))
        for ii in range(1, n_steps):
            x_gen = np.concatenate([x_gen, self.predict_next_generative(x_gen[:ii, :])])
        return x_gen


class LinearDynamicSystem(EventModel):
    def __init__(self, D, eta=0.01, beta=None):
        """
        Parameters
        ----------
        D: int
            dimensions of the vector space of interest

        eta: float
            learning rate
        """
        EventModel.__init__(self, D, beta)
        self.beta = np.zeros(D)
        self.W = np.eye(D)
        self.eta = eta
        self.f0 = np.zeros(D)

    def _predict_next(self, X):
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

        # these models are not recurssive
        if X.ndim > 1:
            X0 = X[-1, :]
        else:
            X0 = X

        Y_hat = self.beta + np.matmul(X0, self.W)
        return Y_hat

    def predict_next_generative(self, X):
        return self._predict_next(X)

    def update(self, X, Y):
        """
        Parameters
        ----------

        X: np.array, length D
            observed state vector

        Y: np.array, length D
            observed sucessor state vector
        """
        Y_hat = np.reshape(self._predict_next(X), (Y.shape[0]))

        # compute gradient of log likelihood w.r.t. parameters
        db = self.eta * (Y - Y_hat)
        dW = self.eta * np.matmul(X.reshape(self.D, 1), (Y - Y_hat).reshape(1, self.D))

        # store the updated parameters
        self.beta += db
        self.W += dW
        self.f_is_trained = True

    def _predict_f0(self):
        return self.f0

    def update_f0(self, Y):
        self.f0 += self.eta * (Y - self.f0)
        self.f0_is_trained = True


class KerasLDS(EventModel):

    def __init__(self, D, optimizer='adam', n_epochs=50, init_model=True, beta=None, kernel_initializer='glorot_uniform'):
        EventModel.__init__(self, D, beta)
        self.x_train = np.zeros((0, self.D))  # initialize empty arrays
        self.y_train = np.zeros((0, self.D))
        if optimizer is None:
            sgd_kwargs = {
                'nesterov': True,
                'lr': 0.1,
                'momentum': 0.5,
                'decay': 0.0001
            }
            optimizer = optimizers.SGD(**sgd_kwargs)

        self.compile_opts = dict(optimizer=optimizer, loss='mean_squared_error')
        self.kernel_initializer = kernel_initializer
        self.n_epochs = n_epochs

        self.is_visited = False  # governs the special case of model's first prediction (i.e. with no experience)

        if init_model:
            self._init_model()

    def _init_model(self):
        self.sess = tf.Session()

        N, D = self.x_train.shape

        self.model = Sequential([
            Dense(D, input_shape=(D,), use_bias=True, kernel_initializer=self.kernel_initializer),
            Activation('linear')
        ])

        self.model.compile(**self.compile_opts)

    def reestimate(self):
        self._init_model()
        self.estimate()

    def append_observation(self, X, Y):
        if np.ndim(X) == 1:
            N = 1
        else:
            N, _ = np.shape(X)
        self.x_train = np.concatenate([self.x_train, np.reshape(X, newshape=(N, self.D))])
        self.y_train = np.concatenate([self.y_train, np.reshape(Y, newshape=(N, self.D))])

    def estimate(self):
        self.model.fit(self.x_train, self.y_train, verbose=0, epochs=self.n_epochs, shuffle=True)

    def update(self, X, Y):
        if np.ndim(X) == 1:
            N = 1
        else:
            N, _ = np.shape(X)
        self.x_train = np.concatenate([self.x_train, np.reshape(X, newshape=(N, self.D))])
        self.y_train = np.concatenate([self.y_train, np.reshape(Y, newshape=(N, self.D))])
        self.estimate()
        self.f_is_trained = True

    def _predict_next(self, X):
        """
        Parameters
        ----------
        X: 1xD array-like data of inputs

        Returns
        -------
        y: 1xD array of prediction vectors

        """
        if X.ndim > 1:
            X0 = X[-1, :]
        else:
            X0 = X

        return self.model.predict(np.reshape(X0, newshape=(1, self.D)))

    def predict_next_generative(self, X):
        # the LDS is a markov model, so these functions are the same
        return self._predict_next(X)

    def _predict_f0(self):
        return self._predict_next(np.zeros(self.D))

    def update_f0(self, Y):
        self.update(np.zeros(self.D), Y)
        self.f0_is_trained = True

class KerasLDS_b(KerasLDS):

    def __init__(self, D, optimizer='adam', n_epochs=50, init_model=True, beta=None):
        KerasLDS.__init__(self, D, optimizer=optimizer, init_model=init_model, beta=beta, n_epochs=n_epochs)
        # self.pe_dot = np.zeros(0)
        self.pe = np.zeros((0, D))

    def update(self, X, Y):
        super(KerasLDS_b, self).update(X, Y)

        # cache a prediction error term
        Y_hat = self._predict_next(X)

        self.pe = np.concatenate([self.pe, Y - Y_hat])
        if np.shape(self.pe)[0] > 1:
            # self.beta = np.var(self.pe, axis=0)
            # self.Sigma = np.eye(self.D) * self.beta

            # useing the MLE can be too sensitive, so here we'll weight the MLE by there prior acording to a
            # decending function
            n = np.shape(self.pe)[0]
            w = 1. / (1+ np.log(n))
            self.Sigma = np.eye(self.D) * ((1.0 - w) * np.var(self.pe, axis=0) + np.ones(self.D) * w * self.beta)


class KerasMultiLayerPerceptron(KerasLDS):

    def __init__(self, D, n_hidden=None, hidden_act='tanh', optimizer='adam', n_epochs=50, l2_regularization=0.00,
                 beta=None):
        KerasLDS.__init__(self, D, optimizer=optimizer, init_model=False, beta=beta)
        if n_hidden is None:
            n_hidden = D
        self.n_hidden = n_hidden
        self.hidden_act = hidden_act
        self.n_epochs = n_epochs
        self.kernel_regularizer = regularizers.l2(l2_regularization)
        self._init_model()

    def _init_model(self):
        N, D = self.x_train.shape
        self.model = Sequential()
        self.model.add(Dense(self.n_hidden, input_shape=(D,), activation=self.hidden_act,
                             kernel_regularizer=self.kernel_regularizer))
        self.model.add(Dense(D, activation='linear', kernel_regularizer=self.kernel_regularizer))
        self.model.compile(**self.compile_opts)


class KerasSimpleRNN(KerasLDS):

    # RNN which is initialized once and then trained using stochastic gradient descent
    # i.e. each new scene is a single example batch of size 1

    def __init__(self, D, t=3, n_hidden1=None, n_hidden2=None, hidden_act1='relu', hidden_act2='relu',
                 optimizer='adam', n_epochs=50, dropout=0.10, l2_regularization=0.00, batch_size=32,
                 kernel_initializer='glorot_uniform', beta=None):
        #
        # D = dimension of single input / output example
        # t = number of time steps to unroll back in time for the recurrent layer
        # n_hidden1 = # of nodes in first hidden layer
        # n_hidden2 = # of nodes in second hidden layer
        # hidden_act1 = activation f'n of first hidden layer
        # hidden_act2 = activation f'n of second hidden layer
        # sgd_kwargs = arguments for the stochastic gradient descent algorithm
        # n_epochs = how many gradient descent steps to perform for each training batch
        # dropout = what fraction of nodes to drop out during training (to prevent overfitting)

        KerasLDS.__init__(self, D, optimizer=optimizer, init_model=False, beta=beta)

        self.t = t
        self.n_epochs = n_epochs
        if n_hidden1 is None:
            self.n_hidden1 = D
        else:
            self.n_hidden1 = n_hidden1
        if n_hidden2 is None:
            self.n_hidden2 = D
        else:
            self.n_hidden2 = n_hidden2
        self.hidden_act1 = hidden_act1
        self.hidden_act2 = hidden_act2
        self.D = D
        self.dropout = dropout
        self.kernel_regularizer = regularizers.l2(l2_regularization)
        self.kernel_initializer = kernel_initializer

        # list of clusters of scenes:
        # each element of list = history of scenes for given cluster
        # history = N x D tensor, N = # of scenes in cluster, D = dimension of single scene
        #
        self.x_history = [np.zeros((0, self.D))]
        self.y_history = [np.zeros((0, self.D))]

        self._init_model()
        self.batch_size = batch_size

    # initialize model once so we can then update it online
    #
    def _init_model(self):
        self.sess = tf.Session()

        self.model = Sequential()
        # input_shape[0] = timesteps; we pass the last self.t examples for train the hidden layer
        # input_shape[1] = input_dim; each example is a self.D-dimensional vector
        self.model.add(SimpleRNN(self.n_hidden1, input_shape=(self.t, self.D), activation=self.hidden_act1,
                                 kernel_initializer=self.kernel_initializer))
        self.model.add(Dense(self.n_hidden2, activation=self.hidden_act2, kernel_regularizer=self.kernel_regularizer,
                             kernel_initializer=self.kernel_initializer))
        self.model.add(Dropout(self.dropout))
        self.model.add(Dense(self.D, activation=None, kernel_regularizer=self.kernel_regularizer,
                             kernel_initializer=self.kernel_initializer))
        self.model.compile(**self.compile_opts)

    # concatenate current example with the history of the last t-1 examples
    # this is for the recurrent layer
    #
    def _unroll(self, x_example):
        x_train = np.concatenate([self.x_history[-1][-(self.t - 1):, :], x_example], axis=0)
        x_train = np.concatenate([np.zeros((self.t - x_train.shape[0], self.D)), x_train], axis=0)
        x_train = x_train.reshape((1, self.t, self.D))
        return x_train

    # train on a single example
    #
    def update(self, X, Y, update_estimate=True):
        if X.ndim > 1:
            X = X[-1, :]  # only consider last example
        assert X.ndim == 1
        assert X.shape[0] == self.D
        assert Y.ndim == 1
        assert Y.shape[0] == self.D

        x_example = X.reshape((1, self.D))
        y_example = Y.reshape((1, self.D))

        # concatenate the training example to the active event token

        self.x_history[-1] = np.concatenate([self.x_history[-1], x_example], axis=0)
        self.y_history[-1] = np.concatenate([self.y_history[-1], y_example], axis=0)

        if update_estimate:
            self.batch()

        self.f_is_trained = True

    # predict a single example
    def _predict_next(self, X):
        # Note: this function predicts the next conditioned on the training data the model has seen

        if X.ndim > 1:
            X = X[-1, :]  # only consider last example
        assert np.ndim(X) == 1
        assert X.shape[0] == self.D

        x_test = X.reshape((1, self.D))

        # concatenate current example with history of last t-1 examples
        # this is for the recurrent part of the network
        x_test = self._unroll(x_test)

        return self.model.predict(x_test)

    def predict_next_generative(self, X):
        X0 = np.reshape(unroll_data(X, self.t)[-1, :, :], (1, self.t, self.D))
        return self.model.predict(X0)

    def _predict_f0(self):
        return self.predict_next_generative(np.zeros(self.D))

    # create a new cluster of scenes
    def new_token(self):
        if len(self.x_history) == 1 and self.x_history[0].shape[0] == 0:
            # special case for the first cluster which is already created
            return
        self.x_history.append(np.zeros((0, self.D)))
        self.y_history.append(np.zeros((0, self.D)))

    # optional: run batch gradient descent on all past event clusters
    def batch(self):
        # run batch gradient descent on all of the past events!
        for _ in range(self.n_epochs):

            # draw a set of training examples from the history
            x_batch = []
            y_batch = []
            for _ in range(self.batch_size):
                # draw a random cluster for the history
                clust_id = np.random.randint(len(self.x_history))
                x_history = self.x_history[clust_id]
                y_history = self.y_history[clust_id]

                t = np.random.randint(len(x_history))

                x_batch.append(np.reshape(
                    unroll_data(x_history[max(t - self.t, 0):t + 1, :], self.t)[-1, :, :], (1, self.t, self.D)
                ))
                y_batch.append(y_history[t, :])

            x_batch = np.reshape(x_batch, (self.batch_size, self.t, self.D))
            y_batch = np.reshape(y_batch, (self.batch_size, self.D))
            self.model.train_on_batch(x_batch, y_batch)

    def batch_last_clust(self):
        # draw a set of training examples from the history

        # pull the last cluster
        x_batch = self.x_history[-1]
        y_batch = self.y_history[-1]

        x_batch = unroll_data(x_batch, t=np.shape(x_batch)[0])

        self.model.train_on_batch(x_batch, y_batch)

class KerasElmanSRN(KerasSimpleRNN):
    def _init_model(self):
        self.sess = tf.Session()
        self.model = Sequential()
        self.model.add(SimpleRNN(self.D, input_shape=(self.t, self.D),
                                 activation=None, kernel_initializer=self.kernel_initializer))
        self.model.compile(**self.compile_opts)


class KerasGRU0(KerasSimpleRNN):
    def _init_model(self):
        self.sess = tf.Session()
        self.model = Sequential()
        self.model.add(GRU(self.n_hidden1, input_shape=(self.t, self.D), activation=self.hidden_act1,
                               kernel_initializer=self.kernel_initializer))
        self.model.add(Dropout(self.dropout))
        self.model.add(Dense(self.D, activation=None, kernel_regularizer=None))
        self.model.compile(**self.compile_opts)

class KerasGRU(KerasSimpleRNN):
    def _init_model(self):
        self.sess = tf.Session()

        # input_shape[0] = timesteps; we pass the last self.t examples for train the hidden layer
        # input_shape[1] = input_dim; each example is a self.D-dimensional vector
        self.model = Sequential()
        self.model.add(GRU(self.n_hidden1, input_shape=(self.t, self.D), activation=self.hidden_act1))
        self.model.add(Dense(self.n_hidden2, activation=self.hidden_act2, kernel_regularizer=self.kernel_regularizer))
        self.model.add(Dropout(self.dropout))
        self.model.add(Dense(self.D, activation=None, kernel_regularizer=self.kernel_regularizer))
        self.model.compile(**self.compile_opts)

class KerasGRU_stacked(KerasSimpleRNN):
    def _init_model(self):
        self.sess = tf.Session()

        # input_shape[0] = timesteps; we pass the last self.t examples for train the hidden layer
        # input_shape[1] = input_dim; each example is a self.D-dimensional vector
        self.model = Sequential()
        self.model.add(GRU(self.n_hidden1, input_shape=(self.t, self.D), activation=self.hidden_act1,
                           return_sequences=True))
        self.model.add(GRU(self.n_hidden1, activation=self.hidden_act1))
        self.model.add(Dense(self.n_hidden2, activation=self.hidden_act2, kernel_regularizer=self.kernel_regularizer))
        self.model.add(Dropout(self.dropout))
        self.model.add(Dense(self.D, activation=None, kernel_regularizer=self.kernel_regularizer))
        self.model.compile(**self.compile_opts)

class KerasLSTM(KerasSimpleRNN):
    def _init_model(self):
        self.sess = tf.Session()

        # input_shape[0] = timesteps; we pass the last self.t examples for train the hidden layer
        # input_shape[1] = input_dim; each example is a self.D-dimensional vector
        self.model = Sequential()
        self.model.add(LSTM(self.n_hidden1, input_shape=(self.t, self.D), activation=self.hidden_act1))
        self.model.add(Dense(self.n_hidden2, activation=self.hidden_act2, kernel_regularizer=self.kernel_regularizer))
        self.model.add(Dropout(self.dropout))
        self.model.add(Dense(self.D, activation=None, kernel_regularizer=self.kernel_regularizer))
        self.model.compile(**self.compile_opts)
