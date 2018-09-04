import tensorflow as tf
import numpy as np
from utils import unroll_data
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, SimpleRNN, GRU, Dropout, LSTM, LeakyReLU, BatchNormalization
from keras import regularizers
from keras.optimizers import Adam
from scipy.stats import multivariate_normal as mvnormal

print("TensorFlow Version: {}".format(tf.__version__))
print("Keras      Version: {}".format(keras.__version__))


def map_variance(samples, df0, scale0):
    """
    This estimator assumes an scaled inverse-chi squared prior over the
    variance and a Gaussian likelihood. The parameters d and scale
    of the internal function parameterize the posterior of the variance.
    Taken from Bayesian Data Analysis, ch2 (Gelman)

    samples: N length array or NxD array
    df0: prior degrees of freedom
    scale0: prior scale parameter
    mu: (optional) mean function

    returns: float or d-length array, mode of the posterior
    """
    if np.ndim(samples) > 1:
        n, d = np.shape(samples)
    else:
        n = np.shape(samples)[0]
        d = 1

    v = np.var(samples, axis=0)
    df = df0 + n
    scale = (df0 * scale0 + n * v) / df
    return df * scale / (df * 2)


class EventModel(object):
    """ this is the base clase of the event model """

    def __init__(self, d):
        """

        :param d: dimensions of the input space
        :param beta: (float) or D length np.array -- event noise
        """
        self.d = d
        self.f_is_trained = False
        self.f0_is_trained = False

        # initialize the covariance matrix
        self.Sigma = np.eye(d) * 0.1

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

    def log_likelihood_f0(self, Y):

        # predict the initial point
        Y_hat = self.predict_f0()

        # return the probability
        return mvnormal.logpdf(Y.reshape(-1), mean=Y_hat.reshape(-1), cov=self.Sigma)

    def log_likelihood_next(self, X, Y):
        Y_hat = self.predict_next(X)
        return mvnormal.logpdf(Y.reshape(-1), mean=Y_hat.reshape(-1), cov=self.Sigma)

    def log_likelihood_sequence(self, X, Y):
        Y_hat = self.predict_next_generative(X)
        return mvnormal.logpdf(Y.reshape(-1), mean=Y_hat.reshape(-1), cov=self.Sigma)

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
            return np.zeros(self.d)

        return self._predict_f0()

    def _predict_f0(self):
        return self._predict_next(np.zeros(self.d))

    def update_f0(self, Y):
        pass

    def close(self):
        pass

    def new_token(self):
        pass

    def predict_next_generative(self, X):
        X0 = np.reshape(unroll_data(X, self.t)[-1, :, :], (1, self.t, self.d))
        return self._predict_next(X0)

    def run_generative(self, n_steps, initial_point=None):
        if initial_point is None:
            x_gen = self._predict_f0()
        else:
            x_gen = np.reshape(initial_point, (1, self.d))
        for ii in range(1, n_steps):
            x_gen = np.concatenate([x_gen, self.predict_next_generative(x_gen[:ii, :])])
        return x_gen


class Gaussian(EventModel):
    def __init__(self, d, var_df0, var_scale0):

        EventModel.__init__(self, d)
        self.mu = np.zeros(d)
        # initialize Sigma with the mode of the prior distribution
        self.Sigma = np.eye(d) * var_df0 * var_scale0 / (var_df0 + 2)
        self.history = np.zeros((0, d))
        self.var_df0 = var_df0
        self.var_scale0 = var_scale0

    def _predict_next(self, X):
        return self.mu

    def update(self, X, Y):
        self.update_f0(Y)
        self.f_is_trained = True

    def _predict_f0(self):
        return self._predict_next(np.zeros(self.d))

    def update_f0(self, Y):
        self.history = np.concatenate([self.history, Y.reshape(1, -1)])
        self.mu = np.mean(self.history, axis=0)

        n = np.shape(self.history)[0]
        if n > 1:
            self.Sigma = np.eye(self.d) * map_variance(self.history, self.var_df0, self.var_scale0)


class GaussianRandomWalk(EventModel):
    def __init__(self, d, var_df0, var_scale0):
        EventModel.__init__(self, d)

        # initialize Sigma with the mode of the prior distribution
        self.Sigma = np.eye(d) * var_df0 * var_scale0 / (var_df0 + 2)
        self.pe = np.zeros((0, d))
        self.var_df0 = var_df0
        self.var_scale0 = var_scale0

    def _predict_next(self, X):
        if X.ndim > 1:
            return np.copy(X[-1, :])
        else:
            return np.copy(X)

    def update(self, X, Y):
        self.pe = np.concatenate([self.pe, (Y - X).reshape(1, -1)])
        if np.shape(self.pe)[0] > 1:
            self.Sigma = np.eye(self.d) * map_variance(self.pe, self.var_df0, self.var_scale0)

    def _predict_f0(self):
        return self._predict_next(np.zeros(self.d))

    def update_f0(self, Y):
        self.pe = np.concatenate([self.pe, np.reshape(Y, (1, -1))])
        self.f_is_trained = True


class LinearDynamicSystem(EventModel):
    def __init__(self, d, var_df0, var_scale0, eta=0.01):
        """
        Parameters
        ----------
        d: int
            dimensions of the vector space of interest

        eta: float
            learning rate
        """
        EventModel.__init__(self, d)
        self.bias = np.zeros(d)
        self.W = np.eye(d)
        self.f0 = np.zeros(d)
        self.n = 0

        # initialize the covariance with the mode of the prior distribution
        self.Sigma = np.eye(d) * var_df0 * var_scale0 / (var_df0 + 2)
        self.history_pe = np.zeros((0, d))
        self.var_df0 = var_df0
        self.var_scale0 = var_scale0
        self.eta = eta

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

        Y_hat = self.bias + np.matmul(X0, self.W)
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

        # calculate eta
        # eta = 1.0 / (self.prior_lr + self.n)
        self.n += 1

        # compute gradient of log likelihood w.r.t. parameters
        pe = Y - Y_hat

        db = self.eta * pe
        dW = self.eta * np.matmul(X.reshape(self.d, 1), pe.reshape(1, -1))

        # update the variance estimate
        self.history_pe = np.concatenate([self.history_pe, pe.reshape(1, -1)])
        if self.n > 1:
            self.Sigma = np.eye(self.d) * map_variance(self.history_pe, self.var_df0, self.var_scale0)

        # store the updated parameters
        self.bias += db
        self.W += dW
        self.f_is_trained = True

    def _predict_f0(self):
        return self.f0

    def update_f0(self, Y):
        self.update(np.zeros(np.shape(Y)), Y)
        self.f0 = self.bias


class KerasLDS(EventModel):

    def __init__(self, d, var_df0, var_scale0, optimizer=None, n_epochs=100, init_model=True,
                 kernel_initializer='glorot_uniform', l2_regularization=0.00):
        EventModel.__init__(self, d)
        self.x_train = np.zeros((0, self.d))  # initialize empty arrays
        self.y_train = np.zeros((0, self.d))

        if optimizer is None:
            optimizer = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

        self.compile_opts = dict(optimizer=optimizer, loss='mean_squared_error')
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = regularizers.l2(l2_regularization)
        self.n_epochs = n_epochs
        self.var_df0 = var_df0
        self.var_scale0 = var_scale0
        self.pe = np.zeros((0, d))
        self.d = d

        # initialize the covariance with the mode of the prior distribution
        self.Sigma = np.eye(d) * var_df0 * var_scale0 / (var_df0 + 2)

        self.is_visited = False  # governs the special case of model's first prediction (i.e. with no experience)

        # switch for inheritance -- don't want to init the model for sub-classes
        if init_model:
            self._init_model()

    def _init_model(self):
        self.sess = tf.Session()

        N, D = self.x_train.shape

        self.model = Sequential([
            Dense(D, input_shape=(D,), use_bias=True, kernel_initializer=self.kernel_initializer,
                  kernel_regularizer=self.kernel_regularizer),
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
        self.x_train = np.concatenate([self.x_train, np.reshape(X, newshape=(N, self.d))])
        self.y_train = np.concatenate([self.y_train, np.reshape(Y, newshape=(N, self.d))])

    def estimate(self):
        self.model.fit(self.x_train, self.y_train, verbose=0, epochs=self.n_epochs, shuffle=True)

    def update(self, X, Y):
        if np.ndim(X) == 1:
            N = 1
        else:
            N, _ = np.shape(X)
        self.x_train = np.concatenate([self.x_train, np.reshape(X, newshape=(N, self.d))])
        self.y_train = np.concatenate([self.y_train, np.reshape(Y, newshape=(N, self.d))])
        self.estimate()
        self.f_is_trained = True

        # cache a prediction error term
        Y_hat = self.predict_next(X)

        self.pe = np.concatenate([self.pe, Y - Y_hat])
        if np.shape(self.pe)[0] > 1:
            self.Sigma = np.eye(self.d) * map_variance(self.pe, self.var_df0, self.var_scale0)

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

        return self.model.predict(np.reshape(X0, newshape=(1, self.d)))

    def predict_next_generative(self, X):
        # the LDS is a markov model, so these functions are the same
        return self._predict_next(X)

    def _predict_f0(self):
        return self._predict_next(np.zeros(self.d))

    def update_f0(self, Y):
        self.update(np.zeros(self.d), Y)
        self.f0_is_trained = True


class KerasMultiLayerPerceptron(KerasLDS):

    def __init__(self, d, var_df0, var_scale0, n_hidden=None, hidden_act='tanh',
                 optimizer=None, n_epochs=100, init_model=True, kernel_initializer='glorot_uniform',
                 l2_regularization=0.00, dropout=0.50):
        KerasLDS.__init__(self, d, var_df0, var_scale0, optimizer=optimizer, n_epochs=n_epochs,
                          init_model=False, kernel_initializer=kernel_initializer,
                          l2_regularization=l2_regularization)

        if n_hidden is None:
            n_hidden = d
        self.n_hidden = n_hidden
        self.hidden_act = hidden_act
        self.dropout = dropout

        if init_model:
            self._init_model()

    def _init_model(self):
        N, D = self.x_train.shape
        self.model = Sequential()
        self.model.add(Dense(self.n_hidden, input_shape=(D,), activation=self.hidden_act,
                             kernel_regularizer=self.kernel_regularizer,
                             kernel_initializer=self.kernel_initializer))
        self.model.add(Dropout(self.dropout))
        self.model.add(Dense(D, activation='linear',
                             kernel_regularizer=self.kernel_regularizer,
                             kernel_initializer=self.kernel_initializer))
        self.model.compile(**self.compile_opts)


class KerasMultiLayerPerceptron_BN(KerasMultiLayerPerceptron):
    def _init_model(self):
        N, D = self.x_train.shape
        self.model = Sequential()
        self.model.add(Dense(self.n_hidden, input_shape=(D,), activation=self.hidden_act,
                             kernel_regularizer=self.kernel_regularizer,
                             kernel_initializer=self.kernel_initializer))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(self.dropout))
        self.model.add(Dense(D, activation='linear',
                             kernel_regularizer=self.kernel_regularizer,
                             kernel_initializer=self.kernel_initializer))
        self.model.compile(**self.compile_opts)


class KerasSRN(KerasLDS):

    # RNN which is initialized once and then trained using stochastic gradient descent
    # i.e. each new scene is a single example batch of size 1

    def __init__(self, d, var_df0, var_scale0, t=3,
                 optimizer=None, n_epochs=100, l2_regularization=0.00, batch_size=32,
                 kernel_initializer='glorot_uniform', init_model=True):
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

        KerasLDS.__init__(self, d, var_df0, var_scale0, optimizer=optimizer, n_epochs=n_epochs,
                          init_model=False, kernel_initializer=kernel_initializer,
                          l2_regularization=l2_regularization)

        self.t = t
        self.n_epochs = n_epochs

        # list of clusters of scenes:
        # each element of list = history of scenes for given cluster
        # history = N x D tensor, N = # of scenes in cluster, D = dimension of single scene
        #
        self.x_history = [np.zeros((0, self.d))]
        self.y_history = [np.zeros((0, self.d))]
        self.batch_size = batch_size

        if init_model:
            self._init_model()

    # initialize model once so we can then update it online
    #
    def _init_model(self):
        self.sess = tf.Session()
        self.model = Sequential()
        self.model.add(SimpleRNN(self.d, input_shape=(self.t, self.d),
                                 activation=None, kernel_initializer=self.kernel_initializer,
                                 kernel_regularizer=self.kernel_regularizer))
        self.model.compile(**self.compile_opts)

    # concatenate current example with the history of the last t-1 examples
    # this is for the recurrent layer
    #
    def _unroll(self, x_example):
        x_train = np.concatenate([self.x_history[-1][-(self.t - 1):, :], x_example], axis=0)
        x_train = np.concatenate([np.zeros((self.t - x_train.shape[0], self.d)), x_train], axis=0)
        x_train = x_train.reshape((1, self.t, self.d))
        return x_train

    # train on a single example
    #
    def update(self, X, Y, update_estimate=True):
        if X.ndim > 1:
            X = X[-1, :]  # only consider last example
        assert X.ndim == 1
        assert X.shape[0] == self.d
        assert Y.ndim == 1
        assert Y.shape[0] == self.d

        x_example = X.reshape((1, self.d))
        y_example = Y.reshape((1, self.d))

        # concatenate the training example to the active event token

        self.x_history[-1] = np.concatenate([self.x_history[-1], x_example], axis=0)
        self.y_history[-1] = np.concatenate([self.y_history[-1], y_example], axis=0)

        if update_estimate:
            self.estimate()

        self.f_is_trained = True

        # cache a prediction error term
        Y_hat = self.predict_next(X)

        # update the variance
        self.pe = np.concatenate([self.pe, Y - Y_hat])
        if np.shape(self.pe)[0] > 1:
            self.Sigma = np.eye(self.d) * map_variance(self.pe, self.var_df0, self.var_scale0)

    # predict a single example
    def _predict_next(self, X):
        # Note: this function predicts the next conditioned on the training data the model has seen

        if X.ndim > 1:
            X = X[-1, :]  # only consider last example
        assert np.ndim(X) == 1
        assert X.shape[0] == self.d

        x_test = X.reshape((1, self.d))

        # concatenate current example with history of last t-1 examples
        # this is for the recurrent part of the network
        x_test = self._unroll(x_test)

        return self.model.predict(x_test)

    def predict_next_generative(self, X):
        X0 = np.reshape(unroll_data(X, self.t)[-1, :, :], (1, self.t, self.d))
        return self.model.predict(X0)

    def _predict_f0(self):
        return self.predict_next_generative(np.zeros(self.d))

    # create a new cluster of scenes
    def new_token(self):
        if len(self.x_history) == 1 and self.x_history[0].shape[0] == 0:
            # special case for the first cluster which is already created
            return
        self.x_history.append(np.zeros((0, self.d)))
        self.y_history.append(np.zeros((0, self.d)))

    # optional: run batch gradient descent on all past event clusters
    def estimate(self):
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
                    unroll_data(x_history[max(t - self.t, 0):t + 1, :], self.t)[-1, :, :], (1, self.t, self.d)
                ))
                y_batch.append(y_history[t, :])

            x_batch = np.reshape(x_batch, (self.batch_size, self.t, self.d))
            y_batch = np.reshape(y_batch, (self.batch_size, self.d))
            self.model.train_on_batch(x_batch, y_batch)

    def batch_last_clust(self):
        """
        This function is exclusively used for the Botvinick Simulations
        :return:
        """

        # draw a set of training examples from the history

        # pull the last cluster
        x_batch = self.x_history[-1]
        y_batch = self.y_history[-1]

        x_batch = unroll_data(x_batch, t=np.shape(x_batch)[0])

        self.model.train_on_batch(x_batch, y_batch)


class KerasRecurrentMLP(KerasSRN):

    def __init__(self, d, var_df0, var_scale0, t=3, n_hidden=None, hidden_act='tanh', optimizer=None,
                 n_epochs=100, dropout=0.50, l2_regularization=0.00, batch_size=32,
                 kernel_initializer='glorot_uniform', init_model=True):

        KerasSRN.__init__(self, d, var_df0, var_scale0, t=t, optimizer=optimizer, n_epochs=n_epochs,
                          l2_regularization=l2_regularization, batch_size=batch_size,
                          kernel_initializer=kernel_initializer, init_model=False)

        if n_hidden is None:
            self.n_hidden = d
        else:
            self.n_hidden = n_hidden
        self.hidden_act = hidden_act
        self.dropout = dropout

        if init_model:
            self._init_model()

    def _init_model(self):
        self.sess = tf.Session()

        self.model = Sequential()
        # input_shape[0] = timesteps; we pass the last self.t examples for train the hidden layer
        # input_shape[1] = input_dim; each example is a self.d-dimensional vector
        self.model.add(SimpleRNN(self.n_hidden, input_shape=(self.t, self.d), #activation=self.hidden_act,
                                 kernel_regularizer=self.kernel_regularizer,
                                 kernel_initializer=self.kernel_initializer))
        self.model.add(LeakyReLU(alpha=0.3))
        self.model.add(Dropout(self.dropout))
        self.model.add(Dense(self.d, activation=None, kernel_regularizer=self.kernel_regularizer,
                  kernel_initializer=self.kernel_initializer))
        self.model.compile(**self.compile_opts)


class KerasRecurrentMLP_BN(KerasRecurrentMLP):

    def _init_model(self):
        self.sess = tf.Session()

        self.model = Sequential()
        # input_shape[0] = timesteps; we pass the last self.t examples for train the hidden layer
        # input_shape[1] = input_dim; each example is a self.d-dimensional vector
        self.model.add(SimpleRNN(self.n_hidden, input_shape=(self.t, self.d),
                                 kernel_regularizer=self.kernel_regularizer,
                                 kernel_initializer=self.kernel_initializer))
        self.model.add(LeakyReLU(alpha=0.3))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(self.dropout))
        self.model.add(Dense(self.d, activation=None, kernel_regularizer=self.kernel_regularizer,
                  kernel_initializer=self.kernel_initializer))
        self.model.compile(**self.compile_opts)


class KerasGRU(KerasSRN):

    def __init__(self, d, var_df0, var_scale0, t=3, n_hidden=None, hidden_act='tanh', optimizer=None,
                 n_epochs=100, dropout=0.50, l2_regularization=0.00,batch_size=32,
                 kernel_initializer='glorot_uniform', init_model=True):

        KerasSRN.__init__(self, d, var_df0, var_scale0, t=t, optimizer=optimizer, n_epochs=n_epochs,
                          l2_regularization=l2_regularization, batch_size=batch_size,
                          kernel_initializer=kernel_initializer,  init_model=False)

        if n_hidden is None:
            self.n_hidden = d
        else:
            self.n_hidden = n_hidden
        self.hidden_act = hidden_act
        self.dropout = dropout

        if init_model:
            self._init_model()

    def _init_model(self):
        self.sess = tf.Session()

        self.model = Sequential()
        # input_shape[0] = timesteps; we pass the last self.t examples for train the hidden layer
        # input_shape[1] = input_dim; each example is a self.d-dimensional vector
        self.model.add(GRU(self.n_hidden, input_shape=(self.t, self.d), #activation=self.hidden_act,
                                 kernel_regularizer=self.kernel_regularizer,
                                 kernel_initializer=self.kernel_initializer))
        self.model.add(LeakyReLU(alpha=0.3))
        self.model.add(Dropout(self.dropout))
        self.model.add(Dense(self.d, activation=None, kernel_regularizer=self.kernel_regularizer,
                  kernel_initializer=self.kernel_initializer))
        self.model.compile(**self.compile_opts)


class KerasLSTM(KerasSRN):

    def __init__(self, d, var_df0, var_scale0, t=3, n_hidden=None, hidden_act='tanh', optimizer=None,
                 n_epochs=100, dropout=0.50, l2_regularization=0.00,
                 batch_size=32,
                 kernel_initializer='glorot_uniform', init_model=True):

        KerasSRN.__init__(self, d, var_df0, var_scale0, t=t, optimizer=optimizer, n_epochs=n_epochs,
                          l2_regularization=l2_regularization, batch_size=batch_size,
                          kernel_initializer=kernel_initializer, init_model=False)

        if n_hidden is None:
            self.n_hidden = d
        else:
            self.n_hidden = n_hidden
        self.hidden_act = hidden_act
        self.dropout = dropout

        if init_model:
            self._init_model()

    def _init_model(self):
        self.sess = tf.Session()

        self.model = Sequential()
        # input_shape[0] = timesteps; we pass the last self.t examples for train the hidden layer
        # input_shape[1] = input_dim; each example is a self.d-dimensional vector
        self.model.add(LSTM(self.n_hidden, input_shape=(self.t, self.d), #activation=self.hidden_act,
                           kernel_regularizer=self.kernel_regularizer,
                           kernel_initializer=self.kernel_initializer))
        self.model.add(LeakyReLU(alpha=0.3))
        self.model.add(Dropout(self.dropout))
        self.model.add(Dense(self.d, activation=None, kernel_regularizer=self.kernel_regularizer,
                             kernel_initializer=self.kernel_initializer))
        self.model.compile(**self.compile_opts)