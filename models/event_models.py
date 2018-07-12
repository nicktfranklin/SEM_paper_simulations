import tensorflow as tf
import numpy as np
from utils import unroll_data
from keras.models import Sequential
from keras.layers import Dense, Activation, SimpleRNN, GRU, Dropout, LSTM
from keras import backend as K
from keras import initializers
from keras import optimizers
from keras import regularizers

from collections import OrderedDict


class GRULN(GRU):
    '''Gated Recurrent Unit with Layer Normalization
    Current impelemtation only works with consume_less = 'gpu' which is already
    set.
    # Arguments
        output_dim: dimension of the internal projections and the final output.
        ...: see GRU documentation for all other arguments.
        gamma_init: name of initialization function for scale parameter.
            The default is 1, but in some cases this is too high resulting
            in NaN loss while training. If this happens try reducing to 0.2
    # References
        -[Layer Normalization](https://arxiv.org/abs/1607.06450)
    '''
    def __init__(self, output_dim, gamma_init=1., **kwargs):
        if 'consume_less' in kwargs:
            assert kwargs['consume_less'] == 'gpu'
        else:
            kwargs = kwargs.copy()
            kwargs['consume_less']='gpu'
        super(GRULN, self).__init__(output_dim, **kwargs)

        def gamma_init_func(shape, c=gamma_init, **kwargs):
            if c == 1.:
                return initializers.Ones(shape, **kwargs)
            return K.variable(np.ones(shape) * c, **kwargs)

        self.gamma_init = gamma_init_func
        self.beta_init = initializers.Zeros()
        self.epsilon = 1e-5

    def build(self, input_shape):
        super(GRULN, self).build(input_shape)
        shape = (self.output_dim,)
        shape1 = (2*self.output_dim,)
        # LN is applied in 4 inputs/outputs (fields) of the cell
        gammas = OrderedDict()
        betas = OrderedDict()
        # each location has its own BN
        for slc, shp in zip(['state_below', 'state_belowx', 'preact', 'preactx'], [shape1, shape, shape1, shape]):
            gammas[slc] = self.gamma_init(shp,
                                          name='{}_gamma_{}'.format(
                                              self.name, slc))
            betas[slc] = self.beta_init(shp,
                                        name='{}_beta_{}'.format(
                                            self.name, slc))

        self.gammas = gammas
        self.betas = betas

        self.trainable_weights += self.gammas.values() + self.betas.values()

    def ln(self, x, slc):
        # sample-wise normalization
        m = K.mean(x, axis=-1, keepdims=True)
        std = K.sqrt(K.var(x, axis=-1, keepdims=True) + self.epsilon)
        x_normed = (x - m) / (std + self.epsilon)
        x_normed = self.gammas[slc] * x_normed + self.betas[slc]
        return x_normed

    def step(self, x, states):
        h_tm1 = states[0]  # previous memory
        B_U = states[1]  # dropout matrices for recurrent units
        B_W = states[2]

        matrix_x = K.dot(x * B_W[0], self.W) + self.b
        x_ = self.ln(matrix_x[:, : 2 * self.output_dim], 'state_below')
        xx_ = self.ln(matrix_x[:, 2 * self.output_dim:], 'state_belowx')
        matrix_inner = self.ln(K.dot(h_tm1 * B_U[0], self.U[:, :2 * self.output_dim]), 'preact')

        x_z = x_[:, :self.output_dim]
        x_r = x_[:, self.output_dim: 2 * self.output_dim]
        inner_z = matrix_inner[:, :self.output_dim]
        inner_r = matrix_inner[:, self.output_dim: 2 * self.output_dim]

        z = self.inner_activation(x_z + inner_z)
        r = self.inner_activation(x_r + inner_r)

        x_h = xx_
        inner_h = r * self.ln(K.dot(h_tm1 * B_U[0], self.U[:, 2 * self.output_dim:]), 'preactx')
        hh = self.activation(x_h + inner_h)

        h = z * h_tm1 + (1 - z) * hh
        return h, [h]


class EventModel(object):

    """ this is the base clase of the event model """

    def __init__(self, D):
        self.D = D
        self.f_is_trained = False
        self.f0_is_trained = False

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

    def __init__(self, D, optimizer='adam', n_epochs=50, init_model=True):
        EventModel.__init__(self, D)
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

        self.compile_opts = dict(optimizer=optimizer, loss='mean_squared_error', )
        self.n_epochs = n_epochs

        self.is_visited = False  # governs the special case of model's first prediction (i.e. with no experience)

        if init_model:
            self._init_model()

    def _init_model(self):
        self.sess = tf.Session()

        N, D = self.x_train.shape

        self.model = Sequential([
            Dense(D, input_shape=(D, )),
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



class KerasMultiLayerPerceptron(KerasLDS):

    def __init__(self, D, n_hidden=None, hidden_act='tanh', optimizer='adam', n_epochs=50, l2_regularization=0.00):
        KerasLDS.__init__(self, D, optimizer=optimizer, init_model=False)
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
        self.model.add(Dense(D, activation='linear',  kernel_regularizer=self.kernel_regularizer))
        self.model.compile(**self.compile_opts)


class KerasSimpleRNN(KerasLDS):

    # RNN which is initialized once and then trained using stochastic gradient descent
    # i.e. each new scene is a single example batch of size 1

    def __init__(self, D, t=3, n_hidden1=None, n_hidden2=None, hidden_act1='relu', hidden_act2='relu',
                 optimizer='adam', n_epochs=50, dropout=0.10, l2_regularization=0.00, batch_size=32,
                 kernel_initializer='glorot_uniform'):
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

        KerasLDS.__init__(self, D, optimizer=optimizer, init_model=False)

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
        self.model.add(Dense(self.D, activation=None,  kernel_regularizer=self.kernel_regularizer,
                             kernel_initializer=self.kernel_initializer))
        self.model.compile(**self.compile_opts)

    # concatenate current example with the history of the last t-1 examples
    # this is for the recurrent layer
    #
    def _unroll(self, x_example):
        x_train = np.concatenate([self.x_history[-1][-(self.t-1):, :], x_example], axis=0)
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
        x_batch = []
        y_batch = []

        # pull the last cluster
        x_batch = self.x_history[-1]
        y_batch = self.y_history[-1]

        # t = np.random.randint(len(x_history))
        #
        # x_batch.append(np.reshape(
        #     unroll_data(x_history[max(t - self.t, 0):t + 1, :], self.t)[-1, :, :], (1, self.t, self.D)
        # ))
        # y_batch.append(y_history[t, :])

        x_batch = unroll_data(x_batch, t=np.shape(x_batch)[0])

        # x_batch = np.reshape(x_batch, (self.batch_size, self.t, self.D))
        # y_batch = np.reshape(y_batch, (self.batch_size, self.D))
        self.model.train_on_batch(x_batch, y_batch)


class KerasGRU(KerasSimpleRNN):
    def _init_model(self):
        self.sess = tf.Session()

        # input_shape[0] = timesteps; we pass the last self.t examples for train the hidden layer
        # input_shape[1] = input_dim; each example is a self.D-dimensional vector
        self.model = Sequential()
        self.model.add(GRU(self.n_hidden1, input_shape=(self.t, self.D), activation=self.hidden_act1))
        self.model.add(Dense(self.n_hidden2, activation=self.hidden_act2, kernel_regularizer=self.kernel_regularizer))
        self.model.add(Dropout(self.dropout))
        self.model.add(Dense(self.D, activation=None,  kernel_regularizer=self.kernel_regularizer))
        self.model.compile(**self.compile_opts)


class KerasGRU_norm(KerasSimpleRNN):
    def _init_model(self):
        self.sess = tf.Session()

        # input_shape[0] = timesteps; we pass the last self.t examples for train the hidden layer
        # input_shape[1] = input_dim; each example is a self.D-dimensional vector
        self.model = Sequential()
        self.model.add(GRULN(self.n_hidden1, input_shape=(self.t, self.D), activation=self.hidden_act1))
        self.model.add(Dense(self.n_hidden2, activation=self.hidden_act2, kernel_regularizer=self.kernel_regularizer))
        # self.model.add(GRU(self.n_hidden1, input_shape=(self.t, self.D), activation=self.hidden_act1))
        # self.model.add(Dense(self.n_hidden2, activation=self.hidden_act2, kernel_constraint=max_norm(1.)))
        self.model.add(Dropout(self.dropout))
        self.model.add(Dense(self.D, activation=None,  kernel_regularizer=self.kernel_regularizer))
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
        self.model.add(Dense(self.D, activation=None,  kernel_regularizer=self.kernel_regularizer))
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
        self.model.add(Dense(self.D, activation=None,  kernel_regularizer=self.kernel_regularizer))
        self.model.compile(**self.compile_opts)


class CustomGRU(EventModel):

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

    def _predict_next(self, X):
        """
        Parameters
        ----------
        X: 1xD array-like data of inputs

        Returns
        -------
        y: 1xD array of prediction vectors

        """
        return np.copy(X)

    def _predict_f0(self):
        return np.zeros(self.D)

    def predict_next_generative(self, X):
        X0 = np.reshape(unroll_data(X, self.t)[-1, :, :], (1, self.t, self.D))
        return self._predict_next(X0)

    def update_f0(self, Y):
        pass

    def close(self):
        pass

    def new_token(self):
        pass

    def run_generative(self, n_steps, initial_point=None):
        if initial_point is None:
            x_gen = self._predict_f0()
        else:
            x_gen = np.reshape(initial_point, (1, self.D))
        for ii in range(1, n_steps):
            x_gen = np.concatenate([x_gen, self.predict_next_generative(x_gen[:ii, :])])
        return x_gen