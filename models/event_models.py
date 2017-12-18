import tensorflow as tf
import numpy as np
from utils import unroll_data
from keras.models import Sequential
from keras.layers import Dense, Activation, SimpleRNN, GRU, Dropout
from keras import optimizers
from keras import regularizers


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

    def predict_next(self, X):
        """
        Parameters
        ----------
        X: 1xD array-like data of inputs

        Returns
        -------
        y: 1xD array of prediction vectors

        """
        return np.copy(X)

    def predict_f0(self):
        return np.zeros(self.D)

    def update_f0(self, Y):
        pass

    def close(self):
        pass

    def new_cluster(self):
        pass


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
        Y_hat = np.reshape(self.predict_next(X), (Y.shape[0]))

        # compute gradient of log likelihood w.r.t. parameters
        db = self.eta * (Y - Y_hat)
        dW = self.eta * np.matmul(X.reshape(self.D, 1), (Y - Y_hat).reshape(1, self.D))

        # store the updated parameters
        self.beta += db
        self.W += dW

    def predict_f0(self):
        return self.f0

    def update_f0(self, Y):
        self.f0 += self.eta * (Y - self.f0)


class KerasLDS(EventModel):

    def __init__(self, D, sgd_kwargs=None, n_epochs=50, init_model=True):
        EventModel.__init__(self, D)
        self.x_train = np.zeros((0, self.D))  # initialize empty arrays
        self.y_train = np.zeros((0, self.D))
        if sgd_kwargs is None:
            sgd_kwargs = {
                'nesterov': True,
                'lr': 0.1,
                'momentum': 0.5,
                'decay': 0.0001
            }

        self.compile_opts = dict(optimizer=optimizers.SGD(**sgd_kwargs), loss='mean_squared_error')
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

    def update(self, X, Y):
        if np.ndim(X) == 1:
            N = 1
        else:
            N, _ = np.shape(X)
        self.x_train = np.concatenate([self.x_train, np.reshape(X, newshape=(N, self.D))])
        self.y_train = np.concatenate([self.y_train, np.reshape(Y, newshape=(N, self.D))])
        self.model.fit(self.x_train, self.y_train, verbose=0, nb_epoch=self.n_epochs)
        self.is_visited = True

    def predict_next(self, X):
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
        return self.predict_next(X)

    def predict_f0(self):
        if self.is_visited:
            return self.predict_next(np.zeros(self.D))
        return np.zeros(self.D)

    def update_f0(self, Y):
        # eta = 1. / (1.0 + self.f0_count)
        # self.f0 += eta * (Y - self.f0)
        # self.f0_count += 1.0
        self.update(np.zeros(self.D), Y)

class KerasMultiLayerPerceptron(KerasLDS):

    def __init__(self, D, n_hidden=None, hidden_act='tanh', sgd_kwargs=None, n_epochs=50, l2_regularization=0.01):
        KerasLDS.__init__(self, D, sgd_kwargs=sgd_kwargs, init_model=False)
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
#

    def __init__(self, D, t=5, n_hidden1=10, n_hidden2=10, hidden_act1='relu', hidden_act2='relu',
                 sgd_kwargs=None, n_epochs=50, dropout=0.50, l2_regularization=0.01):
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

        KerasLDS.__init__(self, D, sgd_kwargs=sgd_kwargs, init_model=False)

        self.t = t
        self.n_epochs = n_epochs
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2
        self.hidden_act1 = hidden_act1 
        self.hidden_act2 = hidden_act2
        self.D = D
        self.dropout = dropout
        self.kernel_regularizer = regularizers.l2(l2_regularization)

        # list of clusters of scenes:
        # each element of list = history of scenes for given cluster
        # history = N x D tensor, N = # of scenes in cluster, D = dimension of single scene
        #
        self.x_history = [np.zeros((0, self.D))]
        self.y_history = [np.zeros((0, self.D))]

        self._init_model()
        self.n_epochs_trained = 0

    # initialize model once so we can then update it online
    #
    def _init_model(self):
        self.sess = tf.Session()

        self.model = Sequential()
        # input_shape[0] = timesteps; we pass the last self.t examples for train the hidden layer
        # input_shape[1] = input_dim; each example is a self.D-dimensional vector
        self.model.add(SimpleRNN(self.n_hidden1, input_shape=(self.t, self.D), activation=self.hidden_act1))
        self.model.add(Dense(self.n_hidden2, activation=self.hidden_act2, kernel_regularizer=self.kernel_regularizer))
        self.model.add(Dropout(self.dropout))
        self.model.add(Dense(self.D, activation=None,  kernel_regularizer=self.kernel_regularizer))
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
    def update(self, X, Y):
        if X.ndim > 1:
            X = X[-1, :]  # only consider last example
        assert X.ndim == 1
        assert X.shape[0] == self.D
        assert Y.ndim == 1
        assert Y.shape[0] == self.D

        x_example = X.reshape((1, self.D))
        y_example = Y.reshape((1, self.D))

        # concatenate current example with history of last t-1 examples
        # this is for the recurrent part of the network
        # so a single training "example" = the last t training examples
        # notice there is still only one target label
        #
        x_train = self._unroll(x_example)
        y_train = y_example

        h = self.model.fit(x_train, y_train, verbose=0, initial_epoch=self.n_epochs_trained,
                           epochs=self.n_epochs + self.n_epochs_trained, shuffle=False)
        self.n_epochs_trained += self.n_epochs  # keep track that we have already done training on the models

        self.x_history[-1] = np.concatenate([self.x_history[-1], x_example], axis=0)
        self.y_history[-1] = np.concatenate([self.y_history[-1], y_example], axis=0)
        self.is_visited = True
        return h

    # predict a single example
    def predict_next(self, X):
        # Note: this function predicts the next conditioned on the training data the model has seen

        if X.ndim > 1:
            X = X[-1, :]  # only consider last example
        assert np.ndim(X) == 1
        assert X.shape[0] == self.D

        x_test = X.reshape((1, self.D))

        # concatenate current example with history of last t-1 examples
        # this is for the recurrent part of the network
        #
        x_test = self._unroll(x_test)

        return self.model.predict(x_test)

    def predict_next_generative(self, X):
        X0 = np.reshape(unroll_data(X, self.t)[-1, :, :], (1, self.t, self.D))
        return self.model.predict(X0)


    # create a new cluster of scenes
    def new_cluster(self):
        if len(self.x_history) == 1 and self.x_history[0].shape[0] == 0:
            # special case for the first cluster which is already created
            return 
        self.x_history.append(np.zeros((0, self.D)))
        self.y_history.append(np.zeros((0, self.D)))

    # optional: run batch gradient descent on all past event clusters
    def batch(self):
        for clust_id in range(len(self.x_history)):
            x_train = unroll_data(self.x_history[clust_id], self.t)
            y_train = self.y_history[clust_id]
            h = self.model.fit(x_train, y_train, verbose=0, epochs=self.n_epochs, shuffle=False)


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
