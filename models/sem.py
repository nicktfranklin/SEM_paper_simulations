import numpy as np
import tensorflow as tf
from scipy.misc import logsumexp
from tqdm import tnrange
from scipy.stats import multivariate_normal as mvnormal
from keras.models import model_from_json
import copy
import os

class SEM(object):
    """
    This port of SAM's code (done with a different programming logic)
    in python. More documentation to come!
    """

    def __init__(self, lmda=1., alfa=10.0, beta=0.1, f_class=None, f_opts=None):
        """
        Parameters
        ----------

        lmbda: float
            sCRP stickiness parameter

        alfa: float
            sCRP concentration parameter

        beta: float
            gaussian noise parameter

        f_class: class
            object class that has the functions "predict" and "update".
            used as the event model

        f_opts: dictionary
            kwargs for initializing f_class
        """
        self.lmda = lmda
        self.alfa = alfa
        self.beta = beta

        if f_class is None:
            raise ValueError("f_model must be specified!")

        self.f_class = f_class
        self.f_opts = f_opts

        # SEM internal state
        #
        self.K = 0 # maximum number of clusters (event types)
        self.C = np.array([]) # running count of the clustering process = number of scenes for each event type (the CRP prior)
        self.D = None # dimension of scenes
        self.event_models = dict() # event model for each event type

        self.x_prev = None # last scene
        self.k_prev = None # last event type


    def serialize(self, weights_dir='.'):
        """
        Serialize SEM object into a dict that can be pickled. Need to take special care of the event models
        as they don't pickle straightforwardly.
        """
        from opt.utils import randstr
        print 'Serializing SEM object ...'

        dump = dict()
        for attr_name in get_object_attributes(self):
            if attr_name == 'event_models':
                # event models are special as we can't just pickle them
                #
                event_dump = dict()
                for k, event_model in self.event_models.iteritems():
                    # make shallow copy of event model
                    # we only want to change a couple of the fields
                    event_model = copy.copy(event_model)

                    # save model weights to a separate HDF5 file
                    # TODO that's pretty lame but that's the easiest way I could do it. Should probably make this better
                    # https://machinelearningmastery.com/save-load-keras-deep-learning-models/
                    event_model.weights_filename = os.path.join(weights_dir, 'event_model_weights_' + randstr(10) + '.h5')
                    event_model.model.save_weights(event_model.weights_filename)

                    print '           saving event model ', k, event_model, ' to ', event_model.weights_filename

                    # change problematic fields (this is why we make a copy of the event_model)
                    event_model.sess = None # cannot serialize TF session
                    event_model.model = event_model.model.to_json() # jsonify the model structure
                    event_model.compile_opts = None # can't serialize the keras optimizers

                    # save the altered event model
                    event_dump[k] = event_model

                dump['event_models'] = event_dump
            else:
                print '           saving attribute ', attr_name
                dump[attr_name] = getattr(self, attr_name)

        return dump


    def deserialize(self, dump):
        """
        Deserialize SEM object after it has been serialized with serialize().
        Special care is taken for the event models.
        """
        print 'Deserializing SEM object ...'

        for attr_name in get_object_attributes(self):
            if attr_name == 'event_models':
                D = dump['D']
                dummy_event_model = self.f_class(D, **self.f_opts) # dummy event model to get the keras optimizer (can't serialize it)

                event_dump = dump['event_models']
                for k, event_model in event_dump.iteritems():
                    print '       loading event model', k, ' from ', event_model.weights_filename

                    # bring model back to normal
                    event_model.sess = tf.Session() # restart TF session for model
                    event_model.model = model_from_json(event_model.model) # restore model structure from json
                    event_model.model.load_weights(event_model.weights_filename) # restore model weights from HDF5
                    event_model.model.compile(**dummy_event_model.compile_opts) # compile model using dummy event model options (can't serialize them)

                    self.event_models[k] = event_model
            else:
                print '      setting ', attr_name, ' to ', dump[attr_name]
                setattr(self, attr_name, dump[attr_name])



    def pretrain(self, X, y):
        """
        Pretrain a bunch of event models on sequence of scenes X
        with corresponding event labels y, assumed to be between 0 and K-1
        where K = total # of distinct event types
        """
        assert X.shape[0] == y.size

        # update internal state
        K = np.max(y) + 1
        self.update_state(X, K)
        del K # use self.K

        N = X.shape[0]

        # loop over all scenes
        #
        for n in tnrange(N, desc='Pretraining'):
            # print 'pretraining at scene ', n

            x_curr = X[n, :].copy() # current scene
            k = y[n] # current event

            if k not in self.event_models.keys():
                # initialize new event model
                self.event_models[k] = self.f_class(self.D, **self.f_opts)

            # update event model
            if self.k_prev == k:
                # we're in the same event -> update using previous scene
                assert self.x_prev is not None
                self.event_models[k].update(self.x_prev, x_curr)
            else:
                # we're in a new event -> update the initialization point only
                self.event_models[k].new_cluster()
                self.event_models[k].update_f0(x_curr)

            self.C[k] += 1  # update counts

            self.x_prev = x_curr  # store the current scene for next trial
            self.k_prev = k # store the current event for the next trial


    def update_state(self, X, K=None):
        """
        Update internal state based on input data X and max # of event types (clusters) K
        """
        # get dimensions of data
        [N, D] = np.shape(X)
        if self.D is None:
            self.D = D
        else:
            assert self.D == D # scenes must be of same dimension

        # get max # of clusters / event types
        if K is None:
            K = N
        self.K = max(self.K, K)

        # initialize CRP prior = running count of the clustering process
        if self.C.size < self.K:
            self.C = np.concatenate((self.C, np.zeros(self.K - self.C.size)), axis=0)
        assert self.C.size == self.K


    def run(self, X, K=None, return_pe=False, return_err=False, return_lik_prior=False):
        """
        Parameters
        ----------
        X: N x D array of

        K: int
            maximum number of clusters

        return_pe: bool
            return the "prediction error" of the model - i.e. the euclidean distance between the prediction
            of the previously active cluster and the current observation.

        return_lik_prior: bool
            return the model's log likelihood and log prior over clusterings

        Return
        ------
        post: N by K array of posterior probabilities
        """

        # update internal state
        self.update_state(X, K)
        del K # use self.K and self.D

        N = X.shape[0]

        Sigma = np.eye(self.D) * self.beta  # noise for multivariate gaussian likelihood

        if return_pe:
            pe = np.zeros(np.shape(X)[0])

        post = np.zeros((N, self.K))

        # debugging function
        if return_lik_prior:
            log_like = np.zeros((N, self.K))
            log_prior = np.zeros((N, self.K))

        # print 'Running SEM on', N, 'scenes of dimension', self.D, 'with a maximum of', self.K, 'event types'

        for n in tnrange(N):
            x_curr = X[n, :].copy()

            # calculate sCRP prior
            prior = self.C.copy()
            idx = len(np.nonzero(self.C)[0])  # get number of visited clusters

            if idx < self.K:
                prior[idx] = self.alfa  # set new cluster probability to alpha

            # add stickiness parameter for n>0, only for the previously chosen event
            if n > 0:
                prior[np.argmax(post[n-1, :])] += self.lmda

            prior /= np.sum(prior)

            # likelihood
            active = np.nonzero(prior)[0]
            lik = np.zeros(len(active))

            for k in active:
                if k not in self.event_models.keys():
                    self.event_models[k] = self.f_class(self.D, **self.f_opts)

                # get the log likelihood for each event model
                model = self.event_models[k]

                if k == self.k_prev:
                    assert self.x_prev is not None
                    Y_hat = model.predict_next(self.x_prev)
                else:
                    Y_hat = model.predict_f0()
                lik[k] = mvnormal.logpdf(x_curr - Y_hat, mean=np.zeros(self.D), cov=Sigma)

            # posterior
            p = np.log(prior[:len(active)]) + lik - np.max(lik)  # subtracting the max like doesn't change proportionality
            post[n, :len(active)] = np.exp(p - logsumexp(p))
            # update

            # this is a diagnostic readout and does not effect the model
            if return_lik_prior:
                log_like[n, :len(active)] = lik - np.max(lik)
                log_prior[n, :len(active)] = np.log(prior[:len(active)])

            k = np.argmax(post[n, :])  # MAP cluster

            # prediction error: euclidean distance of the last model and the current scene vector
            if return_pe and n > 0:
                assert self.x_prev is not None and self.k_prev is not None
                model = self.event_models[self.k_prev]
                pe[n] = np.linalg.norm(x_curr - model.predict_next(self.x_prev))

            self.C[k] += 1  # update counts

            # update event model
            if self.k_prev == k:
                # we're in the same event -> update using previous scene
                assert self.x_prev is not None
                self.event_models[k].update(self.x_prev, x_curr)
            else:
                # we're in a new event -> update the initialization point only
                self.event_models[k].new_cluster()
                self.event_models[k].update_f0(x_curr)

            # print self.C[0:np.max(active)+1]
            # print 'scene ', n, ' map = ', k, ' prior = ', prior[0:np.max(active)+1], ', lik = ', lik, ' post = ', post[n,0:np.max(active)+1]

            self.x_prev = x_curr  # store the current scene for next trial
            self.k_prev = k # store the current event for the next trial

        if return_pe:
            if return_lik_prior:
                return post, pe, log_like, log_prior
            return post, pe

        if return_lik_prior:
            return post, None, log_like, log_prior

        return post


# helper f'n that gets all attributes of an object
#
def get_object_attributes(obj):
    return [a for a in dir(obj) if not a.startswith('__') and not callable(getattr(obj,a))]

