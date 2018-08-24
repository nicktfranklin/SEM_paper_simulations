import numpy as np
import tensorflow as tf
from scipy.misc import logsumexp
from tqdm import tqdm
from keras.models import model_from_json
import copy
import os


# helper f'n that gets all attributes of an object
#
def get_object_attributes(obj):
    return [a for a in dir(obj) if not a.startswith('__') and not callable(getattr(obj,a))]


class Results(object):
    """ placeholder object to store video_results """
    pass


class SEM(object):
    """
    This port of SAM's code (done with a different programming logic)
    in python. More documentation to come!
    """

    def __init__(self, lmda=1., alfa=10.0, f_class=None, f_opts=None, ):
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
        # self.beta = beta

        if f_class is None:
            raise ValueError("f_model must be specified!")

        self.f_class = f_class
        self.f_opts = f_opts

        # SEM internal state
        #
        self.K = 0  # maximum number of clusters (event types)
        self.C = np.array([])  # running count of the clustering process = n of scenes
                               # for each event type (the CRP prior)
        self.d = None  # dimension of scenes
        self.event_models = dict()  # event model for each event type

        self.x_prev = None  # last scene
        self.k_prev = None  # last event type

        # instead of dumping the video_results, store them to the object
        self.results = None

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
                    event_model.weights_filename = os.path.join\
                        (weights_dir, 'event_model_weights_' + randstr(10) + '.h5')
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
                dummy_event_model = self.f_class(D, **self.f_opts)  # dummy event model to get the keras
                                                                    # optimizer (can't serialize it)

                event_dump = dump['event_models']
                for k, event_model in event_dump.iteritems():
                    print '       loading event model', k, ' from ', event_model.weights_filename

                    # bring model back to normal
                    event_model.sess = tf.Session()  # restart TF session for model
                    event_model.model = model_from_json(event_model.model)  # restore model structure from json
                    event_model.model.load_weights(event_model.weights_filename)  # restore model weights from HDF5
                    event_model.model.compile(**dummy_event_model.compile_opts)  # compile model using dummy event model
                    #  options (can't serialize them)

                    self.event_models[k] = event_model
            else:
                print '      setting ', attr_name, ' to ', dump[attr_name]
                setattr(self, attr_name, dump[attr_name])

    def pretrain(self, X, y, progress_bar=True, leave_progress_bar=True):
        """
        Pretrain a bunch of event models on sequence of scenes X
        with corresponding event labels y, assumed to be between 0 and K-1
        where K = total # of distinct event types
        """
        assert X.shape[0] == y.size

        # update internal state
        k = np.max(y) + 1
        self._update_state(X, k)
        del k  # use self.k

        N = X.shape[0]

        # loop over all scenes
        if progress_bar:
            def my_it(N):
                return tqdm(range(N), desc='Pretraining', leave=leave_progress_bar)
        else:
            def my_it(N):
                return range(N)

        #
        for n in my_it(N):
            # print 'pretraining at scene ', n

            x_curr = X[n, :].copy() # current scene
            k = y[n] # current event

            if k not in self.event_models.keys():
                # initialize new event model
                self.event_models[k] = self.f_class(self.d, **self.f_opts)

            # update event model
            if self.k_prev == k:
                # we're in the same event -> update using previous scene
                assert self.x_prev is not None
                self.event_models[k].update(self.x_prev, x_curr)
            else:
                # we're in a new event -> update the initialization point only
                self.event_models[k].new_token()
                self.event_models[k].update(np.zeros((1, self.d)), x_curr)

            self.C[k] += 1  # update counts

            self.x_prev = x_curr  # store the current scene for next trial
            self.k_prev = k  # store the current event for the next trial

    def _update_state(self, X, K=None):
        """
        Update internal state based on input data X and max # of event types (clusters) K
        """
        # get dimensions of data
        [n, d] = np.shape(X)
        if self.d is None:
            self.d = d
        else:
            assert self.d == d  # scenes must be of same dimension

        # get max # of clusters / event types
        if K is None:
            K = n
        self.K = max(self.K, K)

        # initialize CRP prior = running count of the clustering process
        if self.C.size < self.K:
            self.C = np.concatenate((self.C, np.zeros(self.K - self.C.size)), axis=0)
        assert self.C.size == self.K

    def _calculate_prior_from_counts(self, prev_cluster=None):
        # internal function for consistencey across "run" methods

        # calculate sCRP prior
        prior = self.C.copy()
        idx = len(np.nonzero(self.C)[0])  # get number of visited clusters

        if idx <= self.K:
            prior[idx] += self.alfa  # set new cluster probability to alpha

        # add stickiness parameter for n>0, only for the previously chosen event
        if prev_cluster is not None:
            prior[prev_cluster] += self.lmda

        prior /= np.sum(prior)
        return prior

    def run(self, X, K=None, progress_bar=True, leave_progress_bar=True):
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
        self._update_state(X, K)
        del K  # use self.k and self.d

        N = X.shape[0]

        # initialize arrays
        post = np.zeros((N, self.K))
        pe = np.zeros(np.shape(X)[0])
        y_hat = np.zeros(np.shape(X))

        # debugging functions
        log_like = np.zeros((N, self.K)) - np.inf
        log_prior = np.zeros((N, self.K)) - np.inf
        total_log_loss = 0.0

        # this code just controls the presence/absence of a progress bar -- it isn't important
        if progress_bar:
            def my_it(N):
                return tqdm(range(N), desc='Run SEM', leave=leave_progress_bar)
        else:
            def my_it(N):
                return range(N)

        for n in my_it(N):

            x_curr = X[n, :].copy()

            # calculate sCRP prior
            prior = self._calculate_prior_from_counts(self.k_prev)
            # N.B. k_prev should be none for the first event if there wasn't pre-training

            # likelihood
            active = np.nonzero(prior)[0]
            lik = np.zeros(len(active))

            for k0 in active:
                if k0 not in self.event_models.keys():
                    self.event_models[k0] = self.f_class(self.d, **self.f_opts)

                # get the log likelihood for each event model
                model = self.event_models[k0]

                # detect event boundaries when there is a change
                event_boundary0 = (k0 != self.k_prev)  # N.B this allows for experimenter override

                if not event_boundary0:
                    assert self.x_prev is not None
                    lik[k0] = model.likelihood_next(self.x_prev, x_curr)
                    # print "PE, Current Event: {}".format(np.linalg.norm(x_curr - model.predict_next(self.x_prev)))
                else:
                    lik[k0] = model.likelihood_next(np.zeros((1, self.d)), x_curr)
                    # print "PE, New Event:     {}".format(np.linalg.norm(x_curr - model.predict_f0()))

            # posterior
            p = np.log(prior[:len(active)]) + lik - np.max(lik)   # subtracting the max doesn't change proportionality
            post[n, :len(active)] = np.exp(p - logsumexp(p))
            # update

            # this is a diagnostic readout and does not effect the model
            log_like[n, :len(active)] = lik #- np.max(lik)
            log_prior[n, :len(active)] = np.log(prior[:len(active)])

            # get the MAP cluster and only update it
            K = np.argmax(post[n, :])  # MAP cluster

            # determine whether there was a boundary
            event_boundary = K != self.k_prev

            # prediction error: euclidean distance of the last model and the current scene vector
            if n > 0:
                model = self.event_models[self.k_prev]
                y_hat[n, :] = model.predict_next(self.x_prev)
                pe[n] = np.linalg.norm(x_curr - y_hat[n, :])

                # also calculate the log-loss for the observations
                # total_log_loss += model.predict_next

            self.C[K] += 1  # update counts
            # update event model
            if not event_boundary:
                # we're in the same event -> update using previous scene
                assert self.x_prev is not None
                self.event_models[K].update(self.x_prev, x_curr)
            else:
                # we're in a new event token -> update the initialization point only
                self.event_models[K].new_token()
                self.event_models[K].update(np.zeros((1, self.d)), x_curr)

            self.x_prev = x_curr  # store the current scene for next trial
            self.k_prev = K  # store the current event for the next trial

        self.results = Results()
        self.results.post = post
        self.results.pe = pe
        self.results.log_like = log_like
        self.results.log_prior = log_prior
        self.results.e_hat = np.argmax(post, axis=1)
        self.results.y_hat = y_hat
        self.results.log_loss = logsumexp(log_like + log_prior, axis=1)

        return post

    def run_w_boundaries(self, list_events, progress_bar=True, leave_progress_bar=True):
        """
        This method is the same as the above except the event boundaries are pre-specified by the experimenter
        as a list of event tokens (the event/schema type is still inferred).

        One difference is that the event token-type association is bound at the last scene of an event type.
        N.B. ! also, all of the updating is done at the event-token level.  There is no updating within an event!

        evaluate the probability of each event over the whole token


        Parameters
        ----------
        list_events: list of N x D arrays -- each an event


        progress_bar: bool


        leave_progress_bar: bool


        Return
        ------
        post: N by K array of posterior probabilities
        """

        # update internal state

        k = len(list_events)
        self._update_state(np.concatenate(list_events, axis=0), k)
        del k  # use self.k and self.d

        # initialize
        # initialize arrays -- these are calculated per scene!
        n_scenes = np.shape(np.concatenate(list_events, axis=0))[0]
        pe = np.zeros(n_scenes)
        y_hat = np.zeros((n_scenes, self.d))

        # debugging functions -- these are calculated per event!
        post = np.zeros((0, self.K))
        log_like = np.zeros((0, self.K)) - np.inf
        log_prior = np.zeros((0, self.K)) - np.inf

        # loop through the other events in the list
        if progress_bar:
            def my_it(iterator):
                return tqdm(iterator, desc='Run SEM', leave=leave_progress_bar)
        else:
            def my_it(iterator):
                return iterator

        if self.k_prev is None:
            self.event_models[0] = self.f_class(self.d, **self.f_opts)  # initialize the first event model

        for X in my_it(list_events):

            # extend the size of the posterior, etc
            post = np.concatenate([post, np.zeros((1, self.K))], axis=0)
            log_like = np.concatenate([log_like, np.zeros((1, self.K)) - np.inf], axis=0)
            log_prior = np.concatenate([log_prior, np.zeros((1, self.K)) - np.inf], axis=0)

            # calculate sCRP prior
            prior = self._calculate_prior_from_counts(self.k_prev)

            # likelihood
            active = np.nonzero(prior)[0]
            lik = np.zeros((np.shape(X)[0], len(active)))

            # again, this is for diagnostics only, but also keep track of the within event posterior
            _pe = np.zeros(np.shape(X)[0])
            k_within_event = np.argmax(prior)  # prior to the first scene within an event having been observed, the
            # prior determines what the event type will be

            for ii, x_curr in enumerate(X):

                # we need to maintain a distribution over possible event types for the current events --
                # this gets locked down after termination of the event.
                # Also: none of the event models can be updated until *after* the event has been observed

                # special case the first scene within the event
                if ii == 0:
                    event_boundary = True
                else:
                    event_boundary = False

                if event_boundary:
                    event_boundary = self.event_models[k_within_event]
                    _pe[ii] = np.linalg.norm(x_curr - event_boundary.predict_next_generative(np.zeros(1, self.d)))
                else:
                    _pe[ii] = np.linalg.norm(
                        x_curr - self.event_models[k_within_event].predict_next_generative(X[:ii, :]))

                # loop through each potentially active event model
                for k0 in active:
                    if k0 not in self.event_models.keys():
                        self.event_models[k0] = self.f_class(self.d, **self.f_opts)

                    # get the log likelihood for each event model
                    model = self.event_models[k0]

                    if not event_boundary:
                        lik[ii, k0] = model.likelihood_sequence(X[:ii, :], x_curr)
                    else:
                        lik[ii, k0] = model.likelihood_sequence(np.zeros((1, self.d)), x_curr)

                # for the purpose of calculating a prediction error and a prediction error only, calculate
                # a within event estimate of the event type (the real estimate is at the end of the event,
                # taking into account the accumulated evidence
                k_within_event = np.argmax(np.sum(lik[:ii+1, :len(active)], axis=0) + np.log(prior[:len(active)]))

            # cache the diagnostic measures
            log_like[-1, :len(active)] = np.sum(lik, axis=0)
            log_prior[-1, :len(active)] = np.log(prior[:len(active)])

            # at the end of the event, find the winning model!
            log_post = log_prior[-1, :len(active)] + log_like[-1, :len(active)]
            post[-1, :len(active)] = np.exp(log_post - logsumexp(log_post))
            k = np.argmax(log_post)

            # update the prior
            self.C[k] += np.shape(X)[0]
            # cache for next event
            self.k_prev = k

            # update the winning model's estimate
            self.event_models[k].update(np.zeros((1, self.d)), X[0])
            X_prev = X[0]
            for X0 in X[1:]:
                self.event_models[k].update(X0, X_prev)
                X_prev = X0

        #
        self.results = Results()
        self.results.post = post
        self.results.pe = pe
        self.results.log_like = log_like
        self.results.log_prior = log_prior
        self.results.e_hat = np.argmax(post, axis=1)
        self.results.y_hat = y_hat

        return post




