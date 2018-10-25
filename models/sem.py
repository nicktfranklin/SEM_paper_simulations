import numpy as np
import tensorflow as tf
from scipy.misc import logsumexp
from tqdm import tqdm
from keras.models import model_from_json
import copy
import os
from keras import backend as K


# helper f'n that gets all attributes of an object
#
def get_object_attributes(obj):
    return [a for a in dir(obj) if not a.startswith('__') and not callable(getattr(obj, a))]


class Results(object):
    """ placeholder object to store results """
    pass


class SEM(object):
    """
    This port of SAM's code (done with a different programming logic)
    in python. More documentation to come!
    """

    def __init__(self, lmda=1., alfa=10.0, f_class=None, f_opts=None):
        """
        Parameters
        ----------

        lmda: float
            sCRP stickiness parameter

        alfa: float
            sCRP concentration parameter

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
        self.k = 0  # maximum number of clusters (event types)
        self.c = np.array([])  # used by the sCRP prior -> running count of the clustering process
        self.d = None  # dimension of scenes
        self.event_models = dict()  # event model for each event type

        self.x_prev = None  # last scene
        self.k_prev = None  # last event type

        # instead of dumping the results, store them to the object
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
                    # Momchil: that's pretty lame but that's the easiest way I could do it.
                    # https://machinelearningmastery.com/save-load-keras-deep-learning-models/
                    event_model.weights_filename = os.path.join(
                        weights_dir, 'event_model_weights_' + randstr(10) + '.h5'
                    )
                    event_model.model.save_weights(event_model.weights_filename)

                    print '           saving event model ', k, event_model, ' to ', event_model.weights_filename

                    # change problematic fields (this is why we make a copy of the event_model)
                    event_model.sess = None  # cannot serialize TF session
                    event_model.model = event_model.model.to_json()  # jsonify the model structure
                    event_model.compile_opts = None  # can't serialize the keras optimizers

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
                d = dump['D']
                dummy_event_model = self.f_class(d, **self.f_opts)  # dummy event model to get the keras
                #  optimizer (can't serialize it)

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

    def pretrain(self, x, event_types, event_boundaries, progress_bar=True, leave_progress_bar=True):
        """
        Pretrain a bunch of event models on sequence of scenes X
        with corresponding event labels y, assumed to be between 0 and K-1
        where K = total # of distinct event types
        """
        assert x.shape[0] == event_types.size

        # update internal state
        k = np.max(event_types) + 1
        self._update_state(x, k)
        del k  # use self.k

        n = x.shape[0]

        # loop over all scenes
        if progress_bar:
            def my_it(l):
                return tqdm(range(l), desc='Pretraining', leave=leave_progress_bar)
        else:
            def my_it(l):
                return range(l)

        #
        for ii in my_it(n):
            # print 'pretraining at scene ', n

            x_curr = x[ii, :].copy()  # current scene
            k = event_types[ii]  # current event

            if k not in self.event_models.keys():
                # initialize new event model
                self.event_models[k] = self.f_class(self.d, **self.f_opts)

            # update event model
            if not event_boundaries[ii]:
                # we're in the same event -> update using previous scene
                assert self.x_prev is not None
                self.event_models[k].update(self.x_prev, x_curr)
            else:
                # we're in a new event -> update the initialization point only
                self.event_models[k].new_token()
                self.event_models[k].update_f0(x_curr)

            self.c[k] += 1  # update counts

            self.x_prev = x_curr  # store the current scene for next trial
            self.k_prev = k  # store the current event for the next trial

        self.x_prev = None  # Clear this for future use
        self.k_prev = None  #

    def _update_state(self, x, k=None):
        """
        Update internal state based on input data X and max # of event types (clusters) K
        """
        # get dimensions of data
        [n, d] = np.shape(x)
        if self.d is None:
            self.d = d
        else:
            assert self.d == d  # scenes must be of same dimension

        # get max # of clusters / event types
        if k is None:
            k = n
        self.k = max(self.k, k)

        # initialize CRP prior = running count of the clustering process
        if self.c.size < self.k:
            self.c = np.concatenate((self.c, np.zeros(self.k - self.c.size)), axis=0)
        assert self.c.size == self.k

    def _calculate_unnormed_sCRP(self, prev_cluster=None):
        # internal function for consistency across "run" methods

        # calculate sCRP prior
        prior = self.c.copy()
        idx = len(np.nonzero(self.c)[0])  # get number of visited clusters

        if idx <= self.k:
            prior[idx] += self.alfa  # set new cluster probability to alpha

        # add stickiness parameter for n>0, only for the previously chosen event
        if prev_cluster is not None:
            prior[prev_cluster] += self.lmda

        # prior /= np.sum(prior)
        return prior

    def run(self, x, k=None, progress_bar=True, leave_progress_bar=True, minimize_memory=False):
        """
        Parameters
        ----------
        x: N x D array of

        k: int
            maximum number of clusters

        progress_bar: bool
            use a tqdm progress bar?

        leave_progress_bar: bool
            leave the progress bar after completing?

        Return
        ------
        post: n by k array of posterior probabilities

        """

        # update internal state
        self._update_state(x, k)
        del k  # use self.k and self.d

        n = x.shape[0]

        # initialize arrays
        post = np.zeros((n, self.k))
        pe = np.zeros(np.shape(x)[0])
        y_hat = np.zeros(np.shape(x))
        log_boundary_probability = np.zeros(np.shape(x)[0])

        # these are special case variables to deal with the posibility the current event is restarted
        lik_restart_event = -np.inf
        repeat_prob = -np.inf
        restart_prob = 0

        # debugging functions
        log_like = np.zeros((n, self.k)) - np.inf
        log_prior = np.zeros((n, self.k)) - np.inf

        # this code just controls the presence/absence of a progress bar -- it isn't important
        if progress_bar:
            def my_it(l):
                return tqdm(range(l), desc='Run SEM', leave=leave_progress_bar)
        else:
            def my_it(l):
                return range(l)

        for ii in my_it(n):

            x_curr = x[ii, :].copy()

            # calculate sCRP prior
            prior = self._calculate_unnormed_sCRP(self.k_prev)
            # N.B. k_prev should be none for the first event if there wasn't pre-training

            # likelihood
            active = np.nonzero(prior)[0]
            lik = np.zeros(len(active))

            for k0 in active:
                if k0 not in self.event_models.keys():
                    self.event_models[k0] = self.f_class(self.d, **self.f_opts)

                # get the log likelihood for each event model
                model = self.event_models[k0]

                # detect when there is a change in event types (not the same thing as boundaries)
                current_event = (k0 == self.k_prev)

                if current_event:
                    assert self.x_prev is not None
                    lik[k0] = model.log_likelihood_next(self.x_prev, x_curr)

                    # special case for the possibility of returning to the start of the current event
                    lik_restart_event = model.log_likelihood_f0(x_curr)

                else:
                    lik[k0] = model.log_likelihood_f0(x_curr)

            # determine the event identity (without worrying about event breaks for now)
            _post = np.log(prior[:len(active)]) + lik
            if ii > 0:
                # the probability that the current event is repeated is the OR probability -- but b/c
                # we are using a MAP approximation over all possibilities, it is a max of the repeated/restarted

                # is restart higher under the current event
                restart_prob = lik_restart_event + np.log(prior[self.k_prev] - self.lmda)
                repeat_prob = _post[self.k_prev]
                _post[self.k_prev] = np.max([repeat_prob, restart_prob])

            # get the MAP cluster and only update it
            k = np.argmax(_post)  # MAP cluster

            # determine whether there was a boundary
            event_boundary = (k != self.k_prev) or ((k == self.k_prev) and (restart_prob > repeat_prob))

            # calculate the event boundary probability
            _post[self.k_prev] = restart_prob
            log_boundary_probability[ii] = logsumexp(_post) - logsumexp(np.concatenate([_post, [repeat_prob]]))

            # calculate the probability of an event label, ignoring the event boundaries
            if self.k_prev is not None:
                _post[self.k_prev] = logsumexp([restart_prob, repeat_prob])
                prior[self.k_prev] -= self.lmda / 2.
                lik[self.k_prev] = logsumexp(np.array([lik[self.k_prev], lik_restart_event]))

                # now, the normalized posterior
                p = np.log(prior[:len(active)]) + lik - np.max(lik)   # subtracting the max doesn't change proportionality
                post[ii, :len(active)] = np.exp(p - logsumexp(p))

                # this is a diagnostic readout and does not effect the model
                log_like[ii, :len(active)] = lik
                log_prior[ii, :len(active)] = np.log(prior[:len(active)])
            else:
                log_like[ii, 0] = 0.0
                log_prior[ii, 0] = self.alfa
                post[ii, 0] = 1.0

            # prediction error: euclidean distance of the last model and the current scene vector
            if ii > 0:
                model = self.event_models[self.k_prev]
                y_hat[ii, :] = model.predict_next(self.x_prev)
                pe[ii] = np.linalg.norm(x_curr - y_hat[ii, :])

            self.c[k] += 1  # update counts
            # update event model
            if not event_boundary:
                # we're in the same event -> update using previous scene
                assert self.x_prev is not None
                self.event_models[k].update(self.x_prev, x_curr)
            else:
                # we're in a new event token -> update the initialization point only
                self.event_models[k].new_token()
                self.event_models[k].update_f0(x_curr)

            self.x_prev = x_curr  # store the current scene for next trial
            self.k_prev = k  # store the current event for the next trial

        if minimize_memory:
            self.clear_event_models()
            self.results = Results()
            self.results.log_post = log_like + log_prior
            return

        self.results = Results()
        self.results.post = post
        self.results.pe = pe
        self.results.log_like = log_like
        self.results.log_prior = log_prior
        self.results.e_hat = np.argmax(log_like + log_prior, axis=1)
        self.results.y_hat = y_hat
        self.results.log_loss = logsumexp(log_like + log_prior, axis=1)
        self.results.log_boundary_probability = log_boundary_probability
        # # this is a debugging thing
        self.results.restart_prob = restart_prob
        self.results.repeat_prob = repeat_prob

        return post

    def update_single_event(self, x, update=True):
        """

        :param x: this is an n x d array of the n scenes in an event
        :param update: boolean (default True) update the prior and posterior of the event model
        :return:
        """
        if update:
            self.k += 1
            self._update_state(x, self.k)

            # pull the relevant items from the results
            if self.results is None:
                self.results = Results()
                post = np.zeros((1, self.k))
                log_like = np.zeros((1, self.k)) - np.inf
                log_prior = np.zeros((1, self.k)) - np.inf

            else:
                post = self.results.post
                log_like = self.results.log_like
                log_prior = self.results.log_prior

                # extend the size of the posterior, etc

                n, k0 = np.shape(post)
                while k0 < self.k:
                    post = np.concatenate([post, np.zeros((n, 1))], axis=1)
                    log_like = np.concatenate([log_like, np.zeros((n, 1))], axis=1)
                    log_prior = np.concatenate([log_prior, np.zeros((n, 1))], axis=1)
                    n, k0 = np.shape(post)

                # extend the size of the posterior, etc
                post = np.concatenate([post, np.zeros((1, self.k))], axis=0)
                log_like = np.concatenate([log_like, np.zeros((1, self.k)) - np.inf], axis=0)
                log_prior = np.concatenate([log_prior, np.zeros((1, self.k)) - np.inf], axis=0)
        else:
            log_like = np.zeros((1, self.k)) - np.inf
            log_prior = np.zeros((1, self.k)) - np.inf

        # calculate sCRP prior
        prior = self._calculate_unnormed_sCRP(self.k_prev)

        # likelihood
        active = np.nonzero(prior)[0]
        lik = np.zeros((np.shape(x)[0], len(active)))

        # again, this is a readout of the model only and not used for updating,
        # but also keep track of the within event posterior
        map_prediction = np.zeros(np.shape(x))
        k_within_event = np.argmax(prior)  # prior to the first scene within an event having been observed, the
        # prior determines what the event type will be

        for ii, x_curr in enumerate(x):

            # we need to maintain a distribution over possible event types for the current events --
            # this gets locked down after termination of the event.
            # Also: none of the event models can be updated until *after* the event has been observed

            # special case the first scene within the event
            if ii == 0:
                event_boundary = True
            else:
                event_boundary = False

            # loop through each potentially active event model
            for k0 in active:
                if k0 not in self.event_models.keys():
                    self.event_models[k0] = self.f_class(self.d, **self.f_opts)

                # get the log likelihood for each event model
                model = self.event_models[k0]

                if not event_boundary:
                    lik[ii, k0] = model.log_likelihood_sequence(x[:ii, :], x_curr)
                else:
                    lik[ii, k0] = model.log_likelihood_f0(x_curr)

            if event_boundary:
                map_prediction[ii, :] = self.event_models[k_within_event].predict_f0()
            else:
                map_prediction[ii, :] = self.event_models[k_within_event].predict_next_generative(x[:ii, :])

            # for the purpose of calculating a prediction error and a prediction error only, calculate
            # a within event estimate of the event type (the real estimate is at the end of the event,
            # taking into account the accumulated evidence
            k_within_event = np.argmax(np.sum(lik[:ii+1, :len(active)], axis=0) + np.log(prior[:len(active)]))

        # cache the diagnostic measures
        log_like[-1, :len(active)] = np.sum(lik, axis=0)
        log_prior[-1, :len(active)] = np.log(prior[:len(active)])

        # calculate surprise
        bayesian_surprise = logsumexp(lik + np.tile(log_prior[-1, :len(active)], (np.shape(lik)[0], 1)), axis=1)

        if update:

            # at the end of the event, find the winning model!
            log_post = log_prior[-1, :len(active)] + log_like[-1, :len(active)]
            post[-1, :len(active)] = np.exp(log_post - logsumexp(log_post))
            k = np.argmax(log_post)

            # update the prior
            self.c[k] += np.shape(x)[0]
            # cache for next event
            self.k_prev = k

            # update the winning model's estimate
            self.event_models[k].update_f0(x[0])
            x_prev = x[0]
            for X0 in x[1:]:
                self.event_models[k].update(x_prev, X0)
                x_prev = X0

            # self.results = Results()
            self.results.post = post
            self.results.log_like = log_like
            self.results.log_prior = log_prior
            self.results.e_hat = np.argmax(post, axis=1)
            self.results.log_loss = logsumexp(log_like + log_prior, axis=1)

        return bayesian_surprise, map_prediction

    def init_for_boundaries(self, list_events):
        # update internal state

        k = len(list_events)
        self._update_state(np.concatenate(list_events, axis=0), k)
        del k  # use self.k and self.d

        if self.k_prev is None:
            self.event_models[0] = self.f_class(self.d, **self.f_opts)  # initialize the first event model

    def run_w_boundaries(self, list_events, progress_bar=True, leave_progress_bar=True):
        """
        This method is the same as the above except the event boundaries are pre-specified by the experimenter
        as a list of event tokens (the event/schema type is still inferred).

        One difference is that the event token-type association is bound at the last scene of an event type.
        N.B. ! also, all of the updating is done at the event-token level.  There is no updating within an event!

        evaluate the probability of each event over the whole token


        Parameters
        ----------
        list_events: list of n x d arrays -- each an event


        progress_bar: bool
            use a tqdm progress bar?

        leave_progress_bar: bool
            leave the progress bar after completing?

        Return
        ------
        post: n_e by k array of posterior probabilities

        """

        # loop through the other events in the list
        if progress_bar:
            def my_it(iterator):
                return tqdm(iterator, desc='Run SEM', leave=leave_progress_bar)
        else:
            def my_it(iterator):
                return iterator

        self.init_for_boundaries(list_events)

        for x in my_it(list_events):
            self.update_single_event(x)

    def clear_event_models(self):
        self.event_models = None
        K.clear_session()