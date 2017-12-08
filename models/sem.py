import numpy as np
from scipy.misc import logsumexp
from tqdm import tnrange
from scipy.stats import multivariate_normal as mvnormal


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

        [N, D] = np.shape(X)

        Sigma = np.eye(D) * self.beta  # noise for multivariate gaussian likelihood

        if K is None:
            K = N

        if return_pe:
            pe = np.zeros(np.shape(X)[0])

        C = np.zeros(K)  # running count of the clustering process

        event_models = dict()  # initialize an empty event model space

        X_null = np.zeros(D)  # need a starting location as it's a sequential model
        X_prev = X_null # last scene
        k_prev = None # last event type
        # X_curr = X[0, :]
        post = np.zeros((N, K))
        k_active = None

        # debugging function
        if return_lik_prior:
            log_like = np.zeros((N, K))
            log_prior = np.zeros((N, K))

        for n in tnrange(N):
            X_curr = X[n, :].copy()

            # calculate sCRP prior
            prior = C.copy()
            idx = len(np.nonzero(C)[0])  # get number of visited clusters

            if idx < K:
                prior[idx] = self.alfa  # set new cluster probability to alpha

            # add stickiness parameter for n>0, only for the previously chosen event
            if n > 0:
                prior[np.argmax(post[n-1, :])] += self.lmda

            prior /= np.sum(prior)

            # likelihood
            active = np.nonzero(prior)[0]
            lik = np.zeros(len(active))

            for k in active:
                if k not in event_models.keys():
                    event_models[k] = self.f_class(D, **self.f_opts)

                # get the log likelihood for each event model
                model = event_models[k]

                if k == k_active:
                    Y_hat = model.predict_next(X_prev)
                else:
                    Y_hat = model.predict_f0()
                lik[k] = mvnormal.logpdf(X_curr - Y_hat, mean=np.zeros(D), cov=Sigma)

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
                model = event_models[k_prev]
                pe[n] = np.linalg.norm(X_curr - model.predict_next(X_prev))

            C[k] += 1  # update counts

            # update event model
            if k_prev == k:
                # we're in the same event -> update using previous scene
                event_models[k].update(X_prev, X_curr)
            else:
                # we're in a new event -> update the initialization point only
                event_models[k].new_cluster()
                event_models[k].update_f0(X_curr)

            #print 'scene ', n, ' map = ', k

            X_prev = X_curr  # store the current vector for next trial
            k_prev = k

        # after all of the training, close the models!
        for m in event_models.itervalues():
            m.close()

        if return_pe:
            if return_lik_prior:
                return post, pe, log_like, log_prior
            return post, pe

        if return_lik_prior:
            return post, None, log_like, log_prior

        return post
