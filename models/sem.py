import numpy as np
from scipy.misc import logsumexp
from tqdm import tnrange
from scipy.stats import multivariate_normal as mvnormal
from scipy.optimize import minimize

def get_training_contin_clust(posterior, current_time, horizon):
    if (horizon > 1) & (current_time >= 1):
        # get the last t vectors within the current cluster...
        h0 = np.min([current_time, horizon])
        ks = np.argmax(posterior[current_time-h0:current_time+1, :], axis=1)
        ks = (ks == ks[-1]).tolist()
        t0 = 0
        while ks.pop():
            t0 += 1
            if not ks:
                break
    else:
        t0 = 1
    return t0


class SEM(object):
    """
    This port of SAM's code (done with a different programming logic)
    in python. More documentation to come!
    """

    def __init__(self, lmda=1., alfa=10.0, beta=0.1, t=0, f_class=None, f_opts=None):
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
        self.t = t  # used by recursive models

        if f_class is None:
            raise ValueError("f_model must be specified!")

        self.f_class = f_class
        self.f_opts = f_opts

    def run(self, X, K=None, return_pe=False, split_post=False):
        """
        Parameters
        ----------
        X: N x D array of

        K: int
            maximum number of clusters

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

        X_prev = np.zeros(D)  # need a starting location as is sequential model
        # X_curr = X[0, :]
        post = np.zeros((N, K))

        # debugging function
        if split_post:
            log_like = np.zeros((N, K))
            log_prior = np.zeros((N, K))

        for n in tnrange(N):
            X_curr = X[n, :].copy()

            # calculate sCRP prior
            prior = C.copy()
            idx = len(np.nonzero(C)[0])  # get number of visited clusters

            if idx < K:
                prior[idx] = self.alfa  # set new cluster probability to alpha

            # add stickiness parameter for n>0
            if n > 0:
                prior[np.argmax(post[n, :])] + self.lmda

            prior /= np.sum(prior)

            # likelihood
            active = np.nonzero(prior)[0]
            lik = np.zeros(len(active))

            for k in active:
                if k not in event_models.keys():
                    event_models[k] = self.f_class(D, **self.f_opts)

                # get the log likelihood for each event model
                model = event_models[k]

                Y_hat = model.predict_next(X_prev)
                lik[k] = mvnormal.logpdf(X_curr - Y_hat, mean=np.zeros(D), cov=Sigma)

            # posterior
            p = np.log(prior[:len(active)]) + lik - np.max(lik)  # subtracting the max like doesn't change proportionality
            post[n, :len(active)] = np.exp(p - logsumexp(p))
            # update

            if split_post:
                log_like[n, :len(active)] = lik - np.max(lik)
                log_prior[n, :len(active)] = np.log(prior[:len(active)])

            k = np.argmax(post[n, :])  # MAP cluster

            # get the euclidean distance
            if return_pe:
                model = event_models[k]
                pe[n] = np.linalg.norm(X_curr - model.predict_next(X_prev))

            C[k] += 1  # update counts
            if (X_prev.ndim > 1) & (X_prev.shape[0] > 1):
                event_models[k].update(X_prev[-1, :], X_curr)  # update event model
            else:
                event_models[k].update(X_prev, X_curr)  # update event model

            t0 = get_training_contin_clust(post, n, self.t) # get the time horizon for recursion
            # print n, self.t, t0
            # print max([0, n-t0+1]), n+1
            # print range(max([0, n-t0+1]), n+1)

            X_prev = X[max([0, n-t0+1]):n+1, :]  # store the current vector for next trial
            # print X_prev.shape
            # # update the current scene vector
            # t0 = get_training_contin_clust(post, n+1, self.t)
            # X_curr = X[max([0, n+1-t0]):n+2, :]

        # after all of the training, close the models!
        for m in event_models.itervalues():
            m.close()

        if return_pe:
            if split_post:
                return post, pe, log_like, log_prior
            return post, pe

        if split_post:
            return post, None, log_like, log_prior

        return post
