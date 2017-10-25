import numpy as np
from scipy.misc import logsumexp
from tqdm import tnrange


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

    def run(self, X, K=None):
        """
        Parameters
        ----------
        X: N x D array of

        K: int
            maximum number of clusters

        Return
        ------
        post: N by K array of posterior probabilites
        """

        [N, D] = np.shape(X)

        Sigma = np.eye(D) * self.beta  # noise for multivariate gaussian likelihood

        if K is None:
            K = N

        C = np.zeros(K)  # running count of the clustering process
        prior = C.copy()

        event_models = dict()  # initialize an empty event model space

        x_prev = np.zeros(D)  # need a starting location as is sequential model
        post = np.zeros((N, K))

        for n in tnrange(N):
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
                lik[k] = model.log_likelihood(x_prev, X[n, :], Sigma)

            # posterior
            p = np.log(prior[:len(active)]) + lik
            post[n, :len(active)] = np.exp(p - logsumexp(p))

            # update
            k = np.argmax(post[n, :])  # MAP cluster
            C[k] += 1  # update counts
            event_models[k].update(x_prev, X[n, :])  # update event model

            x_prev = X[0, :].copy()  # store the current vector for next trial

        # after all of the training, close the models!
        for m in event_models.itervalues():
            m.close()

        return post