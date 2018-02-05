import pandas as pd
import numpy as np
import models
import cPickle as pickle
from sklearn.metrics import adjusted_rand_score


def generate_random_events(n_events, data_file=None):
    """

    Parameters
    ----------
    n_events: int

    data_file: str
        full file path of the Reynolds, Braver, & Zachs data.
        contains pandas dataframe (pickled) with 13 events of
        8-12 time-points and 54 dimensions

    :return:
    """

    if data_file is None:
        data_file = './datasets/motion_data.pkl'
    motion_data = pd.read_pickle(data_file)
    n_patterns = len(set(motion_data.EventNumber))

    X = []
    y = []
    for _ in range(n_events):
        p = np.random.randint(n_patterns)
        e = motion_data.loc[motion_data.EventNumber == p, :].values[:, :-1]
        X.append(e)
        y.append([p] * e.shape[0])
    return np.concatenate(X), np.concatenate(y)


def evaluate(X, y, Omega, K=None, number=0, save=True, return_pe=False, split_post=False, list_event_boundaries=None,
             semclass=models.SEM):
    """

    Parameters
    ----------
    X: NxD array
        scene vectors

    y: array of length N
        true class labels

    Omega: dict
        dictionary of kwargs for the SEM model

    K: int
        maximum number of clusters


    Return
    ------
        r: int, adjusted rand score
    """

    sem = semclass(**Omega)

    if K is None:
        K = X.shape[0] / 2

    if return_pe:
        if split_post:
            post, pe, log_lik, log_prior = sem.run(X, K=K, return_pe=True, return_lik_prior=True,
                                                   list_event_boundaries=list_event_boundaries)
        else:
            post, pe = sem.run(X, K=K, return_pe=True)
    else:
        if split_post:
            post, _, log_lik, log_prior = sem.run(X, K=K, return_pe=True, return_lik_prior=True,
                                                  list_event_boundaries=list_event_boundaries)
        else:
            post = sem.run(X, K=K)

    y_hat = np.argmax(post, axis=1)

    r = adjusted_rand_score(y, y_hat)

    if save:
        f = open('SEM_sample_%d.save' % number, 'wb')

        pickle.dump({'AdjRandScore': r, 'Omega': Omega}, f)
        f.close()
        return

    if return_pe:
        if split_post:
            return r, post, pe, log_lik, log_prior
        return r, post, pe
    if split_post:
        return r, post, _, log_lik, log_prior

    return r, post


# generate random string
#
def randstr(N=10):
    import string
    import random
    return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(N))


if __name__ == '__main__':
    pass
