import numpy as np
from tqdm import tqdm
from models.memory import gibbs_memory_sampler
from models import *
import pandas as pd
from keras.backend import clear_session
from itertools import permutations

# pick 6 Gaussian random vectors, one for each list item
d = 25
n = 6
list_items = np.random.randn(n, d) / np.sqrt(d)

T = np.array([
    [0, 0.125, 0.125, 0.25, 0.25, 0.25],
    [0.125, 0, 0.125, 0.25, 0.25, 0.25],
    [0.125, 0.125, 0, 0.25, 0.25, 0.25],
    [0.25, 0.25, 0.25, 0, 0.125, 0.125],
    [0.25, 0.25, 0.25, 0.125, 0, 0.125],
    [0.25, 0.25, 0.25, 0.125, 0.125, 0],
])


def sample_t(x, available):
    pmf = np.array([T[x, ii] for ii in available])
    pmf /= pmf.sum()
    idx = np.sum(pmf.cumsum() < np.random.uniform(0, 1))
    return available[idx]


def score_goodness(X):
    ## this is the category probability

    _X = np.array(X) < 2.5

    def evaluate_array(Y):
        return all(_X == Y) | all(_X == (Y == 0))

    if evaluate_array(np.array([0, 1, 0, 1, 0, 1])):
        return 0.302

    if evaluate_array(np.array([0, 1, 0, 1, 1, 0])):
        return 0.151

    if evaluate_array(np.array([0, 1, 1, 0, 1, 0])):
        return 0.151

    if evaluate_array(np.array([0, 0, 1, 1, 0, 1])):
        return 0.075

    if evaluate_array(np.array([0, 1, 1, 0, 0, 1])):
        return 0.075

    if evaluate_array(np.array([0, 1, 0, 0, 1, 1])):
        return 0.075

    if evaluate_array(np.array([0, 0, 1, 0, 1, 1])):
        return 0.075

    if evaluate_array(np.array([0, 1, 1, 1, 0, 0])):
        return 0.038
    if evaluate_array(np.array([0, 0, 1, 1, 1, 0])):
        return 0.038
    if evaluate_array(np.array([0, 0, 0, 1, 1, 1])):
        return 0.019

    print X,

    raise (Exception)


def generate_exp():
    available = range(6)

    # randomly draw the fist item from a uniform
    X = [np.random.randint(6)]
    available.remove(X[-1])

    # use the inverse CDF method over T to sample the next
    # list item.
    for _ in range(1, 6):
        X0 = sample_t(X[-1], available)

        X.append(X0)
        available.remove(X0)

    return X


def generate_control():
    X = np.random.permutation(range(6))

    return X


def vectorize_seq(X):
    return np.concatenate([[list_items[X0, :]] for X0 in X], axis=0)


def hash_y(y):
    if y is not None:
        return np.concatenate([y[0], [y[1]], [y[2]]]).sum()
    else:
        return y


def get_seq_from_sample(y_samples, y_mem, s):
    seqs = []
    y_key = [hash_y(y_mem[ii]) for ii in range(6)]
    for y_sample in y_samples:
        def get_position(t):
            return np.arange(6)[y_key == hash_y(y_sample[t])][0]
        seqs.append([get_position(t) for t in range(6)])
    return [np.array(s)[np.array(seq0)] for seq0 in seqs]


def pretrain_sess(sem, stim_gen, n_stim=5):
    goodness = np.zeros((n_stim,))

    for n in tqdm(range(n_stim), desc='Pretraining Sess', leave=False):
        s = stim_gen()
        x = vectorize_seq(s)
        goodness[n] = score_goodness(s)

        sem.update_single_event(x)


def batch(sem, stim_gen, gibbs_kwargs, n_test=25):

    goodness = np.zeros(n_test)
    acc = np.zeros(n_test)
    err_goodness = np.zeros(n_test)
    propor_regularization = np.zeros(n_test)

    for itt in tqdm(range(n_test), desc='Test', leave=False):

        s = stim_gen()
        x = vectorize_seq(s)
        goodness[itt] = score_goodness(s)
        n_items, d = np.shape(x)

        sem.update_single_event(x)

        # create a corrupted memory trace, conditioned on the correct event label
        y_mem = list()
        for t in range(n_items):
            x_mem = x[t, :] + np.random.randn(d) * gibbs_kwargs['tau']
            e_mem = sem.results.e_hat[-1]  # condition on the correct event model
            t_mem = t + np.random.randint(-gibbs_kwargs['b'], gibbs_kwargs['b'] + 1)
            y_mem.append([x_mem, e_mem, t_mem])

        # reconstruct
        gibbs_kwargs['y_mem'] = y_mem
        gibbs_kwargs['sem'] = sem
        y_samples, e_samples, x_samples = gibbs_memory_sampler(**gibbs_kwargs)
        x_samples = np.array(x_samples)

        # reconstruct the sequence from the sample!
        seqs = get_seq_from_sample(y_samples, y_mem, s)

        # get an average goodness for errors!
        goodness_recon = np.zeros(gibbs_kwargs['n_samples'])
        acc_recon = np.zeros(gibbs_kwargs['n_samples'])
        for ii in range(gibbs_kwargs['n_samples']):

            # get it's sequence using a nearest neighbor classifier
            seq0 = seqs[ii]

            # accuracy
            acc_recon[ii] = float(np.all(np.array(seq0) == np.array(s)))

            # score it!
            goodness_recon[ii] = score_goodness(seq0)

        # we want the goodness of the error responses only
        err_goodness[itt] = np.mean(goodness_recon[acc_recon == False])

        acc[itt] = np.mean(acc_recon, dtype=float)

        propor_regularization[itt] = np.mean(goodness_recon[acc_recon == False] > goodness[itt])

    return goodness, acc, err_goodness, propor_regularization
