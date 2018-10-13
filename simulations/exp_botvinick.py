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

def get_seq_from_sample(x_samples, x_true, s):


    seqs = []
    for x_sample in x_samples:
        # create a permuation of all of the original scene items and calculate the distance
        # Note: by definition the permutation sequence [0, 1, 2, 3, 4, 5] is always the correct
        # order and corresponds to the ordering [s0, s1, s2, s3, s4, s5].
        all_orders = list(permutations(range(0, 6)))
        distances = [np.linalg.norm(x_sample.reshape(-1) - x_true[order, :].reshape(-1)) for order in all_orders]
        permutation_order = all_orders[np.argmin(distances)]

        seqs.append([s[ii] for ii in permutation_order])
    return seqs


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
        gibbs_kwargs['e_true'] = [sem.results.e_hat[-1]] * n_items
        y_samples, x_samples = gibbs_memory_sampler_given_e(**gibbs_kwargs)

        # reconstruct the sequence from the sample!
        # seqs = get_seq_from_sample(y_samples, y_mem, s)
        seqs = get_seq_from_sample(x_samples, x, s)

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


def main():
    # SEM parameters
    df0 = 20
    scale0 = 3.

    mode = df0 * scale0 / (df0 + 2)
    print("Prior variance (mode): {}".format(mode))
    s = generate_exp()
    x = vectorize_seq(s)
    print("Median Feature variance: {}".format(np.median(np.var(x, axis=0))))

    lmda = 1.0  # stickyness parameter
    alfa = 1.  # concentration parameter

    # f_class = KerasLSTM
    f_class = KerasMultiLayerPerceptron
    f_opts = dict(var_scale0=scale0, var_df0=df0)

    # create the corrupted memory trace
    # noise parameters
    b = 1
    tau = 0.05
    print("tau: {}".format(tau))

    # set the parameters for the Gibbs sampler
    gibbs_kwargs = dict(
        #     memory_alpha = 0.1,
        #     memory_lambda = 1.0,
        memory_epsilon=np.exp(-11),
        b=b,  # re-defined here for completeness
        tau=tau,  # ibid
        n_samples=250,
        n_burnin=100,
        progress_bar=False,
    )

    sem_kwargs = dict(
        lmda=lmda, alfa=alfa, f_class=f_class,
        f_opts=f_opts
    )

    # exp[eriment parameters

    n_sess = 1
    trials_per_sess = 750
    n_test = 250
    n_batch = 25

    def batch_exp(iteration=0):
        sem = SEM(**sem_kwargs)
        pretrain_sess(sem, generate_exp, n_stim=n_sess * trials_per_sess)
        goodness, acc, err_goodness, propor_regularization = batch(sem, generate_exp, gibbs_kwargs, n_test=n_test)

        df = pd.DataFrame({
            'Stimulus goodness': goodness,
            'Proportion correct': acc,
            'Error Response goodness (mean)': err_goodness,
            'Proportion regularizations': propor_regularization,
            'iteration': [iteration] * len(goodness),
        })
        return df

    res = []
    for ii in tqdm(n_batch, desc='OuterLoop', leave=True):
        res.append(batch_exp(ii))
    res = pd.concat(res)


    res.to_pickle('exp_botvinick_sim.pkl')

if __name__ == '__main__':
    main()