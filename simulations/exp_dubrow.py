import pandas as pd
import numpy as np
from models import *
from tqdm import tqdm
from sklearn.metrics import adjusted_rand_score


def generate_experiment(seed=None, scaling_factor=1.0, event_duration=5, n_events=5, d=25):

    n = event_duration * n_events

    if seed:
        np.random.seed(seed)

    x = np.random.randn(n, d)
    e = np.zeros(n, dtype=int)

    # embed a similarity structure within the items of each category
    # by adding the same random vector to all of the items within the
    # category
    categ_one = (np.random.randn(1, d)) * scaling_factor
    categ_two = (np.random.randn(1, d)) * scaling_factor

    for ii in range(n_events):
        if ii % 2 == 0:
            x[ii * event_duration:ii * event_duration + event_duration, :] += categ_one
            e[ii * event_duration:ii * event_duration + event_duration] = 0
        else:
            x[ii * event_duration:ii * event_duration + event_duration, :] += categ_two
            e[ii * event_duration:ii * event_duration + event_duration] = 1

    x /= np.sqrt(d)

    # give the model boundaries....
    e_tokens = np.concatenate([[False], e[1:] != e[:-1]]).cumsum()
    x_list_items = []
    for e0 in set(e_tokens):
        x_list_items.append(x[e0 == e_tokens, :])

    return x_list_items, e_tokens


# diagnostics functions

def evaluate_seg(e_samples, e_true):
    acc = []
    for e in e_samples:
        acc.append(np.mean(np.array(e) == e_true))
    return np.mean(acc)


def hash_y(y):
    if y is not None:
        return np.concatenate([y[0], [y[1]], [y[2]]])
    else:
        return y


def eval_acc(y_samples, y_mem):
    acc = []
    for y_sample in y_samples:
        def item_acc(t0):
            return np.float(any([all(hash_y(yt) == hash_y(y_mem[t0])) for yt in y_sample]))
        # evaluate the accuracy of the boundary items (here, items 10 and 11)
        acc.append(np.mean([item_acc(t) for t in range(20)]))
    return np.mean(acc)


def evaluate_item_position_acc(y_samples, y_mem, t):
    acc = []
    for y_sample in y_samples:
        def item_acc(t0):
            return np.float(any([all(hash_y(yt) == hash_y(y_mem[t0])) for yt in y_sample]))
        acc.append(item_acc(t))
    return np.mean(acc)


def eval_item_acc(y_samples, y_mem, times):
    acc = []
    for y_sample in y_samples:
        def item_acc(t0):
            return np.float(any([all(hash_y(yt) == hash_y(y_mem[t0])) for yt in y_sample]))
        # evaluate the accuracy of the boundary items (here, items 10 and 11)
        acc.append(np.mean([item_acc(t) for t in times]))
    return np.mean(acc)


def score_transitions(y_samples, y_mem, t):
    acc = []
    idx = np.arange(len(y_mem))
    for y_sample in y_samples:
        y_t = [all(hash_y(y0) == hash_y(y_mem[t])) for y0 in y_sample]
        y_t1 = [all(hash_y(y0) == hash_y(y_mem[t - 1])) for y0 in y_sample]
        # position accuracy is conditioned on recall
        if any(y_t):
            if any(y_t1):
                acc.append(idx[y_t][0] == (idx[y_t1][0] + 1))
            else:
                acc.append(False)
    return np.mean(acc)


def batch(sem_kwargs, gibbs_kwargs, epsilon_e, batch_n=0):

    # generate an experiment
    x_list_items, e_tokens = generate_experiment()
    n, d = np.concatenate(x_list_items).shape

    pre_locs = [ii for ii in range(len(e_tokens) - 1) if e_tokens[ii] != e_tokens[ii + 1]]
    pst_locs = [ii for ii in range(1, len(e_tokens)) if e_tokens[ii] != e_tokens[ii - 1]]

    # Train SEM on the stimuli
    sem = SEM(**sem_kwargs)
    sem.run_w_boundaries(list_events=x_list_items, leave_progress_bar=False)

    e_seg = np.reshape([[ii] * np.sum(e_tokens == t, dtype=int) for t, ii in enumerate(sem.results.e_hat)], -1)

    # create the corrupted memory trace
    y_mem = list()  # these are list, not sets, for hashability

    for t in range(n):
        x_mem = np.concatenate(x_list_items)[t, :] + np.random.randn(d) * gibbs_kwargs['tau']
        e_mem = [None, e_seg[t]][np.random.rand() < epsilon_e]
        t_mem = t + np.random.randint(-gibbs_kwargs['b'], gibbs_kwargs['b'] + 1)
        y_mem.append([x_mem, e_mem, t_mem])

    # add the models to the kwargs
    y_samples, e_samples, x_samples = gibbs_memory_sampler(y_mem, sem, **gibbs_kwargs)

    results = pd.DataFrame({
        'Batch': [batch_n],
        'Adj-r2': [adjusted_rand_score(sem.results.e_hat, np.array([0, 1, 0, 1, 0]))],
        'Recon Segment': evaluate_seg(e_samples, e_seg),
        'Overall Acc': eval_acc(y_samples, y_mem),
        'Pre-Boundary': np.mean([evaluate_item_position_acc(y_samples, y_mem, t) for t in pre_locs]),
        'Boundary': np.mean([evaluate_item_position_acc(y_samples, y_mem, t) for t in pst_locs]),
        'Transitions Pre-Boundary': np.mean([score_transitions(y_samples, y_mem, t) for t in pre_locs]),
        'Transitions Boundary': np.mean([score_transitions(y_samples, y_mem, t) for t in pst_locs]),
        'Pre-boundary Acc': eval_item_acc(y_samples, y_mem, pre_locs),
        'Boundary Acc': eval_item_acc(y_samples, y_mem, pst_locs),
    })
    return results


# for speed should generate random parameters
def gen_random_params():
    df0 = np.random.randint(1, 100)
    scale0 = np.random.uniform(0, 1.0) * np.exp(np.random.uniform(-7, 0))

    alpha = np.exp(np.random.uniform(-2, 2))
    lmda = np.random.uniform(0, 25)
    log_epsilon = np.random.uniform(-20, 6)
    b = np.random.randint(2, 5)
    log_tau = np.random.uniform(-10, 0)

    sem_kwargs = dict(
        lmda=1.0,
        alfa=1.0,
        f_class=KerasMultiLayerPerceptron,
        f_opts=dict(var_scale0=scale0, var_df0=df0)
    )

    gibbs_kwargs = dict(
        memory_alpha=alpha,
        memory_lambda=lmda,
        memory_epsilon=np.exp(log_epsilon),
        b=b,  # time corruption
        tau=np.exp(log_tau),  # feature corruption
        n_samples=250,
        n_burnin=100,
        leave_progress_bar=False,
    )
    return df0,  scale0, sem_kwargs, gibbs_kwargs


def main(n_batch=8):

    df0, scale0, sem_kwargs, gibbs_kwargs = gen_random_params()

    results = []
    for ii in tqdm(range(n_batch), desc='Running Batch'):
        results.append(batch(sem_kwargs, gibbs_kwargs, batch_n=ii))
    results = pd.concat(results)

    results.to_pickle('data/seqmem/SeqMem_df_{}_scale_{}_malfa_{}_mlmda_{}_logepsilon_{}_b_{}_tau_{}.pkl'.format(
        df0,
        scale0,
        np.log(gibbs_kwargs['memory_alpha']),
        gibbs_kwargs['memory_lambda'],
        np.log(gibbs_kwargs['memory_epsilon']),
        gibbs_kwargs['b'],
        np.log(gibbs_kwargs['tau'])
    ))


if __name__ == "__main__":
    main()
