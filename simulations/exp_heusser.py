import pandas as pd
import numpy as np
from models import *
from tqdm import tqdm
from sklearn.metrics import adjusted_rand_score


def generate_experiment(seed=None, scaling_factor=1.0, event_duration=6, n_events=6, d=25,
                        color_one=None, color_two=None):

    n = event_duration * n_events

    if seed:
        np.random.seed(seed)

    x = np.random.randn(n, d)
    e = np.zeros(n, dtype=int)

    # embed a similarity structure within the items of each category
    # by adding the same random vector to all of the items within the
    # category
    if color_one is None:
        color_one = (np.random.randn(1, d)) * scaling_factor
        color_two = (np.random.randn(1, d)) * scaling_factor

    for ii in range(n_events):
        if ii % 2 == 0:
            x[ii * event_duration:ii * event_duration + event_duration, :] += color_one
            e[ii * event_duration:ii * event_duration + event_duration] = 0
        else:
            x[ii * event_duration:ii * event_duration + event_duration, :] += color_two
            e[ii * event_duration:ii * event_duration + event_duration] = 1

    x /= np.sqrt(d)

    # give the model boundaries....
    e_tokens = np.concatenate([[False], e[1:] != e[:-1]]).cumsum()
    x_list_items = []
    for e0 in set(e_tokens):
        x_list_items.append(x[e0 == e_tokens, :])

    return x_list_items, e_tokens, color_one, color_two


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


def get_time_estimate(y_mem_t, y_samples):
    hashed_y_mem_t = hash_y(y_mem_t)
    t = []
    for y_sample in y_samples:
        try:
            t.append(np.arange(36)[[all(hashed_y_mem_t == hash_y(y0)) for y0 in y_sample]][0])
        except:
            t.append(None)
    return t


def compare_two_time(t0, t1, y_samples, y_mem):
    y0 = get_time_estimate(y_mem[t0], y_samples)
    y1 = get_time_estimate(y_mem[t1], y_samples)

    out = []
    for y00, y10 in zip(y0, y1):
        if y00 is None:
            if y10 is None:
                out.append(0.5)
            else:
                out.append(1 - float(y10) / len(y_mem))
        elif y10 is None:
            out.append(float(y00) / len(y_mem))

        else:
            out.append(float(y00 < y10))
    return out


def single_subj(sem_kwargs, gibbs_kwargs, epsilon_e, batch_n=0, n_lists=16):
    # code starts here

    sem = SEM(**sem_kwargs)

    results = []
    color_one = None
    color_two = None
    for jj in tqdm(range(n_lists), desc='Single Subject'):
        # generate an experiment
        x_list_items, e_tokens, color_one, color_two = generate_experiment(color_one=color_one, color_two=color_two)
        n, d = np.concatenate(x_list_items).shape

        sem.run_w_boundaries(list_events=x_list_items, progress_bar=False)

        e_seg = np.reshape([[ii] * np.sum(e_tokens == t, dtype=int) for t, ii in enumerate(sem.results.e_hat[-6:])], -1)

        # create the corrupted memory trace
        y_mem = list()  # these are list, not sets, for hashability

        for t in range(n):
            x_mem = np.concatenate(x_list_items)[t, :] + np.random.randn(d) * gibbs_kwargs['tau']
            e_mem = [None, e_seg[t]][np.random.rand() < epsilon_e]
            t_mem = t + np.random.randint(-gibbs_kwargs['b'], gibbs_kwargs['b'] + 1)
            y_mem.append([x_mem, e_mem, t_mem])

        # add the models to the kwargs
        y_samples, e_samples, x_samples = gibbs_memory_sampler(y_mem, sem, **gibbs_kwargs)

        # get the object-color memory
        c1 = np.linalg.norm(x_samples - color_one, axis=2)
        c2 = np.linalg.norm(x_samples - color_two, axis=2)

        c = np.concatenate([
            c1[:, 0:6] > c2[:, 0:6],
            c1[:, 6:12] < c2[:, 6:12],
            c1[:, 12:18] > c2[:, 12:18],
            c1[:, 18:24] < c2[:, 18:24],
            c1[:, 24:30] > c2[:, 24:30],
            c1[:, 30:36] < c2[:, 30:36],
        ], axis=1)
        obj_color_boundary = np.mean([np.mean(c, axis=0)[ii] for ii in np.arange(0, 36, 6)])
        obj_color_nonbound = np.mean([np.mean(c, axis=0)[ii] for ii in np.arange(36) if ii % 6 != 0])

        # get the order memory
        # test pairs for temporal order
        boundary_items = np.arange(0, 36, 6)[1:]
        non_boundary = (np.arange(0, 36, 6) - 1)[1:]

        order_boundary = np.mean([np.mean(compare_two_time(ii, ii - 4, y_samples, y_mem)) for ii in boundary_items])
        order_nonbound = np.mean([np.mean(compare_two_time(ii, ii - 4, y_samples, y_mem)) for ii in non_boundary])

        results.append({
            'Batch': batch_n,
            'List Number': jj,
            'Adj-r2': adjusted_rand_score(sem.results.e_hat[-6:], np.array([0, 1, 0, 1, 0, 1])),
            'Recon Segment': evaluate_seg(e_samples, e_seg),
            'Overall Acc': eval_acc(y_samples, y_mem),
            'Temporal Order, Boundary': order_boundary,
            'Temporal Order, NonBound': order_nonbound,
            'Object-Color, Boundary': obj_color_boundary,
            'Object-Color, NonBound': obj_color_nonbound
        })

    return pd.DataFrame(results)


def single_subj_debug(sem_kwargs, gibbs_kwargs, epsilon_e, batch_n=0, n_lists=16):
    # code starts here

    sem = SEM(**sem_kwargs)

    results = []
    color_one = None
    color_two = None
    for jj in tqdm(range(n_lists), desc='Single Subject'):
        # generate an experiment
        x_list_items, e_tokens, color_one, color_two = generate_experiment(color_one=color_one, color_two=color_two)
        n, d = np.concatenate(x_list_items).shape

        sem.run_w_boundaries(list_events=x_list_items, progress_bar=False)

        e_seg = np.reshape([[ii] * np.sum(e_tokens == t, dtype=int) for t, ii in enumerate(sem.results.e_hat[-6:])], -1)

        # create the corrupted memory trace
        y_mem = list()  # these are list, not sets, for hashability

        for t in range(n):
            x_mem = np.concatenate(x_list_items)[t, :] + np.random.randn(d) * gibbs_kwargs['tau']
            e_mem = [None, e_seg[t]][np.random.rand() < epsilon_e]
            t_mem = t + np.random.randint(-gibbs_kwargs['b'], gibbs_kwargs['b'] + 1)
            y_mem.append([x_mem, e_mem, t_mem])

        # add the models to the kwargs
        y_samples, e_samples, x_samples = gibbs_memory_sampler(y_mem, sem, **gibbs_kwargs)

        # get the object-color memory
        c1 = np.linalg.norm(x_samples - color_one, axis=2)
        c2 = np.linalg.norm(x_samples - color_two, axis=2)

        c = np.concatenate([
            c1[:, 0:6] > c2[:, 0:6],
            c1[:, 6:12] < c2[:, 6:12],
            c1[:, 12:18] > c2[:, 12:18],
            c1[:, 18:24] < c2[:, 18:24],
            c1[:, 24:30] > c2[:, 24:30],
            c1[:, 30:36] < c2[:, 30:36],
        ], axis=1)
        obj_color_boundary = np.mean([np.mean(c, axis=0)[ii] for ii in np.arange(0, 36, 6)])
        obj_color_nonbound = np.mean([np.mean(c, axis=0)[ii] for ii in np.arange(36) if ii % 6 != 0])

        # get the order memory
        # test pairs for temporal order
        boundary_items = np.arange(0, 36, 6)[1:]
        non_boundary = (np.arange(0, 36, 6) - 1)[1:]

        order_boundary = np.mean([np.mean(compare_two_time(ii, ii - 4, y_samples, y_mem)) for ii in boundary_items])
        order_nonbound = np.mean([np.mean(compare_two_time(ii, ii - 4, y_samples, y_mem)) for ii in non_boundary])

        results.append({
            'Batch': batch_n,
            'List Number': jj,
            'Adj-r2': adjusted_rand_score(sem.results.e_hat[-6:], np.array([0, 1, 0, 1, 0, 1])),
            'Recon Segment': evaluate_seg(e_samples, e_seg),
            'Overall Acc': eval_acc(y_samples, y_mem),
            'Temporal Order, Boundary': order_boundary,
            'Temporal Order, NonBound': order_nonbound,
            'Object-Color, Boundary': obj_color_boundary,
            'Object-Color, NonBound': obj_color_nonbound
        })

    return pd.DataFrame(results)

if __name__ == "__main__":
    pass