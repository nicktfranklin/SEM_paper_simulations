import numpy as np
from tqdm import tqdm
import sys
sys.path.append('./')
sys.path.append('../')
from models.memory import gibbs_memory_sampler
from opt import encode
import pandas as pd
from models import *


# construct a library that is reused
#

def make_task(d=25, n_rooms=10):
    constant = np.sqrt(d) * 1.1

    # note: in the experiment there were 66 events and 51 probes
    verbs = {v: np.random.randn(1, d) / constant for v in 'enter put_down pick_up leave'.split()}
    objects_a = {ii: np.random.randn(1, d) / constant for ii in range(n_rooms)}
    objects_b = {ii: np.random.randn(1, d) / constant for ii in range(n_rooms)}
    ctx = {ii: np.random.randn(1, d) / constant for ii in range(n_rooms)}

    # to control the variance of the embedded verbs, each is bound to the same null token if
    # the sentence has not object
    null_token = np.random.randn(1, d) / constant

    list_events = []

    list_objects = []
    for ii in range(n_rooms):
        #         event = np.tile(ctx[ii], (4, 1))
        event = np.zeros((4, d))
        event += np.concatenate([
            np.tile(objects_a[ii], (2, 1)),
            np.tile(objects_b[ii], (2, 1))
        ])
        event += np.concatenate([
            encode(verbs['enter'], null_token),
            encode(verbs['put_down'], objects_a[ii]),
            encode(verbs['pick_up'], null_token),
            encode(verbs['leave'], null_token),
        ])
        list_events.append(event)
        list_objects.append([objects_a[ii], objects_b[ii]])

    return list_events, list_objects

def eval_acc(y_samples, y_mem):
    acc = []
    for y_sample in y_samples:
        def item_acc(t):
            return np.float(any([all(yt[0] == y_mem[t][0]) for yt in y_sample if yt != None]))

        # evaluate the accuracy of the boundary items (here, items 10 and 11)
        acc.append([item_acc(t) for t in range(4)])
    return np.mean(acc, axis=0)


def batch_pre(sem_model, gibbs_kwargs):
    acc = []

    list_events, list_objects = make_task()

    for itt, x in tqdm(enumerate(list_events), leave=False, total=len(list_events), desc='Pre-boundary loop'):

        sem_model.update_single_event(x)
        n_items, d = np.shape(x)

        # create a corrupted memory trace, conditioned on the correct event label
        y_mem = list()
        for t in range(n_items):
            x_mem = x[t, :] + np.random.randn(d) * gibbs_kwargs['tau']
            e_mem = sem_model.results.e_hat[-1]  # condition on the correct event model
            t_mem = t + np.random.randint(-gibbs_kwargs['b'], gibbs_kwargs['b'] + 1)
            y_mem.append([x_mem, e_mem, t_mem])

        # reconstruct
        gibbs_kwargs['y_mem'] = y_mem
        gibbs_kwargs['sem'] = sem_model
        gibbs_kwargs['e_true'] = [sem_model.results.e_hat[-1]] * n_items
        y_samples, x_samples = gibbs_memory_sampler_given_e(**gibbs_kwargs)

        item_acc = eval_acc(y_samples=y_samples, y_mem=y_mem)

        obj_a, obj_b = list_objects[itt]
        foil = np.concatenate([
            obj_a[0, :d / 2].reshape(1, -1),
            obj_b[0, d / 2:].reshape(1, -1)
        ], axis=1)

        acc.append({
            'half 1': item_acc[[0, 1]].mean(),
            'half 2': item_acc[[2, 3]].mean(),
            'Probe A': np.mean(np.sum((obj_a - np.reshape(x_samples, (-1, d))) ** 2, axis=1) ** 0.5),
            'Probe B': np.mean(np.sum((obj_b - np.reshape(x_samples, (-1, d))) ** 2, axis=1) ** 0.5),
            'Foil': np.mean(np.sum((foil - np.reshape(x_samples, (-1, d))) ** 2, axis=1) ** 0.5),
        })
    return pd.DataFrame(acc)


def batch_post(sem_model, gibbs_kwargs, epsilon_e=0.25):

    gibbs_kwargs = {k: v for k, v in gibbs_kwargs.iteritems() if k != 'e_true'}

    acc = []

    list_events, list_objects = make_task()

    y_mem = list()
    for itt, x in tqdm(enumerate(list_events), leave=False, total=len(list_events), desc='Post-boundary loop'):

        sem_model.update_single_event(x)
        n_items, d = np.shape(x)

        # create a corrupted memory trace, conditioned on the correct event label
        for t in range(n_items):
            x_mem = x[t, :] + np.random.randn(d) * gibbs_kwargs['tau']

            e_mem = [None, sem_model.event_models.keys()[-1]][np.random.rand() > epsilon_e]
            t_mem = t + np.random.randint(-gibbs_kwargs['b'], gibbs_kwargs['b'] + 1)
            y_mem.append([x_mem, e_mem, t_mem])

        # # for speed, just reconstruct the past 3 events at max
        # if len(y_mem) > 3 * 4:
        #     y_mem = y_mem[-12:]

        # reconstruct
        gibbs_kwargs['y_mem'] = y_mem
        gibbs_kwargs['sem'] = sem_model
        y_samples, e_samples, x_samples = gibbs_memory_sampler(**gibbs_kwargs)
        x_samples = np.array(x_samples)
        #         print np.shape(x_samples)

        item_acc = eval_acc(y_samples=y_samples, y_mem=y_mem)

        obj_a, obj_b = list_objects[itt]
        foil = np.concatenate([
            obj_a[0, :d / 2].reshape(1, -1),
            obj_b[0, d / 2:].reshape(1, -1)
        ], axis=1)

        acc.append({
            'half 1': item_acc[[0, 1]].mean(),
            'half 2': item_acc[[2, 3]].mean(),
            'Probe A': np.mean(np.sum((obj_a - np.reshape(x_samples[:, -4:, :], (-1, d))) ** 2, axis=1) ** 0.5),
            'Probe B': np.mean(np.sum((obj_b - np.reshape(x_samples[:, -4:, :], (-1, d))) ** 2, axis=1) ** 0.5),
            'Foil': np.mean(np.sum((foil - np.reshape(x_samples[:, -4:, :], (-1, d))) ** 2, axis=1) ** 0.5),
        })
    return pd.DataFrame(acc)


if __name__ == "__main__":
    df0 = 100
    scale0 = .2

    lmda = 1.0  # stickiness parameter
    alfa = 1.0  # concentration parameter

    f_class = KerasGRU
    f_opts = dict(var_scale0=scale0, var_df0=df0)

    # create the corrupted memory trace
    # noise parameters
    b = 2
    tau = 0.1

    sem_kwargs = dict(
        lmda=lmda, alfa=alfa, f_class=f_class,
        f_opts=f_opts
    )

    # experiment parameters

    n_batch = 1

    res = []
    for ii in tqdm(range(n_batch)):
        gibbs_kwargs = dict(
            #         memory_alpha = 0.1,
            #         memory_lambda = 1.0,
            memory_epsilon=np.exp(-11),
            b=b,  # re-defined here for completeness
            tau=tau,  # ibid
            n_samples=250,
            n_burnin=100,
            progress_bar=False,
        )

        sem = SEM(**sem_kwargs)
        res0 = batch_pre(sem, gibbs_kwargs)
        res0['Itteration'] = ii
        res0['Condition'] = 'Pre-Boundary Test'
        if len(res) > 0:
            res = pd.concat([res, res0])
        else:
            res = res0

        gibbs_kwargs['memory_alpha'] = alfa
        gibbs_kwargs['memory_lambda'] = lmda

        sem = SEM(**sem_kwargs)
        res0 = batch_post(sem, gibbs_kwargs, epsilon_e=0.1)
        res0['Itteration'] = ii
        res0['Condition'] = 'Post-Boundary Test'
        res = pd.concat([res, res0])

        res.to_pickle('sims_radvansky_2.pkl')