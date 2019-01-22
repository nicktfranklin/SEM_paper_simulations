import numpy as np
from tqdm import tqdm
import sys
sys.path.append('./')
sys.path.append('../')
from opt import encode
import pandas as pd
from models.memory import gibbs_memory_sampler
from scipy.special import logsumexp


def make_task(d=25, n_rooms=15):
    # note: in the experiment there were 66 events and 51 probes
    verbs = {v: np.random.randn(1, d) / np.sqrt(d) for v in 'enter put_down pick_up leave'.split()}
    objects_a = {ii: np.random.randn(1, d) / np.sqrt(d) for ii in range(n_rooms)}
    objects_b = {ii: np.random.randn(1, d) / np.sqrt(d) for ii in range(n_rooms)}
    ctx = {ii: np.random.randn(1, d) / np.sqrt(d) for ii in range(n_rooms)}

    # to control the variance of the embedded verbs, each is bound to the same null token if
    # the sentence has not object
    null_token = np.random.randn(1, d) / np.sqrt(d)

    list_events = []

    list_objects = []
    for ii in range(n_rooms):
        event = np.tile(ctx[ii], (4, 1))
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


def eval_probe(x_samples, probe):
    acc = []
    for y_sample in y_samples:
        def item_acc(t):
            return np.float(any([all(yt[0] == y_mem[t][0]) for yt in y_sample if yt != None]))

        # evaluate the accuracy of the boundary items (here, items 10 and 11)
        acc.append([item_acc(t) for t in range(4)])
    return np.mean(acc, axis=0)


def batch(sem, gibbs_kwargs, epsilon_e_switch=0.25, epsilon_e_noswitch=0.75, gamma=2.5, n_rooms=25):
    gibbs_kwargs = {k: v for k, v in gibbs_kwargs.iteritems() if k != 'e_true'}

    acc = []
    list_events, list_objects = make_task(n_rooms=n_rooms)
    sem.init_for_boundaries(list_events)

    y_mem_switch = list()
    for itt, x in tqdm(enumerate(list_events), leave=False, total=len(list_events)):

        sem.update_single_event(x)
        n_items, d = np.shape(x)

        # create a corrupted memory trace for the switch condition
        y_mem_noswitch = [yi for yi in y_mem_switch]
        for t in range(n_items):
            x_mem = x[t, :] + np.random.randn(d) * gibbs_kwargs['tau']
            e_mem = [None, sem.event_models.keys()[-1]][np.random.rand() < epsilon_e_switch]
            t_mem = t + np.random.randint(-gibbs_kwargs['b'], gibbs_kwargs['b'] + 1)
            y_mem_switch.append([x_mem, e_mem, t_mem])

            # for the no-switch condition
            e_mem = [None, sem.event_models.keys()[-1]][np.random.rand() < epsilon_e_noswitch]
            y_mem_noswitch.append([x_mem, e_mem, t_mem])

        # for speed, just reconstruct the past 3 events at max
        if len(y_mem_switch) > 2 * 4:
            y_mem_switch = y_mem_switch[-8:]
            y_mem_noswitch = y_mem_noswitch[-8:]

        # reconstruct (Switch)
        gibbs_kwargs['y_mem'] = y_mem_switch
        gibbs_kwargs['sem'] = sem
        y_samples, e_samples, x_samples = gibbs_memory_sampler(**gibbs_kwargs)
        x_samples = np.array(x_samples)

        item_acc = eval_acc(y_samples=y_samples, y_mem=y_mem_switch)

        # evaluate the probability of the associated vs dissociated items
        obj_a, obj_b = list_objects[itt]
        x_samples_ii = np.reshape(x_samples[:, -4:, :], (-1, d))
        p_a_greater_than_b = \
            -logsumexp(-np.linalg.norm(x_samples_ii - obj_a, axis=1) * gamma) < \
            -logsumexp(-np.linalg.norm(x_samples_ii - obj_b, axis=1) * gamma)

        # use the correct scoring method
        acc.append({
            'Condition': 'Switch',
            'Accuracy': item_acc.mean(),
            'Pr(A > B)': p_a_greater_than_b,
        })

        # reconstruct (No-Switch)
        gibbs_kwargs['y_mem'] = y_mem_noswitch
        y_samples, e_samples, x_samples = gibbs_memory_sampler(**gibbs_kwargs)
        x_samples = np.array(x_samples)
        item_acc = eval_acc(y_samples=y_samples, y_mem=y_mem_noswitch)

        # evaluate the probability of the associated vs dissociated items
        obj_a, obj_b = list_objects[itt]
        x_samples_ii = np.reshape(x_samples[:, -4:, :], (-1, d))
        p_a_greater_than_b = \
            -logsumexp(-np.linalg.norm(x_samples_ii - obj_a, axis=1) * gamma) < \
            -logsumexp(-np.linalg.norm(x_samples_ii - obj_b, axis=1) * gamma)

        # use the correct scoring method
        acc.append({
            'Condition': 'No-Switch',
            'Accuracy': item_acc.mean(),
            'Pr(A > B)': p_a_greater_than_b,
        })

    # clear SEM from memory
    sem.clear_event_models()
    sem = None

    return pd.DataFrame(acc)


if __name__ == "__main__":
    pass