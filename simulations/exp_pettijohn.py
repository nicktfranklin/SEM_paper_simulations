import numpy as np
import pandas as pd
from models import *
from sklearn.metrics import adjusted_rand_score

from models.memory import gibbs_memory_sampler


def generate_task(n=20, d=25):
    items = np.random.randn(n, d)

    no_switch_context = np.tile(np.random.randn(1, d), (n, 1))
    switch_context = np.concatenate([np.tile(np.random.randn(1, d), (n / 2, 1)),
                                     np.tile(np.random.randn(1, d), (n / 2, 1))], axis=0)

    x_noswitch = items + no_switch_context
    x_switch = items + switch_context

    x_noswitch /= np.sqrt(d)
    x_switch /= np.sqrt(d)

    # break the stimuli into two lists for one set of stim and one list for the other
    x_list_no_switch = [x_noswitch]
    x_list_switch = [x_switch[:n / 2, :], x_switch[n / 2:, :]]
    return x_list_no_switch, x_list_switch


def evaluate_seg(e_samples, e_true):
    acc = []
    for e in e_samples:
        acc.append(np.mean(np.array(e) == e_true))
    return np.mean(acc)


def eval_acc(y_samples, y_mem):
    acc = []
    for y_sample in y_samples:
        def item_acc(t):
            return np.float(any([all(yt[0] == y_mem[t][0]) for yt in y_sample if yt != None]))

        # evaluate the accuracy of the boundary items (here, items 10 and 11)
        acc.append(np.mean([item_acc(t) for t in range(20)]))
    return np.mean(acc)


def evaluate_bound_acc(y_samples, y_mem):
    acc = []
    for y_sample in y_samples:
        #
        def item_acc(t):
            return np.float(any([all(yt[0] == y_mem[t][0]) for yt in y_sample if yt != None]))

        # evaluate the accuracy of the boundary items (here, items 10 and 11)
        acc.append(np.mean([item_acc(t) for t in [10, 11]]))
    return np.mean(acc)


def evaluate_non_bound_acc(y_samples, y_mem):
    acc = []
    for y_sample in y_samples:
        def item_acc(t):
            return np.float(any([all(yt[0] == y_mem[t][0]) for yt in y_sample if yt != None]))

        # evaluate the accuracy of the boundary items (here, items 10 and 11)
        acc.append(np.mean([item_acc(t) for t in range(20) if (t != 10) & (t != 11)]))
    return np.mean(acc)


def batch(sem_kwargs, gibbs_kwargs, epsilon_e, batch_number=0):

    x_list_no_switch, x_list_switch = generate_task()
    n, d = np.concatenate(x_list_switch).shape

    sem_no_switch = SEM(**sem_kwargs)
    sem_no_switch.run_w_boundaries(list_events=x_list_no_switch, leave_progress_bar=False)

    sem_switch = SEM(**sem_kwargs)
    sem_switch.run_w_boundaries(list_events=x_list_switch, leave_progress_bar=False)

    # create the corrupted memory traces
    y_mem_switch = list()  # these are list, not sets, for hashability
    y_mem_noswitch = list()  # these are list, not sets, for hashability

    for t in range(n):
        x_mem = x_list_switch[t / 10][t % 10, :] + np.random.randn(d) * gibbs_kwargs['tau']
        e_mem = [None, sem_switch.event_models.keys()[t / (n / 2)]][np.random.rand() < epsilon_e]
        t_mem = t + np.random.randint(-gibbs_kwargs['b'], gibbs_kwargs['b'] + 1)
        y_mem_switch.append([x_mem, e_mem, t_mem])

        e_mem = [None, 0][np.random.rand() < epsilon_e]
        y_mem_noswitch.append([x_mem, e_mem, t_mem])


    # sample from memory
    gibbs_kwargs['y_mem'] = y_mem_switch
    gibbs_kwargs['sem'] = sem_switch
    y_samples, e_samples, x_samples = gibbs_memory_sampler(**gibbs_kwargs)


    results = pd.DataFrame({
        'Condition': 'Shift',
        'r2': adjusted_rand_score(sem_switch.results.e_hat, np.array([0, 1])),
        'Reconstruction Segementation': evaluate_seg(e_samples,
                                                     np.concatenate([[e0] * 10 for e0 in sem_switch.event_models])),
        'Overall Acc': eval_acc(y_samples, y_mem_switch),
        'Non-boundary Acc': evaluate_bound_acc(y_samples, y_mem_switch),
        'Boundary Acc': evaluate_non_bound_acc(y_samples, y_mem_switch),
        'Batch': [batch_number],
    }, index=[batch_number])

    gibbs_kwargs['y_mem'] = y_mem_noswitch
    gibbs_kwargs['sem'] = sem_no_switch
    y_samples, e_samples, x_samples = gibbs_memory_sampler(**gibbs_kwargs)

    results = pd.concat([results, pd.DataFrame({
        'Condition': 'No-Shift',
        'Overall Acc': eval_acc(y_samples, y_mem_noswitch),
        'Non-boundary Acc': evaluate_bound_acc(y_samples, y_mem_noswitch),
        'Boundary Acc': evaluate_non_bound_acc(y_samples, y_mem_noswitch),
        'Batch': [batch_number],
    }, index=[batch_number])])

    return results


def batch_reduced(sem_kwargs, gibbs_kwargs, batch_number=0):

    x_list_no_switch, x_list_switch = generate_task()
    n, d = np.concatenate(x_list_switch).shape

    sem_no_switch = SEM(**sem_kwargs)
    sem_no_switch.run_w_boundaries(list_events=x_list_no_switch, leave_progress_bar=False)

    sem_switch = SEM(**sem_kwargs)
    sem_switch.run_w_boundaries(list_events=x_list_switch, leave_progress_bar=False)

    # create the corrupted memor traces
    y_mem_switch = list()  # these are list, not sets, for hashability
    y_mem_noswitch = list()  # these are list, not sets, for hashability

    for t in range(n):
        t_mem = t + np.random.randint(-gibbs_kwargs['b'], gibbs_kwargs['b'] + 1)
        y_mem_switch.append([x_list_switch[t / 10][t % 10, :] + np.random.randn(d) * gibbs_kwargs['tau'], t_mem])
        y_mem_noswitch.append([x_list_no_switch[0][t, :] + np.random.randn(d) * gibbs_kwargs['tau'], t_mem])

    # sample from memory

    gibbs_kwargs['y_mem'] = y_mem_switch
    gibbs_kwargs['sem'] = sem_switch
    gibbs_kwargs['e_true'] = np.concatenate([[ii] * 10 for ii in sem_switch.results.e_hat])
    y_samples, x_samples = gibbs_memory_sampler_reduced(**gibbs_kwargs)

    results = pd.DataFrame({
        'Condition': 'Shift',
        'Overall Acc': eval_acc(y_samples, y_mem_switch),
        'Non-boundary Acc': evaluate_bound_acc(y_samples, y_mem_switch),
        'Boundary Acc': evaluate_non_bound_acc(y_samples, y_mem_switch),
        'Batch': [batch_number],
    }, index=[batch_number])

    gibbs_kwargs['y_mem'] = y_mem_noswitch
    gibbs_kwargs['sem'] = sem_no_switch
    gibbs_kwargs['e_true'] = np.concatenate([[ii] * 20 for ii in sem_no_switch.results.e_hat])
    y_samples, x_samples = gibbs_memory_sampler_reduced(**gibbs_kwargs)

    results = pd.concat([results, pd.DataFrame({
        'Condition': 'No-Shift',
        'Overall Acc': eval_acc(y_samples, y_mem_noswitch),
        'Non-boundary Acc': evaluate_bound_acc(y_samples, y_mem_noswitch),
        'Boundary Acc': evaluate_non_bound_acc(y_samples, y_mem_noswitch),
        'Batch': [batch_number],
    }, index=[batch_number])])

    return results


def batch_switch_only(sem_kwargs, gibbs_kwargs, epsilon_e, batch_number=0):

    _, x_list_switch = generate_task()
    n, d = np.concatenate(x_list_switch).shape

    sem_switch = SEM(**sem_kwargs)
    sem_switch.run_w_boundaries(list_events=x_list_switch, leave_progress_bar=False)

    # create the corrupted memory traces
    y_mem_switch = list()  # these are list, not sets, for hashability

    for t in range(n):
        x_mem = x_list_switch[t / 10][t % 10, :] + np.random.randn(d) * gibbs_kwargs['tau']
        e_mem = [None, sem_switch.event_models.keys()[t / (n / 2)]][np.random.rand() < epsilon_e]
        t_mem = t + np.random.randint(-gibbs_kwargs['b'], gibbs_kwargs['b'] + 1)
        y_mem_switch.append([x_mem, e_mem, t_mem])

    # sample from memory

    gibbs_kwargs['y_mem'] = y_mem_switch
    gibbs_kwargs['sem'] = sem_switch
    y_samples, e_samples, x_samples = gibbs_memory_sampler(**gibbs_kwargs)

    results = pd.DataFrame({
        'Condition': 'Shift',
        'r2': adjusted_rand_score(sem_switch.results.e_hat, np.array([0, 1])),
        'Reconstruction Segementation': evaluate_seg(e_samples,
                                                     np.concatenate([[e0] * 10 for e0 in sem_switch.event_models])),
        'Overall Acc': eval_acc(y_samples, y_mem_switch),
        'Non-boundary Acc': evaluate_bound_acc(y_samples, y_mem_switch),
        'Boundary Acc': evaluate_non_bound_acc(y_samples, y_mem_switch),
        'Batch': [batch_number],
    }, index=[batch_number])

    return results