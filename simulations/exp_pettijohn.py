import numpy as np
import pandas as pd
from models.sem import SEM, clear_sem
from sklearn.metrics import adjusted_rand_score
from models.memory import reconstruction_accuracy, evaluate_seg
from models.memory import multichain_gibbs as gibbs_memory_sampler


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

    # run through with the switch condition
    sem_switch = SEM(**sem_kwargs)
    sem_switch.run_w_boundaries(list_events=x_list_switch, leave_progress_bar=False)

    # create the corrupted memory traces
    y_mem_switch = list()  # these are list, not sets, for hashability
    y_mem_noswitch = list()  # these are list, not sets, for hashability

    for t in range(n):
        # n.b. python uses stdev, not var
        x_mem = x_list_switch[t / 10][t % 10, :] + np.random.normal(scale=gibbs_kwargs['tau'] ** 0.5, size=d) 
        e_mem = [None, sem_switch.event_models.keys()[t / (n / 2)]][np.random.rand() < epsilon_e]
        t_mem = t + np.random.randint(-gibbs_kwargs['b'], gibbs_kwargs['b'] + 1)
        y_mem_switch.append([x_mem, e_mem, t_mem])

        # do the no-switch condition ahead of time
        e_mem = [None, 0][np.random.rand() < epsilon_e]
        y_mem_noswitch.append([x_mem, e_mem, t_mem])

    # sample from memory
    gibbs_kwargs['y_mem'] = y_mem_switch
    gibbs_kwargs['sem_model'] = sem_switch
    y_samples, e_samples, _ = gibbs_memory_sampler(**gibbs_kwargs)

    results = pd.DataFrame({
        'Condition': 'Shift',
        'r2': adjusted_rand_score(sem_switch.results.e_hat, np.array([0, 1])),
        'Reconstruction Segementation': evaluate_seg(e_samples, np.concatenate([[e0] * 10 for e0 in sem_switch.event_models])),
        'Overall Acc': reconstruction_accuracy(y_samples, y_mem_switch).mean(),
        'Non-boundary Acc': evaluate_bound_acc(y_samples, y_mem_switch),
        'Boundary Acc': evaluate_non_bound_acc(y_samples, y_mem_switch),
        'Batch': [batch_number],
    }, index=[batch_number])
    clear_sem(sem_switch)
    sem_switch = None

    # run through with the no-switch condition
    sem_no_switch = SEM(**sem_kwargs)
    sem_no_switch.run_w_boundaries(list_events=x_list_no_switch, leave_progress_bar=False)

    gibbs_kwargs['y_mem'] = y_mem_noswitch
    gibbs_kwargs['sem_model'] = sem_no_switch
    y_samples, e_samples, x_samples = gibbs_memory_sampler(**gibbs_kwargs)

    results = pd.concat([results, pd.DataFrame({
        'Condition': 'No-Shift',
        'Overall Acc': reconstruction_accuracy(y_samples, y_mem_noswitch).mean(),
        'Non-boundary Acc': evaluate_bound_acc(y_samples, y_mem_noswitch),
        'Boundary Acc': evaluate_non_bound_acc(y_samples, y_mem_noswitch),
        'Batch': [batch_number],
    }, index=[batch_number])], sort=True)
    clear_sem(sem_no_switch)
    sem_no_switch = None

    return results