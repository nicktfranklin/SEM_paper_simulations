import numpy as np
from tqdm import tqdm
import sys
sys.path.append('./')
sys.path.append('../')
from opt import encode
import pandas as pd
from models.memory import reconstruction_accuracy, evaluate_seg
from models.memory import gibbs_memory_sampler
from scipy.special import logsumexp
from sklearn.preprocessing import normalize
from models.sem import clear_sem, SEM


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
            verbs['enter'],
            objects_a[ii],
            objects_b[ii],
            verbs['leave'],
        ])
        list_events.append(event)
        list_objects.append([objects_a[ii], objects_b[ii]])

    return list_events, list_objects



def batch(sem_kwargs, gibbs_kwargs, epsilon_e_switch=0.25, epsilon_e_noswitch=0.75, 
          gamma=2.5, n_rooms=25, progress_bar=True):

    sem_model = SEM(**sem_kwargs)
    _gibbs_kwargs = {k: v for k, v in gibbs_kwargs.iteritems() if k != 'e_true'}

    acc = []
    list_events, list_objects = make_task(n_rooms=n_rooms)
    sem_model.init_for_boundaries(list_events)

    if progress_bar:
        def my_it(iterator):
            return tqdm(iterator, desc='Run SEM', leave=False, total=len(list_events))
    else:
        def my_it(iterator):
            return iterator

    y_mem_switch = list()
    for itt, x in my_it(enumerate(list_events)):

        sem_model.update_single_event(x)
        n_items, d = np.shape(x)
        e_list = np.concatenate([[sem_model.results.e_hat[itt]] * n_items for t in range(n_rooms)])


        # create a corrupted memory trace for the switch condition
        y_mem_noswitch = [yi for yi in y_mem_switch]
        for t in range(n_items):
            x_mem = x[t, :] + np.random.normal(scale= _gibbs_kwargs['tau'] ** 0.5, size=d) # note, python uses stdev, not var
            e_mem = [None, sem_model.results.e_hat[-1]][np.random.rand() < epsilon_e_switch]
            t_mem = t + np.random.randint(-_gibbs_kwargs['b'], _gibbs_kwargs['b'] + 1)
            y_mem_switch.append([x_mem, e_mem, t_mem])

            # for the no-switch condition
            e_mem = [None, sem_model.results.e_hat[-1]][np.random.rand() < epsilon_e_noswitch]
            y_mem_noswitch.append([x_mem, e_mem, t_mem])

        # for speed, just reconstruct the past 3 events at max
        if len(y_mem_switch) > 3 * 2:
            y_mem_switch = y_mem_switch[-6:]
            y_mem_noswitch = y_mem_noswitch[-6:]
            e_list = e_list[-6:]

        # reconstruct (Switch)
        _gibbs_kwargs['y_mem'] = y_mem_switch
        _gibbs_kwargs['sem_model'] = sem_model
        y_samples, e_samples, x_samples = gibbs_memory_sampler(**_gibbs_kwargs)
        x_samples = np.array(x_samples)

        item_acc = reconstruction_accuracy(y_samples=y_samples, y_mem=y_mem_switch)

        # evaluate the probability of the associated vs dissociated items
        obj_a, obj_b = list_objects[itt]
        x_samples_ii = np.reshape(x_samples[:, -2:, :], (-1, d))
        p_a_greater_than_b = \
            -logsumexp(-np.linalg.norm(x_samples_ii - obj_a, axis=1) * gamma) < \
            -logsumexp(-np.linalg.norm(x_samples_ii - obj_b, axis=1) * gamma)

        # use the correct scoring method
        acc.append({
            'Room Number': itt,
            'Condition': 'Switch',
            'Reconstruction Accuracy': item_acc.mean(),
            'Last Room Reconstruction Acc': item_acc[-2:].mean(),
            'Pr(A > B)': p_a_greater_than_b,
            'Reconstruction Segementation': evaluate_seg(e_samples, e_list),
        })
        
        # clear things from memory
        y_samples, e_samples, x_samples = None, None, None

        # reconstruct (No-Switch)
        _gibbs_kwargs['y_mem'] = y_mem_noswitch
        y_samples, e_samples, x_samples = gibbs_memory_sampler(**_gibbs_kwargs)
        x_samples = np.array(x_samples)
        item_acc = reconstruction_accuracy(y_samples=y_samples, y_mem=y_mem_noswitch)

        # evaluate the probability of the associated vs dissociated items
        obj_a, obj_b = list_objects[itt]
        x_samples_ii = np.reshape(x_samples[:, -2:, :], (-1, d))
        p_a_greater_than_b = \
            -logsumexp(-np.linalg.norm(x_samples_ii - obj_a, axis=1) * gamma) < \
            -logsumexp(-np.linalg.norm(x_samples_ii - obj_b, axis=1) * gamma)

        # use the correct scoring method
        acc.append({
            'Room Number': itt,
            'Condition': 'No-Switch',
            'Last Room Reconstruction Acc': item_acc[-2:].mean(),
            'Reconstruction Accuracy': item_acc.mean(),
            'Pr(A > B)': p_a_greater_than_b,
            'Reconstruction Segementation': evaluate_seg(e_samples, e_list),
        })
        # clear things from memory
        y_samples, e_samples, x_samples = None, None, None

    # clear SEM from memory
    clear_sem(sem_model)
    sem_model = None

    return pd.DataFrame(acc)


if __name__ == "__main__":
    pass