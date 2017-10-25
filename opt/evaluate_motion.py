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


def sample_act():
    return ['tanh', None, 'relu'][np.random.randint(3)]


def gen_keras_MLP_params():
    n_hidden = np.random.randint(100)
    hidden_act = sample_act()
    return {'n_hidden': n_hidden, 'hidden_act': hidden_act}


def gen_keras_simple_RNN_params():
    n_hidden = np.random.randint(100) + 1
    hidden_act = ['tanh', None, 'relu'][np.random.randint(3)]
    t = np.random.randint(8) + 1
    return {'n_hidden': n_hidden, 'hidden_act': hidden_act, 't':t}

def gen_keras_mRNN_params():
    n_hidden1 = np.random.randint(100)
    n_hidden2 = np.random.randint(100)

    hidden_act1 = sample_act()
    hidden_act2 = sample_act()

    t = np.random.randint(8) + 1

    return {
        'n_hidden1': n_hidden1,
        'n_hidden2': n_hidden2,
        'hidden_act1': hidden_act1,
        'hidden_act2': hidden_act2,
        't':t
    }


def sample_omega():

    available_models = {
        'MLP': (models.KerasMultiLayerNN, gen_keras_MLP_params),
        'SimpleRNN': (models.KerasSimpleRNN, gen_keras_simple_RNN_params),
        'mRNN': (models.KerasRNN, gen_keras_mRNN_params)
    }

    model_name = available_models.keys()[np.random.randint(len(available_models))]

    f_opts = available_models[model_name][1]()

    # sample SGD parameters
    sgd_kwargs = {
        'lr': np.log(1 + np.exp(np.random.standard_t(4))) * 10e-2,
        'momentum': np.log(1 + np.exp(np.random.standard_t(4))) * 10e-3,
        'decay': np.log(1 + np.exp(np.random.standard_t(4))) * 10e-3,
        'nesterov': [True, False][np.random.randint(2)],
    }
    f_opts['sgd_kwargs'] = sgd_kwargs

    Omega = {
        'lmda': np.random.uniform(0, 100),
        'alfa': np.exp(np.random.standard_t(25)),
        'beta': np.random.uniform(0.00001, 0.5),
        'f_class': available_models[model_name][0],
        'f_opts': f_opts,

    }
    return Omega


def evaluate(X, y, Omega, K=None, number=0, save=True):
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

    sem = models.SEM(**Omega)

    if K is None:
        K = X.shape[0] / 2

    post = sem.run(X, K=K)

    y_hat = np.argmax(post, axis=1)

    r = adjusted_rand_score(y, y_hat)
    
    if save:
        f = open('SEM_sample_%d.save' % number, 'wb')

        pickle.dump({'AdjRandScore': r, 'Omega': Omega}, f)
        f.close()
        return
    
    return r, post

if __name__ == '__main__':

    number = np.random.randint(10e8)
    X, y = generate_random_events(50)

    Omega = sample_omega()
    evaluate(X, y, Omega, number=number)

