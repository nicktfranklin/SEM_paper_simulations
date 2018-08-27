import pandas as pd
import numpy as np
from models.sem import SEM
from models.event_models import *
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm
from tqdm import tqdm
from scipy.special import logsumexp


def segment_compare(f_class, f_opts, lmda=10 ** 5, alfa=10 ** -1, downsample=1):
    """
    :param f_class: (EventModel) the event model class
    :param f_opts:  (dict) all of the keyword arguments for the event model class
    :param lmda:    (float) stickiness parameter in sCRP
    :param alfa:    (float) concentration parameter in sCRP
    :return:
    """
    Z = np.load('data/videodata/video_color_Z_embedded_64.npy')

    # the "Sax" movie is from time slices 0 to 5537
    sax = Z[range(0, 5537, downsample), :]

    Omega = {
        'lmda': lmda,  # Stickiness (prior)
        'alfa': alfa,  # Concentration parameter (prior)
        'f_class': f_class,
        'f_opts': f_opts
    }

    sem_model = SEM(**Omega)
    sem_model.run(sax, K=sax.shape[0], leave_progress_bar=False)

    # compare the model segmentation to human data via regression and store the output

    e_hat = sem_model.results.e_hat
    e_idx = np.zeros(np.shape(sem_model.results.post), dtype=bool)
    for ii, e in enumerate(e_hat):
        e_idx[ii, e] = True
    boundary_probability = 1 - np.exp(sem_model.results.log_like[e_idx] + sem_model.results.log_prior[e_idx])

    # bin the boundary probability!
    def prep_compare(bin_size):

        frame_time = np.arange(1, len(boundary_probability) + 1) / (30.0 / downsample)

        index = np.arange(0, np.max(frame_time), bin_size)
        sem_binned = []
        for t in index:
            sem_binned.append(boundary_probability[(frame_time >= t) & (frame_time < (t + bin_size))].max())
        sem_binned = pd.Series(sem_binned, index=index)

        # bin the subject data
        young_warned_sax = pd.read_csv('data/zachs_2006_young_unwarned.csv', header=-1)
        young_warned_sax.set_index(0, inplace=True)
        yw_binned = []
        for t in index:
            l = young_warned_sax[(young_warned_sax.index > t) &
                                 (young_warned_sax.index < (t + bin_size))]
            if l.empty:
                yw_binned.append(0)
            else:
                yw_binned.append(l.max().values[0])

        yw_binned = pd.Series(np.array(yw_binned), index=index)
        X = sem_binned.values
        y = yw_binned.values
        return X, y

    def model_r2(x, y):
        x = (x - np.mean(x)) / np.std(x)
        y = (y - np.mean(y)) / np.std(y)
        sm.add_constant(x)
        results = OLS(y, x).fit()
        return results.rsquared

    r2s = []
    bins = np.arange(0.20, 8.1, 0.2)
    n_permute = 500
    r2s_rand = np.zeros((n_permute, len(bins)))

    for ii, b in enumerate(bins):
        x, y = prep_compare(b)
        r2s.append(model_r2(x, y))
        for jj in range(n_permute):
            np.random.shuffle(x)
            r2s_rand[jj, ii] = model_r2(x, y)

    return pd.DataFrame({
        'Bin Size': bins,
        'Model R2': np.array(r2s),
        'Permutation Mean': r2s_rand.mean(axis=0),
        'Permutation Std': r2s_rand.std(axis=0),
        'Permutation p-value': np.mean(r2s_rand - (np.tile(np.array(r2s), (n_permute, 1))) > 0, axis=0),
        'LogLoss': np.sum(sem_model.results.log_loss)
    })


def main(n_batch=8, df0=10, scale0=0.3, l2_regularization=0.3, dropout=0.1, t=10, downsample=1):


    ####### DP-GMM Events #########
    f_class = Gaussian
    f_opts = dict(var_df0=df0, var_scale0=scale0)

    res = []
    for ii in tqdm(range(n_batch), desc='Running DP-GMM', total=n_batch):
        res0 = segment_compare(f_class, f_opts, downsample=downsample)
        res0['batch'] = [ii] * len(res0)
        res0['EventModel'] = ['DP-GMM'] * len(res0)
        res.append(res0)

    res = pd.concat(res)
    file_name = 'data/video_results/EventR2_DPGMM_batched_df0_{}_scale0_{}_l2_{}_do_{}.pkl'.format(
        df0,scale0,l2_regularization,dropout
    )
    res.to_pickle(file_name)

    ####### Gaussian Random Walk #########
    f_class = GaussianRandomWalk
    f_opts = dict(var_df0=df0, var_scale0=scale0)

    res = []
    for ii in tqdm(range(n_batch), desc='Running RandomWalk', total=n_batch):
        res0 = segment_compare(f_class, f_opts, downsample=downsample)
        res0['batch'] = [ii] * len(res0)
        res0['EventModel'] = ['RandomWalk'] * len(res0)
        res.append(res0)
    res = pd.concat(res)
    file_name = 'data/video_results/EventR2_RandWalk_batched_df0_{}_scale0_{}_l2_{}_do_{}.pkl'.format(
        df0, scale0, l2_regularization, dropout
    )
    res.to_pickle(file_name)

    ####### LDS Events #########
    f_class = KerasLDS
    f_opts = dict(var_df0=df0, var_scale0=scale0, l2_regularization=l2_regularization)

    res = []
    for ii in tqdm(range(n_batch), desc='Running LDS', total=n_batch):
        res0 = segment_compare(f_class, f_opts, downsample=downsample)
        res0['batch'] = [ii] * len(res0)
        res0['EventModel'] = ['LDS'] * len(res0)
        res.append(res0)
    res = pd.concat(res)
    file_name = 'data/video_results/EventR2_LDS_batched_df0_{}_scale0_{}_l2_{}_do_{}.pkl'.format(
        df0,scale0,l2_regularization,dropout
    )
    res.to_pickle(file_name)

    ####### MLP Events #########
    f_class = KerasMultiLayerPerceptron
    f_opts = dict(var_df0=df0, var_scale0=scale0, l2_regularization=l2_regularization,
                  dropout=dropout)
    res = []
    for ii in tqdm(range(n_batch), desc='Running MLP', total=n_batch):
        res0 = segment_compare(f_class, f_opts, downsample=downsample)
        res0['batch'] = [ii] * len(res0)
        res0['EventModel'] = ['MLP'] * len(res0)
        res.append(res0)
    res = pd.concat(res)
    file_name = 'data/video_results/EventR2_MLP_batched_df0_{}_scale0_{}_l2_{}_do_{}.pkl'.format(
        df0,scale0,l2_regularization,dropout
    )
    res.to_pickle(file_name)

    ####### SRN Events #########
    f_class = KerasRecurrentMLP
    f_opts = dict(var_df0=df0, var_scale0=scale0, l2_regularization=l2_regularization,
                  dropout=dropout, t=t)
    res = []
    for ii in tqdm(range(n_batch), desc='Running SRN', total=n_batch):
        res0 = segment_compare(f_class, f_opts, downsample=downsample)
        res0['batch'] = [ii] * len(res0)
        res0['EventModel'] = ['SRN'] * len(res0)
        res.append(res0)
    res = pd.concat(res)
    file_name = 'data/video_results/EventR2_SRN_batched_df0_{}_scale0_{}_l2_{}_do_{}.pkl'.format(
        df0,scale0,l2_regularization,dropout
    )
    res.to_pickle(file_name)

    ####### GRU #########
    f_class = KerasGRU
    f_opts = dict(var_df0=df0, var_scale0=scale0, l2_regularization=l2_regularization,
                  dropout=dropout, t=t)
    res = []
    for ii in tqdm(range(n_batch), desc='Running GRU', total=n_batch):
        res0 = segment_compare(f_class, f_opts, downsample=downsample)
        res0['batch'] = [ii] * len(res0)
        res0['EventModel'] = ['GRU'] * len(res0)
        res.append(res0)
    res = pd.concat(res)
    file_name = 'data/video_results/EventR2_GRU_batched_df0_{}_scale0_{}_l2_{}_do_{}.pkl'.format(
        df0, scale0, l2_regularization, dropout
    )
    res.to_pickle(file_name)

    ####### LSTM #########
    f_class = KerasLSTM
    f_opts = dict(var_df0=df0, var_scale0=scale0, l2_regularization=l2_regularization,
                  dropout=dropout, t=t)
    res = []
    for ii in tqdm(range(n_batch), desc='Running LSTM', total=n_batch):
        res0 = segment_compare(f_class, f_opts, downsample=downsample)
        res0['batch'] = [ii] * len(res0)
        res0['EventModel'] = ['LSTM'] * len(res0)
        res.append(res0)
    res = pd.concat(res)
    file_name = 'data/video_results/EventR2_LSTM_batched_df0_{}_scale0_{}_l2_{}_do_{}.pkl'.format(
        df0, scale0, l2_regularization, dropout
    )
    res.to_pickle(file_name)

if __name__ == "__main__":
    main(downsample=1, n_batch=1)