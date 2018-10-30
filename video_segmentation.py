import pandas as pd
import numpy as np
from models.sem import SEM
from models.event_models import *
from scipy.special import logsumexp
from tqdm import tqdm
import sys, os, gc
from scipy.stats import multivariate_normal


# helper functions
def z_score(x):
    return (x - np.mean(x)) / np.std(x)


def get_r2(x, y):
    return np.corrcoef(z_score(x), z_score(y))[0][1] ** 2


def convert_type_token(event_types):
    tokens = [0]
    for ii in range(len(event_types)-1):
        if event_types[ii] == event_types[ii+1]:
            tokens.append(tokens[-1])
        else:
            tokens.append(tokens[-1] + 1)
    return tokens


def get_event_duration(event_types, frequency=30):
    tokens = convert_type_token(event_types)
    n_tokens = np.max(tokens)+1
    lens = []
    for ii in range(n_tokens):
        lens.append(np.sum(np.array(tokens) == ii))
    return np.array(lens, dtype=float) / frequency


def load_comparison_data(data, bin_size=1.0):

    def bin_times(array, max_seconds, bin_size=bin_size):
        cumulative_binned = [np.sum(array <= t0 * 1000) for t0 in np.arange(bin_size, max_seconds + bin_size, bin_size)]
        binned = np.array(cumulative_binned)[1:] - np.array(cumulative_binned)[:-1]
        binned = np.concatenate([[cumulative_binned[0]], binned])
        return binned

    # Movie A is Saxaphone (185s long)
    # Movie B is making a bed (336s long)
    # Movie C is doing dishes (255s long)

    # here, we'll collapse over all of the groups (old, young; warned, unwarned) for now
    n_subjs = len(set(data.SubjNum))

    sax_times = np.sort(list(set(data.loc[data.Movie == 'A', 'MS']))).astype(np.float32)
    binned_sax = bin_times(sax_times, 185, 1.0) / np.float(n_subjs)

    bed_times = np.sort(list(set(data.loc[data.Movie == 'B', 'MS']))).astype(np.float32)
    binned_bed = bin_times(bed_times, 336, 1.0) / np.float(n_subjs)

    dishes_times = np.sort(list(set(data.loc[data.Movie == 'C', 'MS']))).astype(np.float32)
    binned_dishes = bin_times(dishes_times, 255, 1.0) / np.float(n_subjs)

    return binned_sax, binned_bed, binned_dishes


def segment_video(event_sequence, f_class, f_opts, lmda=10 ** 5, alfa=10 ** -1):
    """
    :param event_sequence: (NxD np.array) the sequence of N event vectors in D dimensions
    :param f_class: (EventModel) the event model class
    :param f_opts:  (dict) all of the keyword arguments for the event model class
    :param lmda:    (float) stickiness parameter in sCRP
    :param alfa:    (float) concentration parameter in sCRP
    :return:
    """

    Omega = {
        'lmda': lmda,  # Stickiness (prior)
        'alfa': alfa,  # Concentration parameter (prior)
        'f_class': f_class,
        'f_opts': f_opts
    }

    sem_model = SEM(**Omega)
    sem_model.run(event_sequence, k=event_sequence.shape[0], leave_progress_bar=True, minimize_memory=True)
    return sem_model.results.log_post


def get_binned_boundary_prop(e_hat, log_post, bin_size=1.0, frequency=30.0):
    """
    :param results: SEM.Results
    :param bin_size: seconds
    :param frequency: in Hz
    :return:
    """

    # normalize
    log_post0 = log_post - np.tile(np.max(log_post, axis=1).reshape(-1, 1), (1, log_post.shape[1]))
    log_post0 -= np.tile(logsumexp(log_post0, axis=1).reshape(-1, 1), (1, log_post.shape[1]))

    boundary_probability = [0]
    for ii in range(1, log_post0.shape[0]):
        idx = range(log_post0.shape[0])
        idx.remove(e_hat[ii - 1])
        boundary_probability.append(logsumexp(log_post0[ii, idx]))
    boundary_probability = np.array(boundary_probability)

    frame_time = np.arange(1, len(boundary_probability) + 1) / float(frequency)

    index = np.arange(0, np.max(frame_time), bin_size)
    boundary_probability_binned = []
    for t in index:
        boundary_probability_binned.append(
            # note: this operation is equivalent to the log of the average boundary probability in the window
            logsumexp(boundary_probability[(frame_time >= t) & (frame_time < (t + bin_size))]) - \
            np.log(bin_size * 30.)
        )
    boundary_probability_binned = pd.Series(boundary_probability_binned, index=index)
    return boundary_probability_binned


def get_binned_boundaries(e_hat, bin_size=1.0, frequency=30.0):
    frame_time = np.arange(1, len(e_hat) + 1) / float(frequency)
    index = np.arange(0, np.max(frame_time), bin_size)

    boundaries = np.concatenate([[0], e_hat[1:] !=e_hat[:-1]])

    boundaries_binned = []
    for t in index:
        boundaries_binned.append(np.sum(
            boundaries[(frame_time >= t) & (frame_time < (t + bin_size))]
        ))
    return np.array(boundaries_binned)


def get_point_biserial(boundaries_binned, binned_comp):
    print np.shape(boundaries_binned), np.shape(binned_comp)
    M_1 = np.mean(binned_comp[boundaries_binned == 1])
    M_0 = np.mean(binned_comp[boundaries_binned == 0])

    n_1 = np.sum(boundaries_binned == 1)
    n_0 = np.sum(boundaries_binned == 0)
    n = n_1 + n_0

    s = np.std(binned_comp)
    print M_1, M_0, n_1, n_0, n, s

    r_pb = (M_1 - M_0) / s * np.sqrt(n_1 * n_0 / (n**2))
    return r_pb


def make_log_prob_prior(Z, df0, scale0):
    mode = df0 * scale0 / (df0 + 2)
    return multivariate_normal.logpdf(np.mean(Z, axis=0), mean=np.zeros(Z.shape[1]), cov=np.eye(Z.shape[1]) * mode)


def seg_comp_new(f_class, f_opts, lmda=10 ** 5, alfa=10 ** -1, bin_size=1.0, n_permute=100,
                 human_data_path=None, video_data_path=None,
                 ):

    if video_data_path is None:
        video_data_path = './'

    if human_data_path is None:
        human_data_path = './'

    # load the raw data
    sax = np.load(video_data_path + 'video_color_Z_embedded_64.npy')[0:5537, :]

    # adjust the prior of a new cluster!
    f_opts['prior_log_prob'] = make_log_prob_prior(sax, df0=f_opts['var_df0'], scale0=f_opts['var_scale0'])

    # load the comparison data
    # experiment 1:
    data = pd.read_csv(human_data_path + 'data021011.dat', delimiter='\t')
    binned_sax, binned_bed, binned_dishes = load_comparison_data(data)

    # for memory management, put these into a method
    def get_summary_stats(data):
        log_post = segment_video(data, f_class, f_opts, lmda, alfa)

        e_hat = np.argmax(log_post, axis=1)
        binned_prob = get_binned_boundary_prop(e_hat, log_post, bin_size=bin_size)
        avg_duration = np.mean(get_event_duration(e_hat))
        log_loss = np.sum(logsumexp(log_post, axis=1))
        binned_boundaries = get_binned_boundaries(e_hat, bin_size=bin_size)

        return binned_prob, log_loss, avg_duration, binned_boundaries

    # evaluate the model
    binned_sax_prob, sax_loss, sax_duration, binned_sax_bounds = get_summary_stats(sax)
    sax = None
    # print "r_pb: {}".format(get_point_biserial(binned_sax_bounds, binned_sax))

    bed = np.load(video_data_path + 'video_color_Z_embedded_64.npy')[5537:5537 + 10071, :]
    binned_bed_prob, bed_loss, bed_duration, binned_bed_bounds = get_summary_stats(bed)
    bed = None

    dishes = np.load(video_data_path + 'video_color_Z_embedded_64.npy')[5537 + 10071: 5537 + 10071 + 7633, :]
    binned_dishes_prob, dishes_loss, dishes_duration, binned_dishes_bounds = get_summary_stats(dishes)
    dishes = None

    # concatenate all of the data to caluclate the r2 values
    binned_comp_data = np.concatenate([binned_sax, binned_bed, binned_dishes])
    binned_model_prob = np.concatenate([binned_sax_prob, binned_bed_prob, binned_dishes_prob])
    r2 = get_r2(binned_comp_data, binned_model_prob)

    # calculate the point-biserial correlation
    binned_bounds = np.concatenate([binned_sax_bounds, binned_bed_bounds, binned_dishes_bounds])
    binned_comp_bounds = np.concatenate([binned_sax, binned_bed, binned_dishes])
    r_bp = get_point_biserial(binned_bounds, binned_comp_bounds)

    # run permutation test
    r2_permuted = [None] * n_permute
    for jj in range(n_permute):

        # permute each of the three videos' model probability
        np.random.shuffle(binned_sax_prob)
        np.random.shuffle(binned_bed_prob)
        np.random.shuffle(binned_dishes_prob)
        binned_model_prob = np.concatenate([binned_sax_prob, binned_bed_prob, binned_dishes_prob])
        r2_permuted[jj] = get_r2(binned_comp_data, binned_model_prob)

    return pd.DataFrame({
        'Bin Size': bin_size,
        'Event Length (Sax)': sax_duration,
        'Event Length (Bed)': bed_duration,
        'Event Length (Dishes)': dishes_duration,
        'Model r2': r2,
        'Permutation Mean': np.mean(r2_permuted),
        'Permutation Std': np.std(r2_permuted),
        'Permutation p-value': np.mean((r2 - np.array(r2_permuted) > 0)),
        'Point-biserial correlation coeff': r_bp,
        'LogLoss': sax_loss + bed_loss + dishes_loss,
    }, index=[0])


def main(df0=10, scale0=0.3, l2_regularization=0.0, dropout=0.5, t=3, output_path=None,
         optimizer=None, output_id_tag=None, n_epochs=100, lmda=10**5, alfa=10 ** -1,
         bin_size=1.0, n_permute=100, comp_data_path=None, video_data_path=None,
         reset_weights=True):

    output_tag = '_df0_{}_scale0_{}_l2_{}_do_{}'.format(df0, scale0, l2_regularization, dropout)
    if output_id_tag is not None:
        output_tag += output_id_tag


    ####### DP-GMM Events #########
    f_class = Gaussian
    f_opts = dict(var_df0=df0, var_scale0=scale0)
    # # we want no stickiness as a control!
    kwargs = dict(lmda=0.0, alfa=alfa, bin_size=bin_size, n_permute=n_permute,
                  human_data_path=comp_data_path, video_data_path=video_data_path)
    # sys.stdout.write('Running DP-GMM\n')
    res = seg_comp_new(f_class, f_opts, **kwargs)
    res['EventModel'] = ['DP-GMM'] * len(res)
    res.to_pickle(output_path + 'EventR2_DPGMM' + output_tag + '.pkl')
    sys.stdout.write('\n')

    # ####### Gaussian Random Walk #########
    f_class = GaussianRandomWalk
    f_opts = dict(var_df0=df0, var_scale0=scale0)
    kwargs['lmda'] = lmda  # change this back for other models
    sys.stdout.write('Running Random Walk\n')
    res = seg_comp_new(f_class, f_opts, **kwargs)
    res['EventModel'] = ['RandomWalk'] * len(res)
    res.to_pickle(output_path + 'EventR2_RandWalk' + output_tag + '.pkl')
    sys.stdout.write('\n')

    # ####### LDS Events #########
    f_class = KerasLDS
    f_opts = dict(var_df0=df0, var_scale0=scale0, l2_regularization=l2_regularization,
                  optimizer=optimizer, n_epochs=n_epochs, reset_weights=reset_weights)
    sys.stdout.write('Running LDS\n')
    res = seg_comp_new(f_class, f_opts, **kwargs)
    res['EventModel'] = ['LDS'] * len(res)
    res.to_pickle(output_path + 'EventR2_LDS' + output_tag + '.pkl')
    sys.stdout.write('\n')

    ####### MLP Events #########
    f_class = KerasMultiLayerPerceptron
    f_opts = dict(var_df0=df0, var_scale0=scale0, l2_regularization=l2_regularization,
                  dropout=dropout, optimizer=optimizer,  n_epochs=n_epochs, reset_weights=reset_weights)

    sys.stdout.write('Running MLP\n')
    res = seg_comp_new(f_class, f_opts, **kwargs)
    res['EventModel'] = ['MLP'] * len(res)
    res.to_pickle(output_path + 'EventR2_MLP' + output_tag + '.pkl')
    sys.stdout.write('\n')

    # ####### SRN Events #########
    f_class = KerasRecurrentMLP
    f_opts = dict(var_df0=df0, var_scale0=scale0, l2_regularization=l2_regularization,
                  dropout=dropout, t=t, optimizer=optimizer,  n_epochs=n_epochs, reset_weights=reset_weights)

    sys.stdout.write('Running RNN\n')
    res = seg_comp_new(f_class, f_opts, **kwargs)
    res['EventModel'] = ['RNN'] * len(res)
    res.to_pickle(output_path + 'EventR2_RNN' + output_tag + '.pkl')
    sys.stdout.write('\n')

    # ####### GRU #########
    f_class = KerasGRU
    f_opts = dict(var_df0=df0, var_scale0=scale0, l2_regularization=l2_regularization,
                  dropout=dropout, t=t, optimizer=optimizer, n_epochs=n_epochs, reset_weights=reset_weights)

    sys.stdout.write('Running GRU\n')
    res = seg_comp_new(f_class, f_opts, **kwargs)
    res['EventModel'] = ['GRU'] * len(res)
    res.to_pickle(output_path + 'EventR2_GRU' + output_tag + '.pkl')
    sys.stdout.write('\n')

    # ####### LSTM #########
    f_class = KerasLSTM
    f_opts = dict(var_df0=df0, var_scale0=scale0, l2_regularization=l2_regularization,
                  dropout=dropout, t=t, optimizer=optimizer,  n_epochs=n_epochs, reset_weights=reset_weights)

    sys.stdout.write('Running LSTM\n')
    res = seg_comp_new(f_class, f_opts, **kwargs)
    res['EventModel'] = ['LSTM'] * len(res)
    res.to_pickle(output_path + 'EventR2_LSTM' + output_tag + '.pkl')
    sys.stdout.write('\n')

    sys.stdout.write('Done!\n')


def gru_only(df0=10, scale0=0.3, l2_regularization=0.0, dropout=0.5, t=3, output_path=None,
         optimizer=None, output_id_tag=None, n_epochs=100, lmda=10**5, alfa=10 ** -1,
         bin_size=1.0, n_permute=100, comp_data_path=None, video_data_path=None,
         reset_weights=True):

    output_tag = '_df0_{}_scale0_{}_l2_{}_do_{}'.format(df0, scale0, l2_regularization, dropout)
    if output_id_tag is not None:
        output_tag += output_id_tag


    kwargs = dict(lmda=lmda, alfa=alfa, bin_size=bin_size, n_permute=n_permute,
                  human_data_path=comp_data_path, video_data_path=video_data_path)

    # ####### GRU #########
    f_class = KerasGRU
    f_opts = dict(var_df0=df0, var_scale0=scale0, l2_regularization=l2_regularization,
                  dropout=dropout, t=t, optimizer=optimizer, n_epochs=n_epochs, reset_weights=reset_weights)

    sys.stdout.write('Running GRU\n')
    res = seg_comp_new(f_class, f_opts, **kwargs)
    res['EventModel'] = ['GRU'] * len(res)
    res.to_pickle(output_path + 'EventR2_GRU' + output_tag + '.pkl')
    sys.stdout.write('\n')


if __name__ == "__main__":

    output_path = './'
    video_data_path = './data/videodata/'
    comp_data_file = './data/Zacks2006/'
    # video_data_path = './'
    # comp_data_file = './'

    lmda = 10 ** 5
    alfa = 10 ** -1

    gru_only(df0=1, scale0=0.3, l2_regularization=0.0, dropout=0.5, t=3, n_epochs=50,
         lmda=lmda, alfa=alfa,
         optimizer=None, output_id_tag="adam", reset_weights=True,
         output_path=output_path, comp_data_path=comp_data_file, video_data_path=video_data_path)

    # time_test(df0=10, scale0=0.9, l2_regularization=0.0, dropout=0.5, t=10, data_path=data_path)
