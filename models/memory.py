import numpy as np
from tqdm import tqdm
from scipy.stats import multivariate_normal as mvnorm
from scipy.special import logsumexp


def sample_pmf(pmf):
    return np.sum(np.cumsum(pmf) < np.random.uniform(0, 1))


def init_y_sample(y_mem, b, epsilon):
    y_sample = [None] * len(y_mem)

    # create a copy of y_mem for sampling without replacement
    y_mem_copy = [[x_i.copy(), e_i, t_mem] for (x_i, e_i, t_mem) in y_mem]

    for t in np.random.permutation(range(len(y_mem))):

        # create a probability function over the sample sets
        log_p = np.zeros(len(y_mem_copy) + 1) - np.inf
        for ii, (x_i, e_i, t_mem) in enumerate(y_mem_copy):
            if np.abs(t_mem - t) <= b:
                log_p[ii] = 0
                # draw a sample
        log_p[-1] = np.log(epsilon)
        p = np.exp(log_p - logsumexp(log_p))  # normalize and exponentiate

        ii = sample_pmf(p)

        if ii < len(y_mem_copy):
            # only create a sample for none-None events
            y_sample[t] = y_mem_copy[ii]
            y_mem_copy = y_mem_copy[:ii] + y_mem_copy[ii + 1:]  # remove the item from the list of available
    return y_sample


def init_x_sample_cond_y(y_sample, n, d, tau):
    x_sample = np.random.randn(n, d) * tau

    for ii, y_ii in enumerate(y_sample):
        if y_ii is not None:
            x_sample[ii, :] = np.random.randn(1, d) * tau + y_ii[0]
    return x_sample


def sample_y_given_x_e(y_mem, x, e, b, tau, epsilon):
    # total number of samples
    n, d = np.shape(x)

    #
    y_sample = [None] * n

    # create a copy of y_mem for sampling without replacement
    y_mem_copy = [[x_i.copy(), e_i, t_mem] for (x_i, e_i, t_mem) in y_mem]

    for t in np.random.permutation(range(n)):

        # create a probability function over the sample sets
        log_p = np.zeros(len(y_mem_copy) + 1) - np.inf
        for ii, (x_i, e_i, t_mem) in enumerate(y_mem_copy):
            if np.abs(t_mem - t) <= b:
                log_p[ii] = mvnorm.logpdf(x_i, mean=x[t, :], cov=np.eye(d) * tau)

            # set probability to zero if event token doesn't match
            if e_i is not None:
                if e_i != e[ii]:
                    log_p[ii] -= np.inf

        # the last token is always the null token
        log_p[-1] = np.log(epsilon)
        p = np.exp(log_p - logsumexp(log_p))  # normalize and exponentiate

        # draw a sample
        ii = sample_pmf(p)

        if ii < len(y_mem_copy):
            # only create a sample for none-None events
            y_sample[t] = y_mem_copy[ii]
            y_mem_copy = y_mem_copy[:ii] + y_mem_copy[ii + 1:]  # remove the item from the list of available

    return y_sample


def sample_e_given_x_y(x, y, event_models, alpha, lmda):
    n, d = np.shape(x)

    # define a special case of the sCRP that caps the number
    # of clusters at k, the number of event models
    k = len(event_models)
    c = np.zeros(k)

    e_prev = None
    e_sample = [None] * n

    # keep a list of all the previous scenes within the sampled event
    x_current = np.zeros((1, d))

    # do this as a filtering operation, just via a forward sweep
    for t in range(n):

        # first see if there is a valid memory token with a event label
        if (y[t] is not None) and (y[t][1] is not None):
            e_sample[t] = y[t][1]
            e_prev = e_sample[t]
            continue
        else:

            # calculate the CRP prior
            p_sCRP = c.copy()
            if e_prev is not None:
                p_sCRP[e_prev] += lmda

            # add the alpha value to the first unvisited cluster
            # (using argmax to get the first non-zero element)
            if any(p_sCRP == 0):
                idx = np.argmax(p_sCRP == 0)
                p_sCRP[idx] = alpha
            # no need to normalize yet

            # calculate the probability of x_t|x_{1:t-1}
            p_model = np.zeros(k) - np.inf
            for idx, e_model in event_models.iteritems():
                if idx != e_prev:
                    x_t_hat = e_model.predict_next_generative(x_current)
                else:
                    x_t_hat = e_model.predict_f0()
                p_model[idx] = mvnorm.logpdf(x[t, :], mean=x_t_hat.reshape(-1), cov=e_model.Sigma)

            log_p = p_model + np.log(p_sCRP)
            log_p -= logsumexp(log_p)

            # draw from the model
            e_sample[t] = sample_pmf(np.exp(log_p))

            # update counters
            if e_prev == e_sample[t]:
                x_current = np.concatenate([x_current, x[t, :].reshape(1, -1)])
            else:
                x_current = x[t, :].reshape(1, -1)
        e_prev = e_sample[t]

        # update the counts!
        c[e_sample[t]] += 1

    return e_sample


def sample_x_given_y_e(x_hat, y, e, event_models, tau):
    """
    x_hat: n x d np.array
        the previous sample, to be updated and returned

    y: list
        the sequence of ordered memory traces. Each element is
        either a list of [x_y_mem, t_mem] or None

    e: np.array of length n
        the sequence of event tokens

    event_models: dict {token: model}
        trained event models

    tau:
        memory corruption noise

    """

    # total number of samples
    n, d = np.shape(x_hat)

    x_hat = x_hat.copy()  # don't want to overwrite the thing outside the loop...

    for t in np.random.permutation(range(n)):
        # pull the active event model
        e_model = event_models[e[t]]

        # pull all preceding scenes within the event
        x_idx = np.arange(len(e))[(e == e[t]) & (np.arange(len(e)) < t)]
        x_prev = np.concatenate([
            np.zeros((1, d)), x_hat[x_idx, :]
        ])

        # pull the prediction of the event model given the previous estimates of x
        f_x = e_model.predict_next_generative(x_prev)

        # is y_t a null tag?
        if y[t] is None:
            x_bar = f_x
            Sigma = e_model.Sigma
        else:
            # calculate noise lambda for each event model
            u_weight = (1. / np.diag(e_model.Sigma)) / (1. / np.diag(e_model.Sigma) + 1. / tau)

            x_bar = u_weight * f_x + (1 - u_weight) * y[t][0]
            Sigma = np.eye(d) * 1. / (1. / np.diag(e_model.Sigma) + 1. / tau)

        # draw a new sample of x_t
        x_hat[t, :] = mvnorm.rvs(mean=x_bar.reshape(-1), cov=Sigma)

    return x_hat


def gibbs_memory_sampler(y_mem, sem, memory_alpha, memory_lambda, memory_epsilon, b, tau,
                         n_samples=100, n_burnin=250, progress_bar=True, leave_progress_bar=True):

    d = np.shape(y_mem[0][0])[0]
    n = len(y_mem)

    #
    e_samples = [None] * n_samples
    y_samples = [None] * n_samples
    x_samples = [None] * n_samples

    y_sample = init_y_sample(y_mem, b, memory_epsilon)
    x_sample = init_x_sample_cond_y(y_sample, n, d, tau)
    e_sample = sample_e_given_x_y(x_sample, y_sample, sem.event_models, memory_alpha, memory_lambda)

    # loop through the other events in the list
    if progress_bar:
        def my_it(iterator):
            return tqdm(iterator, desc='Gibbs Sampler', leave=leave_progress_bar)
    else:
        def my_it(iterator):
            return iterator

    for ii in my_it(range(n_burnin + n_samples)):

        # sample the memory features
        x_sample = sample_x_given_y_e(x_sample, y_sample, e_sample, sem.event_models, tau)

        # sample the event models
        e_sample = sample_e_given_x_y(x_sample, y_sample, sem.event_models, memory_alpha, memory_lambda)

        # sample the memory traces
        y_sample = sample_y_given_x_e(y_mem, x_sample, e_sample, b, tau, memory_epsilon)

        if ii >= n_burnin:
            e_samples[ii - n_burnin] = e_sample
            y_samples[ii - n_burnin] = y_sample
            x_samples[ii - n_burnin] = x_sample

    return y_samples, e_samples, x_samples


def gibbs_memory_sampler_given_e(y_mem, sem, e_true, memory_epsilon, b, tau,
                         n_samples=100, n_burnin=250, progress_bar=True, leave_progress_bar=True):

    d = np.shape(y_mem[0][0])[0]
    n = len(y_mem)

    #
    y_samples = [None] * n_samples
    x_samples = [None] * n_samples

    y_sample = init_y_sample(y_mem, b, memory_epsilon)
    x_sample = init_x_sample_cond_y(y_sample, n, d, tau)

    # loop through the other events in the list
    if progress_bar:
        def my_it(iterator):
            return tqdm(iterator, desc='Gibbs Sampler', leave=leave_progress_bar)
    else:
        def my_it(iterator):
            return iterator

    for ii in my_it(range(n_burnin + n_samples)):

        # sample the memory features
        x_sample = sample_x_given_y_e(x_sample, y_sample, e_true, sem.event_models, tau)

        # sample the memory traces
        y_sample = sample_y_given_x_e(y_mem, x_sample, e_true, b, tau, memory_epsilon)

        if ii >= n_burnin:
            y_samples[ii - n_burnin] = y_sample
            x_samples[ii - n_burnin] = x_sample

    return y_samples, x_samples

