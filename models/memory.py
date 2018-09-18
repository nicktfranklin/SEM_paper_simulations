import numpy as np
from tqdm import tqdm
from scipy.stats import multivariate_normal as mvnorm
from scipy.special import logsumexp


def sample_pmf(pmf):
    return np.sum(np.cumsum(pmf) < np.random.uniform(0, 1))


def sample_y_given_x(y_mem, x, b, tau, epsilon):
    # total number of samples
    n, d = np.shape(x)

    #
    y_sample = [None] * n

    # create a copy of y_mem for sampling without replacement
    y_mem_copy = [[x_yi.copy(), t_mem] for (x_yi, t_mem) in y_mem]

    for t in np.random.permutation(range(n)):

        # create a probability function over the sample sets
        log_p = np.zeros(len(y_mem_copy) + 1) - np.inf
        for ii, (x_yi, t_mem) in enumerate(y_mem_copy):
            if np.abs(t_mem - t) <= b:
                log_p[ii] = mvnorm.logpdf(x_yi, mean=x[t, :], cov=np.eye(d) * tau)
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


def sample_e_given_x(x, event_models, alpha, lmda):
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

        # calculate the CRP prior
        p_sCRP = c.copy()
        if e_prev is not None:
            p_sCRP[e_prev] += lmda
        p_sCRP += (p_sCRP == 0) * alpha
        # no need to normalize yet

        # calculate the probability of x_t|x_{1:t-1}
        p_model = np.zeros(k) - np.inf
        for idx, e_model in event_models.iteritems():
            if idx != e_prev:
                x_t_hat = e_model.predict_next_generative(x_current)
            else:
                x_t_hat = e_model.predict_f0()
            p_model[idx] = mvnorm.logpdf(x[t, :], mean=x_t_hat.reshape(-1), cov=e_model.Sigma)

        log_p = p_model + p_sCRP
        log_p -= logsumexp(log_p)

        # draw from the model
        e_sample[t] = sample_pmf(np.exp(log_p))

        # update counters
        if e_prev == e_sample[t]:
            x_current = np.concatenate([x_current, x[t, :].reshape(1, -1)])
        else:
            x_current = x[t, :].reshape(1, -1)
        e_prev = e_sample[t]

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
                         n_samples=500, n_burnin=500, leave_progress_bar=True):

    # initialize the x_hat with a noisy copy of the memory trace
    x_hat = np.array([y0[0].copy() for y0 in y_mem])
    idx = np.arange(0, np.shape(x_hat)[0]) + np.random.randint(-b, b + 1, np.shape(x_hat)[0])
    idx[idx < 0] = 0
    idx[idx >= len(idx)] = len(idx) - 1
    x_hat = x_hat[idx, :]
    x_hat += 10 * tau * np.random.randn(len(y_mem), sem.d)

    #
    e_samples = [None] * n_samples
    y_samples = [None] * n_samples
    x_samples = [None] * n_samples

    for ii in tqdm(range(n_burnin + n_samples), desc='Gibbs Sampler', leave=leave_progress_bar):

        # sample the event models
        e_hat = sample_e_given_x(x_hat, sem.event_models, memory_alpha, memory_lambda)

        # sample the memory traces
        y_sample = sample_y_given_x(y_mem, x_hat, b, tau, memory_epsilon)

        # sample the memory features
        x_hat = sample_x_given_y_e(x_hat, y_sample, e_hat, sem.event_models, tau)

        if ii >= n_burnin:
            e_samples[ii - n_burnin] = e_hat
            y_samples[ii - n_burnin] = y_sample
            x_samples[ii - n_burnin] = x_hat

    return y_samples, e_samples, x_samples


# this is a debugging function
def gibbs_memory_sampler_reduced(y_mem, sem, e_true, memory_epsilon, b, tau,
                         n_samples=500, n_burnin=500, leave_progress_bar=True):

    # initialize the x_hat with a noisy copy of the memory trace
    x_hat = np.array([y0[0].copy() for y0 in y_mem])
    idx = np.arange(0, np.shape(x_hat)[0]) + np.random.randint(-b, b + 1, np.shape(x_hat)[0])
    idx[idx < 0] = 0
    idx[idx >= len(idx)] = len(idx) - 1
    x_hat = x_hat[idx, :]
    x_hat += 10 * tau * np.random.randn(len(y_mem), sem.d)

    #
    y_samples = [None] * n_samples
    x_samples = [None] * n_samples

    for ii in tqdm(range(n_burnin + n_samples), desc='Gibbs Sampler', leave=leave_progress_bar):

        # sample the memory traces
        y_sample = sample_y_given_x(y_mem, x_hat, b, tau, memory_epsilon)

        # sample the memory features
        x_hat = sample_x_given_y_e(x_hat, y_sample, e_true, sem.event_models, tau)

        if ii >= n_burnin:
            y_samples[ii - n_burnin] = y_sample
            x_samples[ii - n_burnin] = x_hat

    return y_samples, x_samples