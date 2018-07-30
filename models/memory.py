import numpy as np
from tqdm import tqdm_notebook as tqdm
from scipy.stats import multivariate_normal as mvnorm
from scipy.special import logsumexp


def reconstruction_single_event(X_mem, event_model, tau=1.0, burn_in=500, n_samples=500):
    """
    Reconstruct a memory trace from a single big event

    :param X_mem:
    :param event_model:
    :param tau:
    :param burn_in:
    :param n_samples:
    :return:
    """

    n, d = np.shape(X_mem)

    # these are shuffled in order
    trials = range(0, n)
    X_sample = []

    # randomly initialize the sampel of X0 from the corruption process
    X0 = np.random.randn(n, d) * tau + X_mem

    for ii in tqdm(range(n_samples + burn_in), leave=False, desc='Gibbs'):

        np.random.shuffle(trials)  # randomize the order trials are sampled

        for t in trials:
            # pull the transition function
            if t > 1:
                def f(x):
                    return event_model.predict_next_generative(x)
            else:
                def f(x):
                    return event_model.predict_f0()

            # construct the input vector for the model
            X_i = X0[:t, :]

            # calculate the weighted mean and weighted variance of the posterior
            # using the event model's estimate and the assumed corruption noise
            # NOTE: b/c the covariance function is restricted to the form
            # Sigma = beta * I where beta is a vector, each feature can be treated
            # as an independent gaussian and combined independently
            beta = np.diagonal(event_model.Sigma)
            u = (1. / beta) / (1. / beta + 1. / tau)
            Sigma = np.eye(d) * (1.0 / (1. / beta + 1. / tau))

            # use the likelihood function to estimate a new sample.
            # This likelihood function is the product of two Gaussians:
            #  N(x0_t; f(x0_{1:t-1}), beta_f * I) * N(x0_t; \tilde x_t , beta_mem * I )
            mu_t = u * f(X_i) + (1 - u) * X_mem[t]

            # generate a sample from the multivariate normal
            X0[t, :] = np.random.multivariate_normal(mu_t.flatten(), Sigma)

        if ii >= burn_in:
            X_sample.append(X0.copy())

    return np.array(X_sample)


def reconstruction_known_boundaries(X_mem, events_dict, event_tokens, tau=1.0, burn_in=500, n_samples=500):
    """
    Reconstruct a memory trace from an unordered set of dynamics and event boundaries

    :param X_mem: NxD np.ndarraycorrupted memory trace
    :param events_dict: dictionary of the form {k: event_model)
    :param event_tokens: a list of event tokens -- unique id for each event
    :param tau: float corruption noise
    :param burn_in: (int) number of samples to discard from Gibbs
    :param n_samples: number of samples to keep from Gibbs
    :return:
    """

    n, d = np.shape(X_mem)

    # these are shuffled in order, and we don't want to make a prediction for the last scene
    trials = range(0, n)
    X_sample = []

    # randomly initialize the sampel of X0 from the corruption process
    X0 = np.random.randn(n, d) * tau + X_mem

    # randomly initialize the identity of the event tokens
    events_keys_function = np.random.permutation(events_dict.keys())

    for ii in tqdm(range(n_samples + burn_in), leave=False, desc='Gibbs'):

        np.random.shuffle(trials)  # randomize the order trials are sampled

        for t in trials:

            # pull the transition function associated with the current event-token
            token = event_tokens[t]
            token_id = events_keys_function[token]
            event_model = events_dict[token_id]
            if token == event_tokens[t - 1]:
                def f(x):
                    return event_model.predict_next_generative(x)
            else:
                def f(x):
                    return event_model.predict_f0()

            # construct the input vector for the model
            X_i = X0[(np.arange(n) < t) & (np.array(event_tokens) == event_tokens[t]), :]

            # calculate the weighted mean and weighted variance of the posterior
            # using the event model's estimate and the assumed corruption noise
            # NOTE: b/c the covarinace function is restricted to the form
            # Sigma = beta * I where beta is a vector, each feature can be treated
            # as an indepenendent gaussian and combined independently
            beta = np.diagonal(event_model.Sigma)
            u = (1. / beta) / (1. / beta + 1. / tau)
            Sigma = np.eye(d) * (1.0 / (1. / beta + 1. / tau))

            # use the likelihood function to estimate a new sample.
            # This likelihood function is the product of two gaussians:
            #  N(x0_t; f(x0_{1:t-1}), beta_f * I) * N(x0_t; \tilde x_t , beta_mem * I )
            mu_t = u * f(X_i) + (1 - u) * X_mem[t]

            # generate a sample from the multivariate normal
            X0[t, :] = np.random.multivariate_normal(mu_t.flatten(), Sigma)

        # Here, we sample the events type probability, conditioned on the samples

        # we will sample the event models WITHOUT replacement, so keep track
        # of the available models (this also speeds things up)
        available_models = events_dict.keys()

        events_keys_function = np.zeros(len(available_models)) - 1  # reinitialize the key

        for token0 in np.random.permutation(list(set(event_tokens))):
            # get the likelihood for each token in the set
            X00 = X0[event_tokens == token0, :]  # pull the relevant scenes

            # initialize the scores. This includes models that are not avialble,
            # so they have to be initialized at log p(e) = -infty (or np.log(0))
            scores = np.log(np.zeros(np.max(available_models) + 1))

            # loop through the as yet unassigned models and "score" (log likelihood)
            for token_id0 in available_models:
                # pull the corresponding event model
                e00 = events_dict[token_id0]

                # run the model to generate predictions
                X00_hat = np.zeros(np.shape(X00))
                X00_hat[0, :] = e00.predict_f0()
                for jj in range(1, np.shape(X00)[0]):
                    X00_hat[jj, :] = e00.predict_next_generative(X00[:jj, :])

                # evaluate the predictions of the model
                scores[token_id0] = logsumexp(
                    mvnorm.logpdf(X00 - X00_hat, mean=np.zeros(d), cov=e00.Sigma)
                )

            # normalize the loglikelihoods and inverse cdf sample
            pmf_e = np.exp(scores - logsumexp(scores))
            cmf_e = np.cumsum(pmf_e)
            e0 = events_dict.keys()[np.sum(cmf_e < np.random.rand())]

            events_keys_function[token0] = e0
            try:
                available_models.remove(e0)
            except:
                print available_models
                print e0
                print pmf_e
                print scores
                raise(Exception)

        if ii >= burn_in:
            X_sample.append(X0.copy())

    return np.array(X_sample)