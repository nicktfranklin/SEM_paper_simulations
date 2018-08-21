import numpy as np


def unroll_data(X, t=1):
    """
    This function is used by recurrent neural nets to do back-prop through time.

    Unrolls a data_set for with time-steps, truncated for t time-steps
    appends t-1 D-dimensional zero vectors at the beginning.

    Parameters:
        X: array, shape (N, D) or shape (D,)

        t: int
            time-steps to truncate the unroll

    output
    ------

        X_unrolled: array, shape (N-1, t, D)

    """
    if np.ndim(X) == 2:
        N, D = np.shape(X)
    elif np.ndim(X):
        N, D = 1, np.shape(X)[0]
        X = np.reshape(X, (1, D))

    X_unrolled = np.zeros((N, t, D))

    # append a t-1 blank (zero) input patterns to the beginning
    data_set = np.concatenate([np.zeros((t - 1, D)), X])

    for ii in range(N):
        X_unrolled[ii, :, :] = data_set[ii: ii + t, :]

    return X_unrolled
