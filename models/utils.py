import numpy as np


def unroll_data(x, t=1):
    """
    This function is used by recurrent neural nets to do back-prop through time.

    Unrolls a data_set for with time-steps, truncated for t time-steps
    appends t-1 D-dimensional zero vectors at the beginning.

    Parameters:
        x: array, shape (N, D) or shape (D,)

        t: int
            time-steps to truncate the unroll

    output
    ------

        X_unrolled: array, shape (N-1, t, D)

    """
    if np.ndim(x) == 2:
        n, d = np.shape(x)
    elif np.ndim(x):
        n, d = 1, np.shape(x)[0]
        x = np.reshape(x, (1, d))

    x_unrolled = np.zeros((n, t, d))

    # append a t-1 blank (zero) input patterns to the beginning
    data_set = np.concatenate([np.zeros((t - 1, d)), x])

    for ii in range(n):
        x_unrolled[ii, :, :] = data_set[ii: ii + t, :]

    return x_unrolled
