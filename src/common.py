import numpy as np

EPS = 1e-9


def projection_back(sep, ref):
    """Back-projection technique.

    Args:
        sep (numpy.ndarray): (n_frequencies, n_channels, n_frames)
            The STFT data to project back on the reference signal.
        ref (numpy.ndarray): (n_frequencies, n_frames)
            The reference signal.

    Returns:
        numpy.ndarray: (n_frequencies, n_channels)
            The frequency-domain filter which minimizes the square error to
            the reference signal.
    """
    num = np.sum(np.conj(ref[:, None, :]) * sep, axis=2)
    denom = np.sum(np.abs(sep) ** 2, axis=2)
    np.clip(denom, a_min=EPS, a_max=None, out=denom)
    return num / denom