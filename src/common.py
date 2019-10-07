import numpy as np
try:
    import cupy as cp
except ImportError:
    cp = None

EPS = 1e-9


def projection_back(sep, ref):
    """Back-projection technique.

    Args:
        sep (ndarray): (n_frequencies, n_channels, n_frames)
            The STFT data to project back on the reference signal.
        ref (ndarray): (n_frequencies, n_frames)
            The reference signal.

    Returns:
        ndarray: (n_frequencies, n_channels)
            The frequency-domain filter which minimizes the square error to
            the reference signal.
    """
    if isinstance(sep, np.ndarray):
        xp = np
    elif isinstance(sep, cp.ndarray):
        xp = cp
    else:
        raise ValueError(
            'A numpy.ndarray or cupy.ndarray instance should be given')

    num = xp.sum(xp.conj(ref[:, None, :]) * sep, axis=2)
    denom = xp.sum(xp.abs(sep) ** 2, axis=2)
    xp.clip(denom, a_min=EPS, a_max=None, out=denom)
    return num / denom
