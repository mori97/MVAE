import numpy as np
try:
    import cupy as cp
except ImportError:
    cp = None
from torch.utils.dlpack import from_dlpack, to_dlpack

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


def to_cupy(tensor):
    """Convert PyTorch tensor to CuPy array.
    """
    return cp.fromDlpack(to_dlpack(tensor))


def to_tensor(cp_array):
    """Convert CuPy array to PyTorch tensor.
    """
    return from_dlpack(cp_array.toDlpack())
