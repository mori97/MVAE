import numpy as np
try:
    import cupy as cp
except ImportError:
    cp = None

from common import projection_back

EPS = 1e-9


def ilrma(mix, n_iter, n_basis=2, proj_back=True):
    """Implementation of ILRMA (Independent Low-Rank Matrix Analysis).
    This algorithm is called ILRMA1 in http://d-kitamura.net/pdf/misc/AlgorithmsForIndependentLowRankMatrixAnalysis.pdf
    It only works in determined case (n_sources == n_channels).

    Args:
        mix (ndarray): (n_frequencies, n_channels, n_frames)
            STFT representation of the observed signal.
        n_iter (int): Number of iterations.
        n_basis (int): Number of basis in the NMF model.
        proj_back (bool): If use back-projection technique.

    Returns:
        tuple[ndarray, ndarray]: Tuple of separated signal and
            separation matrix. The shapes of separated signal and separation
            matrix are (n_frequencies, n_sources, n_frames) and
            (n_sources, n_channels), respectively.
    """
    if isinstance(mix, np.ndarray):
        xp = np
    elif isinstance(mix, cp.ndarray):
        xp = cp
    else:
        raise ValueError('A numpy.ndarray or cupy.ndarray instance should be '
                         'given as `mix` argument')

    n_freq, n_src, n_frame = mix.shape

    sep_mat = xp.stack([xp.eye(n_src, dtype=mix.dtype) for _ in range(n_freq)])
    basis = xp.abs(xp.random.randn(n_src, n_freq, n_basis))
    act = xp.abs(xp.random.randn(n_src, n_basis, n_frame))
    sep = sep_mat @ mix
    sep_pow = xp.power(xp.abs(sep), 2)  # (n_freq, n_src, n_frame)
    model = basis @ act  # (n_src, n_freq, n_frame)
    m_reci = 1 / model

    eye = xp.tile(xp.eye(n_src), (n_freq, 1, 1))

    for _ in range(n_iter):
        for src in range(n_src):
            h = (sep_pow[:, src, :] * m_reci[src]**2) @ act[src].T
            h /= m_reci[src] @ act[src].T
            h = xp.sqrt(h, out=h)
            basis[src] *= h
            xp.clip(basis[src], a_min=EPS, a_max=None, out=basis[src])

            model[src] = basis[src] @ act[src]
            m_reci[src] = 1 / model[src]

            h = basis[src].T @ (sep_pow[:, src, :] * m_reci[src]**2)
            h /= basis[src].T @ m_reci[src]
            h = xp.sqrt(h, out=h)
            act[src] *= h
            xp.clip(act[src], a_min=EPS, a_max=None, out=act[src])

            model[src] = basis[src] @ act[src]
            m_reci[src] = 1 / model[src]

            h = m_reci[src, :, :, None] @ xp.ones((1, n_src))
            h = mix.conj() @ (mix.swapaxes(1, 2) * h)
            u_mat = h.swapaxes(1, 2) / n_frame
            h = sep_mat @ u_mat + EPS * eye
            sep_mat[:, src, :] = xp.linalg.solve(h, eye[:, :, src]).conj()
            h = sep_mat[:, src, None, :] @ u_mat
            h = (h @ sep_mat[:, src, :, None].conj()).squeeze(2)
            sep_mat[:, src, :] = (sep_mat[:, src, :] / xp.sqrt(h).conj())

        sep = sep_mat @ mix
        xp.power(xp.abs(sep), 2, out=sep_pow)
        xp.clip(sep_pow, a_min=EPS, a_max=None, out=sep_pow)

        for src in range(n_src):
            lbd = xp.sqrt(xp.sum(sep_pow[:, src, :]) / n_freq / n_frame)
            sep_mat[:, src, :] /= lbd
            sep_pow[:, src, :] /= lbd ** 2
            model[src] /= lbd ** 2
            basis[src] /= lbd ** 2

    # Back-projection technique
    if proj_back:
        z = projection_back(sep, mix[:, 0, :])
        sep *= xp.conj(z[:, :, None])

    return sep, sep_mat
