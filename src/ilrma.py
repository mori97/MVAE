import numpy as np

EPS = 1e-9


def ilrma(mix, n_iter, n_basis=2):
    """Implementation of ILRMA (Independent Low-Rank Matrix Analysis).
    This algorithm is called ILRMA1 in http://d-kitamura.net/pdf/misc/AlgorithmsForIndependentLowRankMatrixAnalysis.pdf
    It only works in determined case (n_sources == n_channels).

    Args:
        mix (numpy.ndarray): (n_frequencies, n_channels, n_frames)
            STFT representation of the observed signal.
        n_iter (int): Number of iterations.
        n_basis (int): Number of basis in the NMF model.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: Tuple of separated signal and
            separation matrix. The shapes of separated signal and separation
            matrix are (n_frequencies, n_sources, n_frames) and
            (n_sources, n_channels), respectively.
    """
    n_freq, n_src, n_frame = mix.shape

    sep_mat = np.stack([np.eye(n_src, dtype=mix.dtype) for _ in range(n_freq)])
    basis = np.abs(np.random.randn(n_src, n_freq, n_basis))
    act = np.abs(np.random.randn(n_src, n_basis, n_frame))
    sep = sep_mat @ mix
    sep_pow = np.power(np.abs(sep), 2)  # (n_freq, n_src, n_frame)
    model = basis @ act  # (n_src, n_freq, n_frame)
    m_reci = 1 / model

    eye = np.tile(np.eye(n_src), (n_freq, 1, 1))

    for _ in range(n_iter):
        for src in range(n_src):
            h = (sep_pow[:, src, :] * m_reci[src]**2) @ act[src].T
            h /= m_reci[src] @ act[src].T
            h = np.sqrt(h, out=h)
            basis[src] *= h
            np.clip(basis[src], a_min=EPS, a_max=None, out=basis[src])

            model[src] = basis[src] @ act[src]
            m_reci[src] = 1 / model[src]

            h = basis[src].T @ (sep_pow[:, src, :] * m_reci[src]**2)
            h /= basis[src].T @ m_reci[src]
            h = np.sqrt(h, out=h)
            act[src] *= h
            np.clip(act[src], a_min=EPS, a_max=None, out=act[src])

            model[src] = basis[src] @ act[src]
            m_reci[src] = 1 / model[src]

            h = m_reci[src, :, :, None] @ np.ones((1, n_src))
            h = mix.conj() @ (mix.swapaxes(1, 2) * h)
            u_mat = h.swapaxes(1, 2) / n_frame
            h = sep_mat @ u_mat + EPS * eye
            sep_mat[:, src, :] = np.linalg.solve(h, eye[:, :, src]).conj()
            h = sep_mat[:, src, None, :] @ u_mat
            h = (h @ sep_mat[:, src, :, None].conj()).squeeze(2)
            sep_mat[:, src, :] = (sep_mat[:, src, :] / np.sqrt(h).conj())

        np.matmul(sep_mat, mix, out=sep)
        np.power(np.abs(sep), 2, out=sep_pow)

        for src in range(n_src):
            lbd = np.sqrt(np.sum(sep_pow[:, src, :]) / n_freq / n_frame)
            sep_mat[:, src, :] /= lbd
            sep_pow[:, src, :] /= lbd ** 2
            model[src] /= lbd ** 2
            basis[src] /= lbd ** 2

    return sep, sep_mat
