try:
    import cupy as cp
except ImportError:
    cp = None
import numpy as np
import torch

from common import projection_back, to_cupy, to_tensor
from ilrma import ilrma

EPS = 1e-9


def mvae(mix, model, n_iter, device, proj_back=True, return_sigma=False):
    """Implementation of Multichannel Conditional VAE.
    It only works in the determined case (n_sources == n_channels).

    Args:
        mix (ndarray): (n_frequencies, n_channels, n_frames)
            STFT representation of the observed signal.
        model (cvae.CVAE): Trained Conditional VAE model.
        n_iter (int): Number of iterations.
        device (torch.device): Device used for computation.
        proj_back (bool): If use back-projection technique.
        return_sigma (bool): If also return estimated power spectrogram for
            each speaker.

    Returns:
        tuple[ndarray, ndarray]: Tuple of separated signal and
            separation matrix. The shapes of separated signal and separation
            matrix are (n_frequencies, n_sources, n_frames) and
            (n_frequencies, n_sources, n_channels), respectively.
    """
    if isinstance(mix, np.ndarray):
        if_use_cuda = False
        xp = np
    elif isinstance(mix, cp.ndarray):
        if_use_cuda = True
        xp = cp
    else:
        raise ValueError('A numpy.ndarray or cupy.ndarray instance should be '
                         'given as `mix` argument')

    n_freq, n_src, n_frame = mix.shape

    sep, sep_mat = ilrma(mix, n_iter=30, n_basis=2)
    sep_pow = xp.power(xp.abs(sep), 2)  # (n_freq, n_src, n_frame)
    c = torch.full((n_src, model.n_speakers), 1 / model.n_speakers,
                   device=device, requires_grad=True)
    log_g = torch.full((n_src, 1, 1), model.log_g.item(), device=device)

    with torch.no_grad():
        if if_use_cuda:
            sep_pow_tensor = to_tensor(sep_pow).transpose(0, 1)
        else:
            sep_pow_tensor =\
                torch.from_numpy(sep_pow).transpose(0, 1).to(device)
        sep_pow_tensor.clamp_(EPS)
        z, _ = model.encode(sep_pow_tensor, c)
        sigma_sq = (model.decode(z, c) + log_g).exp()
        sigma_sq.clamp_(min=EPS)
        sigma_reci = 1 / sigma_sq
        if if_use_cuda:
            sigma_reci = to_cupy(sigma_reci)
        else:
            sigma_reci = sigma_reci.numpy()

    z.requires_grad = True

    eye = xp.tile(xp.eye(n_src), (n_freq, 1, 1))

    for _ in range(n_iter):
        for src in range(n_src):
            h = sigma_reci[src, :, :, None] @ xp.ones((1, n_src))
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

        optimizer = torch.optim.Adam((z, c), lr=1e-3)
        if if_use_cuda:
            sep_pow_tensor = to_tensor(sep_pow).transpose(0, 1)
        else:
            sep_pow_tensor = \
                torch.from_numpy(sep_pow).transpose(0, 1).to(device)
        for _ in range(50):
            log_sigma_sq = model.decode(z, torch.softmax(c, dim=1)) + log_g
            loss = torch.sum(
                log_sigma_sq + (sep_pow_tensor.log() - log_sigma_sq).exp())
            model.zero_grad()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            sigma_sq = (model.decode(z, torch.softmax(c, dim=1)) + log_g).exp()
            lbd = torch.sum(sep_pow_tensor / sigma_sq, dim=(1, 2))
            lbd = lbd / n_freq / n_frame / log_g.squeeze(2).squeeze(1).exp()
            log_g[:, 0, 0] += torch.log(lbd)
            sigma_sq *= lbd.unsqueeze(1).unsqueeze(2)
            if if_use_cuda:
                sep_mat *= to_cupy(lbd.unsqueeze(0).unsqueeze(2))
                sigma_reci = to_cupy(1 / sigma_sq)
            else:
                sep_mat *= lbd.unsqueeze(0).unsqueeze(2).numpy()
                sigma_reci = (1 / sigma_sq).numpy()

    # Back-projection technique
    if proj_back:
        z = projection_back(sep, mix[:, 0, :])
        sep *= xp.conj(z[:, :, None])

    if return_sigma:
        return sep, sep_mat, sigma_sq.cpu().numpy()
    else:
        return sep, sep_mat
