import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter


def _reparameterize(mu, logvar):
    """Reparameterization trick.
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    z = mu + eps * std
    return z


class GatedConvBN1d(torch.nn.Module):
    """1-D Gated convolution layer with batch normalization.
    Arguments are the same as `torch.nn.Conv1d`.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        super(GatedConvBN1d, self).__init__()

        self.conv = torch.nn.Conv1d(
            in_channels, 2 * out_channels, kernel_size, stride, padding,
            dilation, groups, bias, padding_mode)
        self.bn = torch.nn.BatchNorm1d(2 * out_channels)

    def forward(self, x):
        return F.glu(self.bn(self.conv(x)), dim=1)


class GatedDeconvBN1d(torch.nn.Module):
    """1-D Gated deconvolution layer with batch normalization.
    Arguments are the same as `torch.nn.ConvTranspose1d`.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True,
                 dilation=1, padding_mode='zeros'):
        super(GatedDeconvBN1d, self).__init__()

        self.deconv = torch.nn.ConvTranspose1d(
            in_channels, 2 * out_channels, kernel_size, stride, padding,
            output_padding, groups, bias, dilation, padding_mode)
        self.bn = torch.nn.BatchNorm1d(2 * out_channels)

    def forward(self, x):
        return F.glu(self.bn(self.deconv(x)), dim=1)


class CVAE(torch.nn.Module):
    """Conditional VAE (M2 model) for MVAE.

    Args:
        n_speakers (int): Number of speakers.
    """
    def __init__(self, n_speakers):
        super(CVAE, self).__init__()

        self._n_speakers = n_speakers

        self.log_g = Parameter(torch.ones([]))

        self.encoder_conv1 = GatedConvBN1d(
            2049 + n_speakers, 1024, kernel_size=5, stride=1, padding=2)
        self.encoder_conv2 = GatedConvBN1d(
            1024 + n_speakers, 512, kernel_size=4, stride=2, padding=1)
        self.encoder_mu = torch.nn.Conv1d(
            512 + n_speakers, 256, kernel_size=4, stride=2, padding=1)
        self.encoder_logvar = torch.nn.Conv1d(
            512 + n_speakers, 256, kernel_size=4, stride=2, padding=1)

        self.decoder_deconv1 = GatedDeconvBN1d(
            256 + n_speakers, 512, kernel_size=4, stride=2, padding=1)
        self.decoder_deconv2 = GatedDeconvBN1d(
            512 + n_speakers, 1024, kernel_size=4, stride=2, padding=1)
        self.decoder_deconv3 = torch.nn.ConvTranspose1d(
            1024 + n_speakers, 2049, kernel_size=5, stride=1, padding=2)

    @property
    def n_speakers(self):
        return self._n_speakers

    def encode(self, x, c):
        c = c.unsqueeze(2)
        h = torch.cat((x, c.expand(-1, -1, x.size(2))), dim=1)
        h = self.encoder_conv1(h)
        h = torch.cat((h, c.expand(-1, -1, h.size(2))), dim=1)
        h = self.encoder_conv2(h)
        h = torch.cat((h, c.expand(-1, -1, h.size(2))), dim=1)
        mu = self.encoder_mu(h)
        logvar = self.encoder_logvar(h)
        return mu, logvar

    def decode(self, z, c):
        c = c.unsqueeze(2)
        h = torch.cat((z, c.expand(-1, -1, z.size(2))), dim=1)
        h = self.decoder_deconv1(h)
        h = torch.cat((h, c.expand(-1, -1, h.size(2))), dim=1)
        h = self.decoder_deconv2(h)
        h = torch.cat((h, c.expand(-1, -1, h.size(2))), dim=1)
        log_sigma_sq = self.decoder_deconv3(h)
        return log_sigma_sq

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = _reparameterize(mu, logvar)
        log_sigma_sq = self.decode(z, c) + self.log_g
        return log_sigma_sq, mu, logvar


def lossfun(x, log_sigma_sq, mu, logvar):
    """Compute the loss function.
    """
    loss = torch.mean(log_sigma_sq) + torch.mean(x / log_sigma_sq.exp()) \
        - 0.5 * torch.mean(logvar - mu.pow(2) - logvar.exp())
    return loss
