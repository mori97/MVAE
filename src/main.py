import argparse
import os
import re
import statistics as stat

try:
    import cupy as cp
except ImportError:
    cp = None
import librosa
import matplotlib.pyplot as plt
import mir_eval
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from torchaudio.functional import istft

from common import to_tensor
from cvae import CVAE, lossfun
from ilrma import ilrma
from make_dataset import N_FFT, HOP_LEN
from mvae import mvae


def train(model, data_loader, optimizer, device, epoch, writer):
    model.train()

    total_loss = 0
    for x, c in data_loader:
        x, c = x.to(device), c.to(device)
        log_sigma_sq, mu, logvar = model(x, c)
        loss = lossfun(x, log_sigma_sq, mu, logvar)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), 2)
        optimizer.step()

    writer.add_scalar('Loss/train', total_loss / len(data_loader), epoch)


def validate(model, val_dataset, baseline, device, epoch, writer):
    model.eval()

    if_use_cuda = device != torch.device('cpu')
    xp = cp if if_use_cuda else np

    window = torch.hann_window(N_FFT).to(device)

    result = {'SDR': {}, 'SIR': {}, 'SAR': {}}
    for i, (src, mix_spec, speaker) in enumerate(val_dataset):
        if if_use_cuda:
            mix_spec = cp.asarray(mix_spec)
        separated, _ = mvae(mix_spec, model, n_iter=40, device=device)
        separated = separated.transpose(1, 0, 2)
        # Convert to PyTorch-style complex tensor (Shape = (..., 2))
        separated = xp.stack((xp.real(separated), xp.imag(separated)), axis=-1)
        if if_use_cuda:
            separated = to_tensor(separated)
        else:
            separated = torch.from_numpy(separated)
        with torch.no_grad():
            separated = istft(separated, N_FFT, HOP_LEN, window=window)
            separated = separated.cpu().numpy()

        sdr, sir, sar, _ =\
            mir_eval.separation.bss_eval_sources(src, separated)

        if speaker in result['SDR']:
            result['SDR'][speaker].extend(sdr.tolist())
            result['SIR'][speaker].extend(sir.tolist())
            result['SAR'][speaker].extend(sar.tolist())
        else:
            result['SDR'][speaker] = []
            result['SIR'][speaker] = []
            result['SAR'][speaker] = []

        sep_tensor0 = torch.from_numpy(separated[0, :]).unsqueeze(0)
        sep_tensor1 = torch.from_numpy(separated[1, :]).unsqueeze(0)
        writer.add_audio('eval/{}_0'.format(i), sep_tensor0, epoch, 16000)
        writer.add_audio('eval/{}_1'.format(i), sep_tensor1, epoch, 16000)

    for metric in result:
        for speaker in result[metric]:
            result[metric][speaker] = (stat.mean(result[metric][speaker]),
                                       stat.stdev(result[metric][speaker]))

    figures = bar_chart(baseline, result)
    for metric, figure in figures.items():
        writer.add_figure(f'eval/{metric}', figure, epoch)


def bar_chart(baseline, result):
    ret = {}

    speakers = list(result['SDR'].keys())
    speakers.sort()
    x = np.arange(len(speakers))
    width = 0.4

    for metric in result:
        baseline_mean = [baseline[metric][speaker][0] for speaker in speakers]
        baseline_stdv = [baseline[metric][speaker][1] for speaker in speakers]
        result_mean = [result[metric][speaker][0] for speaker in speakers]
        result_stdv = [result[metric][speaker][1] for speaker in speakers]

        figure, ax = plt.subplots()
        ax.bar(x - width / 2, baseline_mean, width,
               yerr=baseline_stdv, label='Baseline')
        ax.bar(x + width / 2, result_mean, width,
               yerr=result_stdv, label='MVAE')

        ax.set_title(metric)
        ax.set_xticks(x)
        ax.set_xticklabels(speakers)
        ax.legend()

        ret[metric] = figure

    return ret


def baseline_ilrma(val_dataset, device):
    """Evaluate with ILRMA.
    """
    if_use_cuda = device != torch.device('cpu')
    xp = cp if if_use_cuda else np

    window = torch.hann_window(N_FFT).to(device)

    ret = {'SDR': {}, 'SIR': {}, 'SAR': {}}
    for src, mix_spec, speaker in val_dataset:
        if if_use_cuda:
            mix_spec = cp.asarray(mix_spec)
        separated, _ = ilrma(mix_spec, n_iter=100)
        separated = separated.transpose(1, 0, 2)
        # Convert to PyTorch-style complex tensor (Shape = (..., 2))
        separated = xp.stack((xp.real(separated), xp.imag(separated)), axis=-1)
        if if_use_cuda:
            separated = to_tensor(separated)
        else:
            separated = torch.from_numpy(separated)
        with torch.no_grad():
            separated = istft(separated, N_FFT, HOP_LEN, window=window)
            separated = separated.cpu().numpy()

        sdr, sir, sar, _ =\
            mir_eval.separation.bss_eval_sources(src, separated)

        if speaker in ret['SDR']:
            ret['SDR'][speaker].extend(sdr.tolist())
            ret['SIR'][speaker].extend(sir.tolist())
            ret['SAR'][speaker].extend(sar.tolist())
        else:
            ret['SDR'][speaker] = []
            ret['SIR'][speaker] = []
            ret['SAR'][speaker] = []

    for metric in ret:
        for speaker in ret[metric]:
            ret[metric][speaker] = (stat.mean(ret[metric][speaker]),
                                    stat.stdev(ret[metric][speaker]))

    return ret


def make_eval_set(path):
    """Make the evaluation dataset.
    """
    src_wav_files = [x for x in os.listdir(path) if x.endswith('_src.wav')]
    ptn = r'(?P<speaker0>[A-Z]{2}\d)(?P<num0>\d\d)_' \
          r'(?P<speaker1>[A-Z]{2}\d)(?P<num1>\d\d)_src\.wav$'
    prog = re.compile(ptn)

    def zero_pad(x):
        if (x.shape[1] + HOP_LEN) % (4 * HOP_LEN) == 0:
            return x
        rest = 4 * HOP_LEN - (x.shape[1] + HOP_LEN) % (4 * HOP_LEN)
        left = rest // 2
        right = rest - left
        return np.pad(x, ((0, 0), (left, right)), mode='constant')

    dataset = []
    for src_wav_file in src_wav_files:
        result = prog.match(src_wav_file)
        speaker0, file_num0 = result.group('speaker0'), result.group('num0')
        speaker1, file_num1 = result.group('speaker1'), result.group('num1')

        mix_wav_file = f'{speaker0}{file_num0}_{speaker1}{file_num1}_mix.wav'
        src_wav_path = os.path.join(path, src_wav_file)
        mix_wav_path = os.path.join(path, mix_wav_file)

        src, _ = librosa.load(src_wav_path, sr=16000, mono=False)
        mix, _ = librosa.load(mix_wav_path, sr=16000, mono=False)
        src, mix = zero_pad(src), zero_pad(mix)

        mix_spec = np.stack(
            [librosa.stft(np.asfortranarray(x), N_FFT, HOP_LEN) for x in mix],
            axis=1)

        dataset.append((src, mix_spec, f'{speaker0}-{speaker1}'))

    return dataset


def main():
    parser = argparse.ArgumentParser(
        description='Train MVAE with VCC2018 dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train-dataset',
                        help='Path of training dataset.',
                        type=str, required=True)
    parser.add_argument('--val-dataset',
                        help='Path of validation dataset.',
                        type=str, required=True)
    parser.add_argument('--batch-size', '-b',
                        help='Batch size.',
                        type=int, default=32)
    parser.add_argument('--epochs', '-e',
                        help='Number of epochs.',
                        type=int, default=800)
    parser.add_argument('--eval-interval',
                        help='Evaluate and save model every N epochs.',
                        type=int, default=200, metavar='N')
    parser.add_argument('--gpu', '-g',
                        help='GPU id. (Negative number indicates CPU)',
                        type=int, default=-1)
    parser.add_argument('--learning-rate', '-l',
                        help='Learning Rate.',
                        type=float, default=1e-3)
    parser.add_argument('--output',
                        help='Save model to PATH',
                        type=str, default='./models')
    args = parser.parse_args()

    if not os.path.isdir(args.output):
        os.mkdir(args.output)

    if_use_cuda = torch.cuda.is_available() and args.gpu >= 0
    if if_use_cuda:
        device = torch.device(f'cuda:{args.gpu}')
        cp.cuda.Device(args.gpu).use()
    else:
        device = torch.device('cpu')

    train_dataset = torch.load(args.train_dataset)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, args.batch_size, shuffle=True)
    val_dataset = make_eval_set(args.val_dataset)

    baseline = baseline_ilrma(val_dataset, device)

    model = CVAE(n_speakers=train_dataset[0][1].size(0)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)

    # TensorBoard
    writer = SummaryWriter()

    for epoch in range(1, args.epochs + 1):
        train(model, train_dataloader, optimizer, device, epoch, writer)
        if epoch % args.eval_interval == 0:
            validate(model, val_dataset, baseline, device, epoch, writer)
            # Save model
            model.cpu()
            path = os.path.join(args.output, f'model-{epoch}.pth')
            torch.save(model.state_dict(), path)
            model.to(device)

    writer.close()


if __name__ == '__main__':
    main()
