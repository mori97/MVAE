import argparse
import os
import pickle
import re

import mir_eval
import numpy as np
from scipy.io import wavfile
import scipy.signal as signal
import torch
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

from cvae import CVAE, lossfun
from ilrma import ilrma


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
        optimizer.step()

    writer.add_scalar('Loss/train', total_loss / len(data_loader), epoch)


def baseline_ilrma(val_dataset):
    """Evaluate with ILRMA.
    """
    ret = {}

    for src, mix_spec, speaker in val_dataset:
        src = src.T  # (n_ch, t)
        separated, _ = ilrma(mix_spec, n_iter=100)
        _, separated = signal.istft(separated, fs=16000, nperseg=4096,
                                    time_axis=2, freq_axis=0)

        min_size = min(src.shape[1], separated.shape[1])
        src, separated = src[:, :min_size], separated[:, :min_size]
        sdr, sir, sar, _ =\
            mir_eval.separation.bss_eval_sources(src, separated)

        if speaker in ret:
            ret[speaker]['SDR'].append(sdr)
            ret[speaker]['SIR'].append(sir)
            ret[speaker]['SAR'].append(sar)
        else:
            ret[speaker] = {'SDR': [], 'SIR': [], 'SAR': []}

    for speaker in ret:
        for k in ret[speaker]:
            ret[speaker][k] = np.mean(np.concatenate(ret[speaker][k]))

    return ret


def make_eval_set(path):
    """Make the evaluation dataset.
    """
    src_wav_files = [x for x in os.listdir(path) if x.endswith('_src.wav')]
    ptn = r'(?P<speaker0>[A-Z]{2}\d)(?P<num0>\d\d)_' \
          r'(?P<speaker1>[A-Z]{2}\d)(?P<num1>\d\d)_src\.wav$'
    prog = re.compile(ptn)

    dataset = []
    for src_wav_file in src_wav_files:
        result = prog.match(src_wav_file)
        speaker0, file_num0 = result.group('speaker0'), result.group('num0')
        speaker1, file_num1 = result.group('speaker1'), result.group('num1')

        mix_wav_file = f'{speaker0}{file_num0}_{speaker1}{file_num1}_mix.wav'
        src_wav_path = os.path.join(path, src_wav_file)
        mix_wav_path = os.path.join(path, mix_wav_file)

        src_fs, src = wavfile.read(src_wav_path)
        mix_fs, mix = wavfile.read(mix_wav_path)
        assert src_fs == 16000, mix_fs == 16000

        _, _, mix_spec = signal.stft(src, nperseg=4096, axis=0)  # (F, C, T)

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
                        type=int, default=100)
    parser.add_argument('--gpu', '-g',
                        help='GPU id. (Negative number indicates CPU)',
                        type=int, default=-1)
    parser.add_argument('--learning-rate', '-l',
                        help='Learning Rate.',
                        type=float, default=1e-4)
    args = parser.parse_args()

    if_use_cuda = torch.cuda.is_available() and args.gpu >= 0
    device = torch.device(f'cuda:{args.gpu}' if if_use_cuda else 'cpu')

    with open(args.train_dataset, 'rb') as f:
        train_dataset = pickle.load(f)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, args.batch_size, shuffle=True)
    val_dataset = make_eval_set(args.val_dataset)

    model = CVAE(n_speakers=train_dataset[0].size(0)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)

    # TensorBoard
    writer = SummaryWriter()

    for epoch in range(1, args.epochs + 1):
        train(model, train_dataloader, optimizer, device, epoch, writer)

    writer.close()


if __name__ == '__main__':
    main()
