import argparse
import os
import pickle

import torch
import torchaudio
from torchaudio.transforms import Resample, Spectrogram


def make_train_dataset(dataset_dir, speakers=None):
    """Make the training dataset for MVAE from the VCC2018 dataset.

    Args:
        dataset_dir (str): Path of the VCC2018 dataset.
        speakers (List[str]): Speakers to be used.

    Returns:
        List[Tuple[torch.Tensor, torch.Tensor]]:
            List of spectrogram and speaker label.
    """
    training_dir = os.path.join(dataset_dir, 'vcc2018_training')
    if speakers is None:
        speakers = [speaker for speaker in os.listdir(training_dir)
                    if speaker.startswith('VCC2') and
                    os.path.isdir(os.path.join(training_dir, speaker))]

    resample = Resample(22050, 16000)
    create_spectrogram = Spectrogram(n_fft=4096)

    dataset = []
    with torch.no_grad():
        for c, speaker in enumerate(speakers):
            speaker_dir = os.path.join(training_dir, speaker)
            wav_files = [os.path.join(speaker_dir, wav_file)
                         for wav_file in os.listdir(speaker_dir)
                         if os.path.splitext(wav_file)[1] == '.wav']
            spectrogram = []
            for wav_file in wav_files:
                sound, _ = torchaudio.load(wav_file)
                sound = resample(sound)
                spectrogram.append(create_spectrogram(sound).squeeze(0))
            spectrogram = torch.cat(spectrogram, dim=1)

            length = 64
            overlap = 48
            for n in range((spectrogram.size(1) - length) // (length - overlap) + 1):
                start = n * (length - overlap)
                data = spectrogram[:, start:start + length]
                label = torch.zeros(len(speakers))
                label[c] = 1
                dataset.append((data, label))

    return dataset


def main():
    parser = argparse.ArgumentParser(
        description='Make training dataset for MVAE from VCC2018 dataset.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset-dir', '-d',
                        help='Path of VCC2018 dataset.',
                        type=str, required=True)
    parser.add_argument('--speakers', '-s',
                        help='Select which speaker to use.',
                        type=str, nargs='+',
                        default=('VCC2SM1', 'VCC2SM2', 'VCC2SF1', 'VCC2SF2'))
    parser.add_argument('--output', '-o',
                        help='Output path.',
                        type=str, required=True)
    args = parser.parse_args()

    dataset = make_train_dataset(args.dataset_dir, args.speakers)

    with open(args.output, 'wb') as f:
        pickle.dump(dataset, f)


if __name__ == '__main__':
    main()
