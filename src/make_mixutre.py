import argparse
from itertools import combinations, product
import os
import random

import librosa
import numpy as np
import pyroomacoustics as pra
from scipy.io import wavfile
from scipy.signal import convolve


def get_rir(echoic=False):
    """Get the room impulse response (RIR).

    Args:
        echoic (bool): if use an echoic environment.

    Returns:
        ret (numpy.ndarray): A 3-dimensional array that ret[m][s] is the RIR
            from the s-th source to the m-th microphone.
    """
    absorption = 0.2 if echoic else 0.8
    room = pra.ShoeBox([6, 5, 3], fs=16000,
                       absorption=absorption, max_order=10)

    mic_coordinates = np.c_[[2.0, 1.95, 1.5], [2.0, 2.05, 1.5]]
    mic_array = pra.MicrophoneArray(mic_coordinates, room.fs)
    room.add_microphone_array(mic_array)

    room.add_source([2.69, 2.40, 1.75])
    room.add_source([2.61, 1.57, 1.75])

    room.compute_rir()

    ret = room.rir
    min_len = min(len(ret[0][0]), len(ret[0][1]),
                  len(ret[1][0]), len(ret[1][1]))
    for m, s in product(range(2), range(2)):
        ret[m][s] = ret[m][s][:min_len]

    return ret


def make_eval_dataset(dataset_dir, output_path, speakers=None, echoic=False,
                      max_per_speaker=10):
    """Make evaluation dataset for MVAE from VCC2018.

    Args:
        dataset_dir (str): Path of VCC2018.
        output_path (str): Output path.
        speakers (List[str]): Speakers to be used.
        echoic (bool): if use an echoic environment.
        max_per_speaker (int): Use max N speakers.
    """
    evaluation_dir = os.path.join(dataset_dir, 'vcc2018_evaluation')
    if speakers is None:
        speakers = [speaker for speaker in os.listdir(evaluation_dir)
                    if speaker.startswith('VCC2') and
                    os.path.isdir(os.path.join(evaluation_dir, speaker))]

    wav_file_list = [f'300{i:0>2}.wav' for i in range(1, 36)]
    rir = get_rir(echoic)

    # Need shuffle
    for speaker0, speaker1 in combinations(speakers, 2):
        n_output = 0
        comb = list(combinations(wav_file_list, 2))
        random.shuffle(comb)
        for wav0, wav1 in comb:
            if n_output >= max_per_speaker:
                break
            wav0_path = os.path.join(evaluation_dir, speaker0, wav0)
            wav1_path = os.path.join(evaluation_dir, speaker1, wav1)
            src0, fs0 = librosa.load(wav0_path)
            src1, fs1 = librosa.load(wav1_path)

            assert fs0 == 22050, fs1 == 22050
            if src0.shape[0] > 3 * 22050 and src1.shape[0] > 3 * 22050:
                n_output += 1
            else:
                continue

            src0 = librosa.resample(src0, 22050, 16000)
            src1 = librosa.resample(src1, 22050, 16000)
            min_len = min(len(src0), len(src1))
            src0, src1 = src0[:min_len], src1[:min_len]

            mic0 = convolve(src0, rir[0][0]) + convolve(src1, rir[0][1])
            mic1 = convolve(src0, rir[1][0]) + convolve(src1, rir[1][1])
            mixture = np.stack((mic0, mic1), axis=1)

            source = np.stack((src0, src1), axis=1)
            filename = f'{speaker0[-3:]}{wav0[3:5]}' \
                       f'_{speaker1[-3:]}{wav1[3:5]}_src.wav'
            path = os.path.join(output_path, filename)
            wavfile.write(path, 16000, source)

            filename = f'{speaker0[-3:]}{wav0[3:5]}' \
                       f'_{speaker1[-3:]}{wav1[3:5]}_mix.wav'
            path = os.path.join(output_path, filename)
            wavfile.write(path, 16000, mixture)


def main():
    parser = argparse.ArgumentParser(
        description='Make speech mixture from VCC2018 dataset.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset-dir', '-d',
                        help='Path of VCC2018 dataset.',
                        type=str, required=True)
    parser.add_argument('--echoic', '-e',
                        help='Use an echoic environment.',
                        action='store_true')
    parser.add_argument('--output', '-o',
                        help='Output path.',
                        type=str, required=True)
    parser.add_argument('--speakers', '-s',
                        help='Select which speaker to use.',
                        type=str, nargs='+',
                        default=('VCC2SM1', 'VCC2SM2', 'VCC2SF1', 'VCC2SF2'))
    args = parser.parse_args()

    make_eval_dataset(args.dataset_dir, args.output,
                      args.speakers, args.echoic)


if __name__ == '__main__':
    main()
