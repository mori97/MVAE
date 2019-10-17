# MVAE

A PyTorch implementation of multichannel CVAE(MVAE).

[[arXiv](https://arxiv.org/abs/1808.00892)][[The MIT Press](https://www.mitpressjournals.org/doi/abs/10.1162/neco_a_01217)]

Kameoka, H., Li, L., Inoue, S., & Makino, S. (2018). Semi-blind source separation with
multichannel variational autoencoder. arXiv:1808.00892.

H.Kameoka, L.Li, S.Inoue, and S.Makino, “Supervised DeterminedSource Separation with Multichannel Variational Autoencoder”, NeuralComputation, vol.31, pp.1891–1914, jul 2019.

## Prerequisites

See `requirements.txt`.

## Usage

First, you should make the training set and the mixture signals for evaluation with the VCC2018 corpus.
```bash
$ python make_dataset.py --dataset-dir {path_to_vcc2018} --output dataset.pth
$ python make_mixutre.py --dataset-dir {path_to_vcc2018} --output mixture
```

Then, start training and evaluation by
```bash
$ python main.py --train-dataset dataset.pth --val-dataset mixture --gpu 0
```

The details of commandline arguments can be confirmed by `python main.py --help`.

You can see the evaluation results and the separated signals with Tensorboard.
Note that Tensorflow should be installed.
