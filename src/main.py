import argparse
import pickle

import torch
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

from cvae import CVAE, lossfun


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


def main():
    parser = argparse.ArgumentParser(
        description='Train MVAE with VCC2018 dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', '-d',
                        help='Path of VCC2018 dataset.',
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

    with open(args.dataset, 'rb') as f:
        train_dataset = pickle.load(f)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, args.batch_size, shuffle=True)

    model = CVAE(n_speakers=train_dataset[0].size(0)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)

    # TensorBoard
    writer = SummaryWriter()

    for epoch in range(1, args.epochs + 1):
        train(model, train_dataloader, optimizer, device, epoch, writer)

    writer.close()


if __name__ == '__main__':
    main()

