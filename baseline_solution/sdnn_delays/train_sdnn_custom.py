# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: MIT
# See: https://spdx.org/licenses/


import sys
sys.path.append('../..')

# Enable stack trace for all warnings
import warnings
warnings.simplefilter("always")
import os
import h5py
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import torch.nn.utils.prune as prune
import torch
from prune_utils import get_prune_params, sparsity_statistics, total_weights, remove_pruning, prune_model
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from lava.lib.dl import slayer
from audio_dataloader import DNSAudio
from snr import si_snr
from torch.utils.data import Dataset
import torch

def print_gpu_memory_usage(device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        torch.cuda.empty_cache()
        current_memory_allocated = torch.cuda.memory_allocated(device)
        max_memory_allocated = torch.cuda.max_memory_allocated(device)
        total_memory = torch.cuda.get_device_properties(device).total_memory

        current_memory_percentage = (current_memory_allocated / total_memory) * 100
        max_memory_percentage = (max_memory_allocated / total_memory) * 100

        print(f"Device: {device}")
        print(f"Current memory allocated: {current_memory_percentage:.2f}%")
        print(f"Maximum memory allocated during this run: {max_memory_percentage:.2f}%")
        print(f"Total GPU memory: {total_memory / 1024**3:.2f} GB")
    else:
        print("GPU memory usage information is not available on CPU.")

# Example usage
class SliceableDataset(Dataset):
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset
        self.length = len(original_dataset)
        self.indices = list(range(self.length))

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(self.length)
            return [self.original_dataset[i] for i in range(start, stop, step)]
        else:
            return self.original_dataset[index]

    def __len__(self):
        return self.length

def collate_fn(batch):
    noisy, clean, noise = [], [], []

    for sample in batch:
        noisy += [torch.FloatTensor(sample[0])]
        clean += [torch.FloatTensor(sample[1])]
        noise += [torch.FloatTensor(sample[2])]

    return torch.stack(noisy), torch.stack(clean), torch.stack(noise)


def stft_splitter(audio, n_fft=512):
    with torch.no_grad():
        audio_stft = torch.stft(audio,
                                n_fft=n_fft,
                                onesided=True,
                                return_complex=True)
        return audio_stft.abs(), audio_stft.angle()


def stft_mixer(stft_abs, stft_angle, n_fft=512):
    return torch.istft(torch.complex(stft_abs * torch.cos(stft_angle),
                                        stft_abs * torch.sin(stft_angle)),
                        n_fft=n_fft, onesided=True)


class Network(torch.nn.Module):
    def __init__(self, threshold=0.1, tau_grad=0.1, scale_grad=0.8, max_delay=64, out_delay=0):
        super().__init__()
        self.stft_mean = 0.2
        self.stft_var = 1.5
        self.stft_max = 140
        self.out_delay = out_delay

        sigma_params = { # sigma-delta neuron parameters
            'threshold'     : threshold,   # delta unit threshold
            'tau_grad'      : tau_grad,    # delta unit surrogate gradient relaxation parameter
            'scale_grad'    : scale_grad,  # delta unit surrogate gradient scale parameter
            'requires_grad' : False,  # trainable threshold
            'shared_param'  : True,   # layer wise threshold
        }
        sdnn_params = {
            **sigma_params,
            'activation'    : F.relu, # activation function
        }

        self.input_quantizer = lambda x: slayer.utils.quantize(x, step=1 / 64)

        self.blocks = torch.nn.ModuleList([
            slayer.block.sigma_delta.Input(sdnn_params),
            slayer.block.sigma_delta.Dense(sdnn_params, 257, 512, weight_norm=False, delay=True, delay_shift=True),
            slayer.block.sigma_delta.Dense(sdnn_params, 512, 512, weight_norm=False, delay=True, delay_shift=True),
            slayer.block.sigma_delta.Output(sdnn_params, 512, 257, weight_norm=False),
        ])

        self.blocks[0].pre_hook_fx = self.input_quantizer

        self.blocks[1].delay.max_delay = max_delay
        self.blocks[2].delay.max_delay = max_delay

    def forward(self, noisy):
        x = noisy - self.stft_mean

        for block in self.blocks:
            x = block(x)

        mask = torch.relu(x + 1)
        return slayer.axon.delay(noisy, self.out_delay) * mask

    def grad_flow(self, path):
        # helps monitor the gradient flow
        grad = [b.synapse.grad_norm for b in self.blocks if hasattr(b, 'synapse')]

        plt.figure()
        plt.semilogy(grad)
        plt.savefig(path + 'gradFlow.png')
        plt.close()

        return grad

    def validate_gradients(self):
        valid_gradients = True
        for name, param in self.named_parameters():
            if param.grad is not None:
                valid_gradients = not (torch.isnan(param.grad).any()
                                       or torch.isinf(param.grad).any())
                if not valid_gradients:
                    break
        if not valid_gradients:
            self.zero_grad()

    def export_hdf5(self, filename):
        # network export to hdf5 format
        h = h5py.File(filename, 'w')
        layer = h.create_group('layer')
        for i, b in enumerate(self.blocks):
            b.export_hdf5(layer.create_group(f'{i}'))

        
def nop_stats(dataloader, stats, sub_stats, print=True):
    t_st = datetime.now()
    for i, (noisy, clean, noise) in enumerate(dataloader):
        with torch.no_grad():
            noisy = noisy
            clean = clean

            score = si_snr(noisy, clean)
            sub_stats.correct_samples += torch.sum(score).item()
            sub_stats.num_samples += noisy.shape[0]

            processed = i * dataloader.batch_size
            total = len(dataloader.dataset)
            time_elapsed = (datetime.now() - t_st).total_seconds()
            samples_sec = time_elapsed / (i + 1) / dataloader.batch_size
            header_list = [f'Train: [{processed}/{total} '
                           f'({100.0 * processed / total:.0f}%)]']
            if print:
                stats.print(0, i, samples_sec, header=header_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu',
                        type=int,
                        default=[0],
                        help='which gpu(s) to use', nargs='+')
    parser.add_argument('-b',
                        type=int,
                        default=32,
                        help='batch size for dataloader')
    parser.add_argument('-lr',
                        type=float,
                        default=0.01,
                        help='initial learning rate')
    parser.add_argument('-lam',
                        type=float,
                        default=0.001,
                        help='lagrangian factor')
    parser.add_argument('-threshold',
                        type=float,
                        default=0.1,
                        help='neuron threshold')
    parser.add_argument('-tau_grad',
                        type=float,
                        default=0.1,
                        help='surrogate gradient time constant')
    parser.add_argument('-scale_grad',
                        type=float,
                        default=0.8,
                        help='surrogate gradient scale')
    parser.add_argument('-n_fft',
                        type=int,
                        default=512,
                        help='number of FFT specturm, hop is n_fft // 4')
    parser.add_argument('-dmax',
                        type=int,
                        default=64,
                        help='maximum axonal delay')
    parser.add_argument('-out_delay',
                        type=int,
                        default=0,
                        help='prediction output delay (multiple of 128)')
    parser.add_argument('-clip',
                        type=float,
                        default=10,
                        help='gradient clipping limit')
    parser.add_argument('-exp',
                        type=str,
                        default='',
                        help='experiment differentiater string')
    parser.add_argument('-seed',
                        type=int,
                        default=None,
                        help='random seed of the experiment')
    parser.add_argument('-epoch',
                        type=int,
                        default=50,
                        help='number of args.epoch to run')
    parser.add_argument('-path',
                        type=str,
                        default='/gv1/projects/neuromorphics/dns/',
                        help='dataset path')
    parser.add_argument("--sparsity_target", default = None, type = float)
    parser.add_argument("--pruning_epochs", default = 0, type = float)
    parser.add_argument("--pretrained_weights_folder", default = None, help = "optional string for pretraining weights from another folder")
    parser.add_argument("--trained_folder", "-tf", default = "Trained", type = str, help = "where to save the trained model and the training-testing statistics. Defaults to 'Trained'")
    args = parser.parse_args()

    identifier = args.exp
    if args.seed is not None:
        torch.manual_seed(args.seed)
        identifier += '_{}{}'.format(args.optim, args.seed)

    trained_folder = args.trained_folder + identifier
    logs_folder = 'Logs' + identifier
    writer = SummaryWriter('runs/' + identifier)

    os.makedirs(trained_folder, exist_ok=True)
    os.makedirs(logs_folder, exist_ok=True)

    with open(trained_folder + '/args.txt', 'wt') as f:
        for arg, value in sorted(vars(args).items()):
            f.write('{} : {}\n'.format(arg, value))

    lam = args.lam

    print('Using GPUs {}'.format(args.gpu))
    device = torch.device('cuda:{}'.format(args.gpu[0]))

    out_delay = args.out_delay
    if len(args.gpu) == 1:
        net = Network(args.threshold,
                      args.tau_grad,
                      args.scale_grad,
                      args.dmax,
                      args.out_delay).to(device)
        module = net
    else:
        net = torch.nn.DataParallel(Network(args.threshold,
                                            args.tau_grad,
                                            args.scale_grad,
                                            args.dmax,
                                            args.out_delay).to(device),
                                    device_ids=args.gpu)
        module = net.module


    # Define optimizer module.
    optimizer = torch.optim.RAdam(net.parameters(),
                                  lr=args.lr,
                                  weight_decay=1e-5)

    # Testing Code
    train_set = DNSAudio(root=args.path + 'training_set/',subsample=None)
    validation_set = DNSAudio(root=args.path + 'validation_set/', subsample=6000)

    train_loader = DataLoader(train_set,
                              batch_size=args.b,
                              shuffle=True,
                              collate_fn=collate_fn,
                              num_workers=32,
                              pin_memory=True, persistent_workers=True)
    validation_loader = DataLoader(validation_set,
                                   batch_size=args.b,
                                   shuffle=True,
                                   collate_fn=collate_fn,
                                   num_workers=32,
                                   pin_memory=True, persistent_workers=True)

    base_stats = slayer.utils.LearningStats(accuracy_str='SI-SNR',
                                            accuracy_unit='dB')

    stats = slayer.utils.LearningStats(accuracy_str='SI-SNR',
                                       accuracy_unit='dB')

    if args.pretrained_weights_folder != None:
        print("load in data")
        noisy, clean, noise, metadata = train_set[0]
        noisy = torch.unsqueeze(torch.FloatTensor(noisy), dim=0).to(device)
        noisy_abs, noisy_arg = stft_splitter(noisy, args.n_fft)
        net.train()
        net(noisy_abs)
        net.load_state_dict(torch.load(args.pretrained_weights_folder + "/network.pt"))
    sparsity_target = args.sparsity_target  
    if sparsity_target != None:
        initial_sparsity, _ = sparsity_statistics(net)
        desired_sparsity = initial_sparsity
        steps_per_epoch = len(train_loader)

        total_pruning_steps = args.pruning_epochs * steps_per_epoch - 1
        pruning_frequency = 30
        prune_amount = 0

    for epoch in range(args.epoch):
        t_st = datetime.now()
        for i, (noisy, clean, noise) in enumerate(train_loader):
            net.train()

            if sparsity_target != None:
                global_step = epoch * steps_per_epoch + i
                # Check if within pruning_epochs
                if epoch < args.pruning_epochs and i % pruning_frequency == 0:
                    print("Recomputing desired sparsity at epoch", epoch, "iteration", i)
                    desired_sparsity = initial_sparsity + (sparsity_target - initial_sparsity) * (1 - (1 - global_step / total_pruning_steps) ** 3)
                elif epoch >= args.pruning_epochs:
                    for param_group in optimizer.param_groups:
                        # Reduce the learning rate by the specified factor
                        param_group['lr'] = args.lr * .70               
                average_sparsity, per_module_sparsity = sparsity_statistics(net)
                # Calculate the amount to prune based on the difference between desired and current sparsity

                print(f"Global step: {global_step}")
                print(f"Desired sparsity: {desired_sparsity}")
                print(f"Current average sparsity: {average_sparsity}")

                # Apply pruning only if prune_amount is positive
                print("Applying pruning...")
                prune_model(net, desired_sparsity)
                # Measure the current actual sparsity again
                average_sparsity, per_module_sparsity = sparsity_statistics(net)
                print("Average sparsity after pruning:", average_sparsity)
                print("--------------------")
            noisy = noisy.to(device)
            clean = clean.to(device)

            noisy_abs, noisy_arg = stft_splitter(noisy, args.n_fft)
            clean_abs, clean_arg = stft_splitter(clean, args.n_fft)

            denoised_abs = net(noisy_abs)
            noisy_arg = slayer.axon.delay(noisy_arg, out_delay)
            clean_abs = slayer.axon.delay(clean_abs, out_delay)
            clean = slayer.axon.delay(clean, args.n_fft // 4 * out_delay)

            clean_rec = stft_mixer(denoised_abs, noisy_arg, args.n_fft)

            score = si_snr(clean_rec, clean)
            loss = lam * F.mse_loss(denoised_abs, clean_abs) + (100 - torch.mean(score))

            assert torch.isnan(loss) == False

            optimizer.zero_grad()
            loss.backward()
            net.validate_gradients()
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)
            optimizer.step()

            # if i < 10:
            #     net.grad_flow(path=trained_folder + '/')

            if torch.isnan(score).any():
                score[torch.isnan(score)] = 0

            stats.training.correct_samples += torch.sum(score).item()
            stats.training.loss_sum += loss.item()
            stats.training.num_samples += noisy.shape[0]

            processed = i * train_loader.batch_size
            total = len(train_loader.dataset)
            time_elapsed = (datetime.now() - t_st).total_seconds()
            samples_sec = time_elapsed / (i + 1) / train_loader.batch_size
            header_list = [f'Train: [{processed}/{total} '
                           f'({100.0 * processed / total:.0f}%)]']
            stats.print(epoch, i, samples_sec, header=header_list)
            print(datetime.now())

        t_st = datetime.now()
        torch.save(module.state_dict(), trained_folder + '/%d_network.pt' % epoch)
        for i, (noisy, clean, noise) in enumerate(validation_loader):
            net.eval()

            with torch.no_grad():
                noisy = noisy.to(device)
                clean = clean.to(device)
                
                noisy_abs, noisy_arg = stft_splitter(noisy, args.n_fft)
                clean_abs, clean_arg = stft_splitter(clean, args.n_fft)

                denoised_abs = net(noisy_abs)
                noisy_arg = slayer.axon.delay(noisy_arg, out_delay)
                clean_abs = slayer.axon.delay(clean_abs, out_delay)
                clean = slayer.axon.delay(clean, args.n_fft // 4 * out_delay)

                clean_rec = stft_mixer(denoised_abs, noisy_arg, args.n_fft)
                
                score = si_snr(clean_rec, clean)
                loss = lam * F.mse_loss(denoised_abs, clean_abs) + (100 - torch.mean(score))
                stats.validation.correct_samples += torch.sum(score).item()
                stats.validation.loss_sum += loss.item()
                stats.validation.num_samples += noisy.shape[0]

                processed = i * validation_loader.batch_size
                total = len(validation_loader.dataset)
                time_elapsed = (datetime.now() - t_st).total_seconds()
                samples_sec = time_elapsed / \
                    (i + 1) / validation_loader.batch_size
                header_list = [f'Valid: [{processed}/{total} '
                               f'({100.0 * processed / total:.0f}%)]']
                stats.print(epoch, i, samples_sec, header=header_list)

        writer.add_scalar('Loss/train', stats.training.loss, epoch)
        writer.add_scalar('Loss/valid', stats.validation.loss, epoch)
        writer.add_scalar('SI-SNR/train', stats.training.accuracy, epoch)
        writer.add_scalar('SI-SNR/valid', stats.validation.accuracy, epoch)

        stats.update()
        stats.plot(path=trained_folder + '/')
        if epoch < args.pruning_epochs or stats.validation.best_accuracy is True:
            print("Saving best model")
            torch.save(module.state_dict(), trained_folder + '/network.pt')
        stats.save(trained_folder + '/')
        print_gpu_memory_usage()
    net.load_state_dict(torch.load(trained_folder + '/network.pt'))
    net.export_hdf5(trained_folder + '/network.net')

    params_dict = {}
    for key, val in args._get_kwargs():
        params_dict[key] = str(val)
    writer.add_hparams(params_dict, {'SI-SNR': stats.validation.max_accuracy})
    writer.flush()
    writer.close()