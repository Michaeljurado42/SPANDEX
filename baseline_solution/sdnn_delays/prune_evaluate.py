# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: MIT
# See: https://spdx.org/licenses/
import collections
import argparse
import sys
sys.path.append('../..')
import numpy as np
import random

import matplotlib.pyplot as plt
import time
from datetime import datetime
import yaml
import librosa
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import IPython.display as ipd

from lava.lib.dl import slayer
from audio_dataloader import DNSAudio
from snr import si_snr
from dnsmos import DNSMOS

# %%
from train_sdnn import collate_fn, stft_splitter, stft_mixer, nop_stats, Network

# %% [markdown]
# # Gather the network statistics
# ## 1. Overload N-DNS `Network` definition
# 
# Modify network's `forward` method to log spiking events in each layer.

# %%
class InferenceNet(Network):
    def forward(self, noisy):
        x = noisy - self.stft_mean

        counts = []
        for i, block in enumerate(self.blocks):
            # print("Layer", i)
            # print("x.shape", x.shape)
            # if hasattr(block, "synapse"):
            #     print("synapse shape:", block.synapse.weight[:, :].shape)
            # else:
            #     print("no synapse (input layer)")
            # print("-------")
            x = block(x)
            count = torch.mean((torch.abs(x) > 0).to(x.dtype))
            counts.append(count.item())
        mask = torch.relu(x + 1)
        return slayer.axon.delay(noisy, self.out_delay) * mask, torch.tensor(counts)

class AdvancedInferenceNet(Network):
    def forward(self, noisy):
        x = noisy - self.stft_mean

        counts = []
        synops = []  # List to store synaptic operations
        for i, block in enumerate(self.blocks):
            if hasattr(block, "synapse"):
                # Find the non-zero spikes in x
                non_zero_spikes_indices = torch.where(torch.abs(x) > 0)

                # Initialize synaptic operations for this block
                synaptic_operations = 0

                # Iterate through non-zero spikes and count the number of non-zero connections for each
                for idx in non_zero_spikes_indices:
                    incoming_neuron_idx = idx[1]  # Assuming x has shape [batch, neurons, time]

                    # Count non-zero weights for this incoming neuron
                    non_zero_weights_count = torch.sum(block.synapse.weight[:, incoming_neuron_idx, :, :, :] < 1e-10).item()
                    synaptic_operations += non_zero_weights_count

                synops.append(synaptic_operations)

            x = block(x)
            count = torch.mean((torch.abs(x) > 0).to(x.dtype))
            counts.append(count.item())

        mask = torch.relu(x + 1)
        return slayer.axon.delay(noisy, self.out_delay) * mask, torch.tensor(counts), synops  # Return synops
def remove_pruning_from_state_dict(state_dict):
    new_state_dict = collections.OrderedDict()
    
    # Iterate through the original weights
    for key, value in state_dict.items():
        if key.endswith('_orig'):
            # Find the corresponding mask
            mask_key = key.replace('_orig', '_mask')
            mask = state_dict.get(mask_key, None)
            
            # Apply the mask to the original weight
            if mask is not None:
                pruned_weight = value * mask
                
                # Remove the '_orig' suffix to revert to the original name
                new_key = key.replace('_orig', '')
                new_state_dict[new_key] = pruned_weight
        elif not key.endswith('_mask'): # Exclude the mask itself
            new_state_dict[key] = value

    # Return the modified state dictionary
    return new_state_dict
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Your description here.")
    parser.add_argument("--trained_folder", type=str,
                        help="Path to the trained folder.")
    args = parser.parse_args()
    print(args)

    # %% [markdown]
    # ## 2. Read the training hyperparameters

    # %%
    trained_folder = args.trained_folder
    args = yaml.safe_load(open(trained_folder + '/args.txt', 'rt'))
    if 'out_delay' not in args.keys():
        args['out_delay'] = 0
    if 'n_fft' not in args.keys():
        args['n_fft'] = 512
    device = torch.device('cuda:0')
    #device = torch.device('cpu')
    #root = args['path']
    root = '/gv1/projects/neuromorphics/dns/'
    out_delay = args['out_delay']
    n_fft = args['n_fft']
    win_length = n_fft
    hop_length = n_fft // 4
    stats = slayer.utils.LearningStats(accuracy_str='SI-SNR', accuracy_unit='dB')

    # %% [markdown]
    # ## 3. Create dataset and dataloader instances

    # %%
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    validation_set = DNSAudio(root=root + 'validation_set/')
    validation_loader = DataLoader(validation_set,
                            batch_size=96,
                            shuffle=True,
                            collate_fn=collate_fn,
                            num_workers=32,
                            persistent_workers = True,
                            pin_memory=True)


    # %% [markdown]
    # ## 4. Instantiate N-DNS network

    # %%
    net = InferenceNet(args['threshold'],
                    args['tau_grad'],
                    args['scale_grad'],
                    args['dmax'],
                    args['out_delay']).to(device)

    # %% [markdown]
    # ## 5. Load trained network

    # %%
    noisy, clean, noise, metadata = validation_set[0]
    noisy = torch.unsqueeze(torch.FloatTensor(noisy), dim=0).to(device)
    noisy_abs, noisy_arg = stft_splitter(noisy, n_fft)
    net(noisy_abs)

    # Load the saved state dictionary
    loaded = torch.load(trained_folder + '/network.pt')
    net.load_state_dict(remove_pruning_from_state_dict(loaded))
    net(noisy_abs)

    # %%
    dnsmos = DNSMOS()
    dnsmos_noisy = np.zeros(3)
    dnsmos_clean = np.zeros(3)
    dnsmos_noise = np.zeros(3)
    dnsmos_cleaned  = np.zeros(3)
    valid_event_counts = []

    t_st = datetime.now()
    from multiprocessing import Pool

    def compute_dnsmos(sample):
        return dnsmos(sample[None, ...])[0]
    print("starting loop")
    with Pool(96) as pool:
        for i, (noisy, clean, noise) in enumerate(validation_loader):
            net.eval()
            with torch.no_grad():
                print(i)
                start = time.time()
                noisy = noisy.to(device)
                clean = clean.to(device)

                noisy_abs, noisy_arg = stft_splitter(noisy, n_fft)
                clean_abs, clean_arg = stft_splitter(clean, n_fft)

                denoised_abs, count = net(noisy_abs)
                valid_event_counts.append(count.cpu().data.numpy())
                noisy_arg = slayer.axon.delay(noisy_arg, out_delay)
                clean_abs = slayer.axon.delay(clean_abs, out_delay)
                clean = slayer.axon.delay(clean, win_length * out_delay)

                loss = F.mse_loss(denoised_abs, clean_abs)
                clean_rec = stft_mixer(denoised_abs, noisy_arg, n_fft)
                score = si_snr(clean_rec, clean)
                combined_data = np.concatenate([noisy.cpu().data.numpy(), clean.cpu().data.numpy(), noise.cpu().data.numpy(), clean_rec.cpu().data.numpy()])
                result = np.array(pool.map(compute_dnsmos, combined_data))
                split_result = np.array_split(result, 4, axis = 0)
                dnsmos_noisy += np.sum(split_result[0], axis=0)
                dnsmos_clean += np.sum(split_result[1], axis=0)
                dnsmos_noise += np.sum(split_result[2], axis=0)
                dnsmos_cleaned += np.sum(split_result[3], axis=0)

                stats.validation.correct_samples += torch.sum(score).item()
                stats.validation.loss_sum += loss.item()
                stats.validation.num_samples += noisy.shape[0]

                processed = i * validation_loader.batch_size
                total = len(validation_loader.dataset)
                time_elapsed = (datetime.now() - t_st).total_seconds()
                samples_sec = time_elapsed / (i + 1) / validation_loader.batch_size
                header_list = [f'Valid: [{processed}/{total} '
                                f'({100.0 * processed / total:.0f}%)]']
                header_list.append(f'Event rate: {[c.item() for c in count]}')
                print(header_list[0])
                print(header_list[1])
                print("Total time for batch", time.time() - start)
                print(dnsmos_cleaned.shape)
                print(f'\r{header_list[0]}', end='')
                

    dnsmos_clean /= len(validation_loader.dataset)
    dnsmos_noisy /= len(validation_loader.dataset)
    dnsmos_noise /= len(validation_loader.dataset)
    dnsmos_cleaned /= len(validation_loader.dataset)

    print()
    stats.print(0, i, samples_sec, header=header_list)
    print('Avg DNSMOS clean   [ovrl, sig, bak]: ', dnsmos_clean)
    print('Avg DNSMOS noisy   [ovrl, sig, bak]: ', dnsmos_noisy)
    print('Avg DNSMOS noise   [ovrl, sig, bak]: ', dnsmos_noise)
    print('Avg DNSMOS cleaned [ovrl, sig, bak]: ', dnsmos_cleaned)

    mean_events = np.mean(valid_event_counts, axis=0)

    neuronops = []
    for block in net.blocks[:-1]:
        neuronops.append(np.prod(block.neuron.shape))

    # for events, block in zip(mean_events, net.blocks[1:]):
    #     synops.append(events * np.prod(block.synapse.shape))
    synops = []
    for events, block in zip(mean_events, net.blocks[1:]):
        # Calculate the percentage of non-zero weights
        non_zero_weights_count = torch.sum(block.synapse.weight.abs() > 1e-10).item()
        total_weights_count = block.synapse.weight.numel()
        non_zero_percentage = non_zero_weights_count / total_weights_count

        # Compute the synaptic operations and multiply by the percentage of non-zero weights
        synaptic_operations = events * np.prod(block.synapse.shape) * non_zero_percentage

        # Append the result to the list, rounding to the nearest integer
        synops.append(int(round(synaptic_operations)))

    print(f'SynOPS: {synops}')
    print(f'Total SynOPS: {sum(synops)} per time-step')
    print(f'Total NeuronOPS: {sum(neuronops)} per time-step')
    print(f'Time-step per sample: {noisy_abs.shape[-1]}')

    # %%
    # ipd.Audio(noisy[0].cpu(), rate=16000)

    # # %%
    # ipd.Audio(clean_rec[0].cpu(), rate=16000)

    # # %%
    # ipd.Audio(clean[0].cpu(), rate=16000)

    # %% [markdown]
    # # Latency
    # 
    # $\text{latency} = \text{latency}_\text{buffer} + \text{latency}_\text{enc+dec} + \text{latency}_\text{N-DNS}$

    # %% [markdown]
    # ## 1. Buffer latency
    # 
    # It is the time required to collect data samples needed by the DNS sample at every time-step. For STFT encoder, it is the `window_length` of STFT processing.

    # %%
    dt = hop_length / metadata['fs']
    # buffer_latency = dt
    buffer_latency = win_length / metadata['fs']
    print(f'Buffer latency: {buffer_latency * 1000} ms')

    # %% [markdown]
    # ## 2. Encode+Decode latency
    # 
    # It is the additional processing time introduced by the encoder+decoder blocks. We will measure the actual computation time on a CPU.

    # %%
    t_st = datetime.now()
    for i in range(noisy.shape[0]):
        audio = noisy[i].cpu().data.numpy()
        stft = librosa.stft(audio, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
        istft = librosa.istft(stft,  win_length=win_length, hop_length=hop_length)

    time_elapsed = (datetime.now() - t_st).total_seconds()

    enc_dec_latency = time_elapsed / noisy.shape[0] / 16000 / 30 * hop_length
    print(f'STFT + ISTFT latency: {enc_dec_latency * 1000} ms')

    # %% [markdown]
    # ## 3. N-DNS latency
    # 
    # It is the algorithmic time shift introduced by the N-DNS network (if any). This can be calculated as the peak cross-correlation between the noisy and clean reconstruction audio. This latency should be the desired output delay (`out_delay * dt`) of the network. The cross correlation calculation below is an alternate way of evaluating the N-DNS latency.

    # %%
    dns_delays = []
    max_len = 50000  # Only evaluate for first clip of audio
    for i in range(noisy.shape[0]):
        delay = np.argmax(np.correlate(noisy[i, :max_len].cpu().data.numpy(),
                                    clean_rec[i, :max_len].cpu().data.numpy(),
                                    'full')) - max_len + 1
        dns_delays.append(delay)
    dns_latency = np.mean(dns_delays) / metadata['fs']
    print(f'N-DNS latency: {dns_latency * 1000} ms')

    # %% [markdown]
    # # Audio Quality Metrics
    # 
    # The audio quality metric is measured by
    # 1. $\text{SI-SNR}$ on the validation set.
    # 2. $\text{SI-SNR}$ improvement on the raw audio ($\text{SI-SNRi}_\text{data}$) and encode+decode operation ($\text{SI-SNRi}_\text{enc+dec}$)
    # 
    # > Note: when the testing dataset is released, we shall use $\text{SI-SNR}$ on testing set as the eventual audio quality metric.

    # %%
    base_stats = slayer.utils.LearningStats(accuracy_str='SI-SNR',
                                            accuracy_unit='dB')
    nop_stats(validation_loader, base_stats, base_stats.validation, print=False)

    # %% [markdown]
    # Here, $\text{SI-SNRi}_\text{data}$ and $\text{SI-SNRi}_\text{encode+decode}$ are same as STFT-ISTFT is a lossless transformation.

    # %%
    si_snr_i = stats.validation.accuracy - base_stats.validation.accuracy
    print(f'SI-SNR  (validation set): {stats.validation.accuracy: .2f} dB')
    print(f'SI-SNRi (validation set): {si_snr_i: .2f} dB')

    # %% [markdown]
    # # Computational Metrics
    # 
    # For Track 1, we will use proxies for **power** and **energy-delay-product** as these require access to actual neuromorphic hardware.
    # 
    # 1. __Power proxy:__ To estimate proxy for the power of the N-DNS solution, we use the weighted sum of $\text{NeuronOPS}$ and $\text{SynOPS}$ which typically consume the majority of the power in a neuromorphic system. Based on our silicon characterization of Loihi and Loihi 2, NeuronOps consume approximately $10\times$ energy than SynOps. Therefore, we use the following power proxy:
    # 
    #     $P_\text{proxy} = \text{Effective SynOPS} = \text{SynOPS} + 10 \times \text{NeuronOPS}$
    # 
    # 2. __Power delay product proxy:__ Power delay product provides a combined metric of power and latency of the solution. The proxy is defined as:
    # 
    #     $\text{PDP}_\text{proxy} = \text{SynOPS delay product} = P_\text{proxy} \times \text{latency}$

    # %%
    latency = buffer_latency + enc_dec_latency + dns_latency
    effective_synops_rate = (sum(synops) + 10 * sum(neuronops)) / dt
    synops_delay_product = effective_synops_rate * latency

    print(f'Solution Latency                 : {latency * 1000: .3f} ms')
    print(f'Power proxy (Effective SynOPS)   : {effective_synops_rate:.3f} ops/s')
    print(f'PDP proxy (SynOPS-delay product) : {synops_delay_product: .3f} ops')


