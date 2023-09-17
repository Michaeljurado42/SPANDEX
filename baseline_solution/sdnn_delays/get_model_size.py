# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: MIT
# See: https://spdx.org/licenses/
import numpy as np
from lava.lib.dl import netx
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate model size and non-zero weights.')
    parser.add_argument('trained_folder', type=str, help='Path to the trained folder.')
    args = parser.parse_args()
    
    trained_folder = args.trained_folder
    net = netx.hdf5.Network(trained_folder + '/network.net', sparsity_map=True)
    weights = [l.synapse.weights.get().toarray() for l in net.layers[1:4]] # convert from csr matrices
    num_weights = sum([np.sum(w != 0) for w in weights])
    delays = [l.synapse.delays.get().toarray() for l in net.layers[1:3]]
    model_bits = sum([np.ceil(np.log2(np.abs(w).max())) * (w != 0).sum() for w in weights]) + sum([np.ceil(np.log2(np.abs(d)).max()) * (d != 0).sum() for d in delays])
    print('Model Size (KB) (Sparsity Estimate):', model_bits / 8 / 1024)
    print("Total number of non-zero weights:",  num_weights)

    # note I am pretty sure something is wrong with this estimate. The delays are 1d for example when they should be 2d
    print("Bamsumit model estimate") 
    net = netx.utils.NetDict(trained_folder + '/network.net')
    weights = [net['layer'][l]['weight'] for l in range(1, 4)]
    delays = [net['layer'][l]['delay'] for l in range(1, 3)]
    model_bits = sum([np.ceil(np.log2(np.abs(w).max())) * np.prod(w.shape) for w in weights]) + sum([np.ceil(np.log2(np.abs(d)).max()) * np.prod(d.shape) for d in delays])
    print('Model Size (KB):', model_bits / 8 / 1024)
