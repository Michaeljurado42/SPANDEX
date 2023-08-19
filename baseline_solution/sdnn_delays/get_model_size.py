import numpy as np
from lava.lib.dl import netx
import argparse

def calculate_kilobyte_layer_sparse(layer):
    synapse = layer.synapse
    weights = synapse.weights.get()
    weight_mask = (weights != 0)
    non_zero_weights = np.sum(weight_mask)
    synapse_size = non_zero_weights * 8
    if hasattr(synapse, "delays"):
        delays = layer.synapse.delays.get()
        delays_size = non_zero_weights * 6
    else:
        delays_size= 0
    return (synapse_size + delays_size)/ 8/ 1024


def count_weights(layer):
    if hasattr(layer, "synapse"):  # center layer or output layer
        synapse = layer.synapse
        weights = synapse.weights.get()
        weight_mask = (weights != 0)
        non_zero_weights = np.sum(weight_mask)
        return non_zero_weights
    return 0

def main(trained_folder):
    net = netx.hdf5.Network(trained_folder + '/network.net')
    total_kilobytes = 0
    total_non_zero_weights = 0

    for layer in net.layers[1:]:
        total_kilobytes += calculate_kilobyte_layer_sparse(layer)
        total_non_zero_weights += count_weights(layer)

    print("Total kilobytes of model taking into account sparsity:", total_kilobytes)
    print("Total number of (non-zero weights):", total_non_zero_weights)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate model size and non-zero weights.')
    parser.add_argument('trained_folder', type=str, help='Path to the trained folder.')
    args = parser.parse_args()
    main(args.trained_folder)
    print("----------------")
    print("Bamsumit model estimate")
    trained_folder = args.trained_folder
    net = netx.utils.NetDict(trained_folder + '/network.net')
    weights = [net['layer'][l]['weight'] for l in range(1, 4)]
    delays = [net['layer'][l]['delay'] for l in range(1, 3)]
    model_bits = sum([np.ceil(np.log2(np.abs(w).max())) * np.prod(w.shape) for w in weights]) + sum([np.ceil(np.log2(np.abs(d)).max()) * np.prod(d.shape) for d in delays])
    print('Model Size (KB):', model_bits / 8 / 1024)

