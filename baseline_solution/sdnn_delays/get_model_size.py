import numpy as np
from lava.lib.dl import netx
import argparse

def calculate_kilobyte_layer(layer):
    if hasattr(layer, "synapse"):  # center layer or output layer
        synapse = layer.synapse
        weights = synapse.weights.get()
        weight_mask = (weights != 0)
        non_zero_weights = np.sum(weight_mask)
        synapse_size = non_zero_weights * 8 + non_zero_weights * 6
    else:
        synapse_size = 0  # Input layer

    neuron = layer.neuron
    sigma_size = np.prod(neuron.sigma.shape) * 24
    sigma_size = 0
    vth_size = act_size = residue_size = error_size = 0
    bias_size = spike_exp_size = state_exp_size = cum_error_size = 0

    if hasattr(neuron, "vth"):  # output layer
        act_size = np.prod(neuron.act.shape) * 24
        cum_error_size = np.prod(neuron.cum_error.shape) * 3
        error_size = np.prod(neuron.error.shape) * 24
        residue_size = np.prod(neuron.residue.shape) * 24
        spike_exp_size = np.prod(neuron.spike_exp.shape) * 3
        state_exp_size = np.prod(neuron.state_exp.shape) * 3
        vth_size = np.prod(neuron.vth.shape) * 24

        if hasattr(neuron, "bias"):  # SigmaDelta (center layer)
            bias_size = np.prod(neuron.bias.shape) * 16
    return (
        synapse_size
        + vth_size
        + sigma_size
        + act_size
        + residue_size
        + error_size
        + bias_size
        + spike_exp_size
        + state_exp_size
        + cum_error_size
    ) / 8 / 1024

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

    for layer in net.layers:
        total_kilobytes += calculate_kilobyte_layer(layer)
        total_non_zero_weights += count_weights(layer)

    print("Total kilobytes of model taking into account sparsity:", total_kilobytes)
    print("Total number of (non-zero weights):", total_non_zero_weights)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate model size and non-zero weights.')
    parser.add_argument('trained_folder', type=str, help='Path to the trained folder.')
    args = parser.parse_args()
    main(args.trained_folder)


