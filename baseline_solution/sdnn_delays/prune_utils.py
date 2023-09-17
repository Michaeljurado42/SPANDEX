# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: MIT
# See: https://spdx.org/licenses/
from lava.lib.dl.slayer.synapse import Dense
import numpy as np
import torch.nn.utils.prune as prune
import torch
import collections
import heapq
def get_prune_params(model):
    parameters_to_prune = []
    for name, module in model.named_modules():
        if "Dense" in str(type(module)) and hasattr(module, 'weight'):
            parameters_to_prune.append((module, 'weight'))

    parameters_to_prune = tuple(parameters_to_prune)
    return parameters_to_prune

def sparsity_statistics(model, threshold=1e-5):
    total_weights = 0
    total_zero_weights = 0
    module_sparsity = {}

    for module, param_name in get_prune_params(model):
        param = getattr(module, param_name)
        weights = param.data.cpu().numpy()
        module_weights = weights.size
        module_zero_weights = (np.abs(weights) < threshold).sum()
        module_sparsity[module] = module_zero_weights / module_weights
        
        total_weights += module_weights
        total_zero_weights += module_zero_weights

    sparsity = total_zero_weights / total_weights
    return sparsity, module_sparsity

def total_weights(model):
    total_weights_count = 0

    for module, param_name in get_prune_params(model):
        if "Dense" in str(type(module)) and hasattr(module, 'weight'):
            param = getattr(module, param_name)
            weights = param.data.cpu().numpy()
            total_weights_count += weights.size
    return total_weights_count

def remove_pruning(model):
    for module, param_name in get_prune_params(model):
        prune.remove(module, param_name)


def prune_model(model, desired_sparsity):
    # Collect all weights
    all_weights = []
    for module, _ in get_prune_params(model):
        weights = module.weight.data.cpu().numpy().flatten()
        all_weights += list(weights)

    # Determine the threshold for pruning based on the pruning_rate
    all_weights = np.array(all_weights)
    threshold = np.percentile(np.abs(all_weights), desired_sparsity * 100)

    # Apply pruning based on the global threshold
    for module, name in get_prune_params(model):
        prune.custom_from_mask(module, name=name, mask=torch.abs(module.weight) > threshold)

    return model

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

# def prune_model(model, pruning_rate):
#     # Collect all weights
#     all_weights = []
#     for module, _ in get_prune_params(model):
#         weights = module.weight.data.cpu().numpy().flatten()
#         all_weights += list(weights)

#     # Determine the k-th value for pruning based on the pruning_rate
#     k = int(pruning_rate * len(all_weights))
#     threshold = heapq.nsmallest(k, np.abs(all_weights))[-1]

#     # Apply pruning based on the global threshold
#     for module, name in get_prune_params(model):
#         prune.custom_from_mask(module, name=name, mask=torch.abs(module.weight) >= threshold)

#     return model