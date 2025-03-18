#!/usr/bin/env python
# -*- coding:utf-8 -*-

import math
import torch
from pathlib import Path
from copy import deepcopy
from collections import OrderedDict
import torch.nn.functional as F

def calculate_parameters(net):
    out = 0
    for param in net.parameters():
        out += param.numel()
    return out

def pad_input(x, mod):
    h, w = x.shape[-2:]
    bottom = int(math.ceil(h/mod)*mod -h)
    right = int(math.ceil(w/mod)*mod - w)
    x_pad = F.pad(x, pad=(0, right, 0, bottom), mode='reflect')
    return x_pad

def forward_chop(net, x, net_kwargs=None, scale=1, shave=10, min_size=160000):
    n_GPUs = 1
    b, c, h, w = x.size()
    h_half, w_half = h // 2, w // 2
    h_size, w_size = h_half + shave, w_half + shave
    lr_list = [
        x[:, :, 0:h_size, 0:w_size],
        x[:, :, 0:h_size, (w - w_size):w],
        x[:, :, (h - h_size):h, 0:w_size],
        x[:, :, (h - h_size):h, (w - w_size):w]]

    if w_size * h_size < min_size:
        sr_list = []
        for i in range(0, 4, n_GPUs):
            lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)
            if net_kwargs is None:
                sr_batch = net(lr_batch)
            else:
                sr_batch = net(lr_batch, **net_kwargs)
            sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))
    else:
        sr_list = [
            forward_chop(patch, shave=shave, min_size=min_size) \
            for patch in lr_list
        ]

    h, w = scale * h, scale * w
    h_half, w_half = scale * h_half, scale * w_half
    h_size, w_size = scale * h_size, scale * w_size
    shave *= scale

    output = x.new(b, c, h, w)
    output[:, :, 0:h_half, 0:w_half] \
        = sr_list[0][:, :, 0:h_half, 0:w_half]
    output[:, :, 0:h_half, w_half:w] \
        = sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
    output[:, :, h_half:h, 0:w_half] \
        = sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
    output[:, :, h_half:h, w_half:w] \
        = sr_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

    return output

def measure_time(net, inputs, num_forward=100):
    '''
    Measuring the average runing time (seconds) for pytorch.
    out = net(*inputs)
    '''
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    with torch.set_grad_enabled(False):
        for _ in range(num_forward):
            out = net(*inputs)
    end.record()

    torch.cuda.synchronize()

    return start.elapsed_time(end) / 1000

def reload_model(model, ckpt):
    module_flag = list(ckpt.keys())[0].startswith('module.')
    compile_flag = '_orig_mod' in list(ckpt.keys())[0]

    for source_key, source_value in model.state_dict().items():
        target_key = source_key
        if compile_flag and (not '_orig_mod.' in source_key):
            target_key = '_orig_mod.' + target_key
        if module_flag and (not source_key.startswith('module')):
            target_key = 'module.' + target_key

        assert target_key in ckpt
        source_value.copy_(ckpt[target_key])


def mem_use(fn, kwargs, title, grad_computation=True, verbose=True, warm_up_runs=10):
    """
    Print running time and peak memory consumption during forward process, including multiple warm-up runs.
    Args:
        fn: forward function to be evaluated
        kwargs: the forward arguments
        title: name of the model
        grad_computation: If True, keep gradient context during forward process
        verbose: If True, print the running time and peak memory consumption
        warm_up_runs: Number of warm-up runs before the actual measurement

    """
    import time
    import torch

    # Clear any cached memory to get a clean measure of memory usage
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Perform warm-up runs to initialize all necessary libraries and cache function calls
    for _ in range(warm_up_runs):
        with torch.no_grad():
            fn(**kwargs)

    # Reset peak memory statistics after warm-up
    torch.cuda.reset_peak_memory_stats()
    temp = torch.cuda.memory_allocated()

    # Time the execution of the function
    start = time.time()
    if grad_computation:
        fn(**kwargs)
    else:
        with torch.no_grad():
            fn(**kwargs)
    torch.cuda.synchronize()  # ensure all CUDA kernels have finished
    stop = time.time()

    # Calculate peak memory usage and running time
    max_memory = (torch.cuda.max_memory_allocated() - temp) // 2 ** 20  # Convert bytes to megabytes
    run_time = round((stop - start) * 1e6) / 1e3  # Convert seconds to milliseconds

    if verbose:
        print(f"{title} - Peak memory use: {max_memory}MB - {run_time}ms")

    return max_memory, run_time

def groupnorm_macs_count(model, x, y):
    model.total_ops += torch.DoubleTensor([x[0].numel() * (1 + 1 + 1 + 2)]) # 1 for affine, 1 for normalization, 1 for mean, 2 for std