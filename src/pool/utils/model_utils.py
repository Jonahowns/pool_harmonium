import json
import sys

import yaml
import math
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.optim import SGD, AdamW, Adagrad, Adadelta, Adam  # Supported Optimizers

from pool.utils.graph_utils import sequence_logo, sequence_logo_multiple, sequence_logo_all


def load_weight_file(data_dir, filename, key, transform=None):
    try:
        with open(data_dir + filename) as f:
            data = json.load(f)
    except IOError:
        print(f"Could Not Open File {data_dir+filename}", file=sys.stderr)
        sys.exit(1)

    n = np.asarray(data[key])
    if transform is not None:
        n = weight_transform(n, transform)
    return n


def process_weight_selection(selection, dataset_directory):
    if selection == 'fasta':
        return selection
    elif selection == 'None':
        return None
    if type(selection) is dict:
        return load_weight_file(dataset_directory,
                               selection['filename'],
                               selection['key'],
                                transform=None if 'transform' not in selection else selection['transform'])

    print(f"Sequence Weights must be dict, or str with values 'fasta' or 'None'."
          f" Received {selection} as input", file=sys.stderr)
    sys.exit(1)


def weight_transform(inp, transform):
    try:
        if type(inp) is np.ndarray:
            if 'exp' in transform:
                return np.exp(inp)
            elif "log" in transform:
                return np.log(inp)
            else:
                return inp
        elif type(inp) is torch.Tensor:
            if 'exp' in transform:
                return inp.exp()
            elif 'log' in transform:
                return inp.log()
            else:
                return inp
    except NotImplementedError:
        print(f"Transform {transform} for input type {type(inp)}")


def load_run(runfile):
    # yaml file
    if type(runfile) is str:
        try:
            with open(runfile, "r") as f:
                run_data = yaml.safe_load(f)
        except IOError:
            print(f"Runfile {runfile} not found or empty! Please check!", file=sys.stderr)
            sys.exit(1)
    # dictionary
    elif type(runfile) is dict:
        run_data = runfile
    else:
        print("Unsupported Format for run configuration. Must be filename of json config file or dictionary")
        exit(1)

    config = {}
    config.update(run_data['general'])
    config.update(run_data['dataset'])
    config.update(run_data['model'])

    def process_weight_selection(selection):
        if selection in ['fasta', 'None']:
            return selection

        if type(selection) is dict:
            return load_weight_file(config['data_directory'],
                                   selection['filename'],
                                   selection['key'])

        print(f"Sequence Weights must be dict, or str with values 'fasta' or 'None'."
              f" Received {selection} as input", file=sys.stderr)
        sys.exit(1)

    config['sequence_weights'] = process_weight_selection(config["sequence_weights_selection"])
    config['sampling_weights'] = process_weight_selection(config["sampling_weights_selection"])

    # Edit config for dataset specific hyperparameters
    if type(config['fasta_file']) is not list:
        config['fasta_file'] = [config['fasta_file']]

    config['seed'] = config['seed_selection']
    if config['seed'] in ['any', 'None']:
        config['seed'] = int(np.random.randint(0, 10000, 1)[0])

    if "CRBM" in config["model_type"]:
        # added since json files don't support tuples
        for key, val in config["convolution_topology"].items():
            for attribute in ["kernel", "dilation", "padding", "stride", "output_padding"]:
                val[f"{attribute}"] = tuple(val[f"{attribute}"])

    return config





def configure_optimizer(optimizer_str):
    try:
        return {"SGD": SGD, "AdamW": AdamW, "Adadelta": Adadelta, "Adagrad": Adagrad, "Adam": Adam}[optimizer_str]
    except KeyError:
        print(f"Optimizer {optimizer_str} is not supported")
        exit(1)


# CRBM only functions
def suggest_conv_size(input_shape, padding_max=3, dilation_max=4, stride_max=5):
    v_num, q = input_shape
    print(f"Finding Whole Convolutions for Input with {v_num} inputs:")
    # kernel size
    for i in range(1, v_num+1):
        kernel = [i, q]
        # padding
        for p in range(padding_max+1):
            padding = [p, 0]
            # dilation
            for d in range(1, dilation_max+1):
                dilation = [d, 1]
                # stride
                for s in range(1, stride_max+1):
                    stride = [s, 1]
                    # Convolution Output size
                    convx_num = int(math.floor((v_num + padding[0] * 2 - dilation[0] * (kernel[0] - 1) - 1) / stride[0] + 1))
                    convy_num = int(math.floor((q + padding[1] * 2 - dilation[1] * (kernel[1] - 1) - 1) / stride[1] + 1))
                    # Reconstruction Size
                    recon_x = (convx_num - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kernel[0] - 1) + 1
                    recon_y = (convy_num - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kernel[1] - 1) + 1
                    if recon_x == v_num:
                        print(f"Whole Convolution Found: Kernel: {kernel[0]}, Stride: {stride[0]}, Dilation: {dilation[0]}, Padding: {padding[0]}")
    return


# used by crbm to initialize weight sizes
def pool1d_dim(input_shape, pool_topology):
    [batch_size, h_number, convolutions] = input_shape
    stride = pool_topology["stride"]
    padding = pool_topology["padding"]
    kernel = pool_topology["kernel"]
    # dilation = pool_topology["dilation"]
    dilation = 1

    pool_out_num = int(math.floor((convolutions + padding * 2 - dilation * (kernel - 1) - 1) / stride + 1))

    # Copied from https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html
    un_pool_out_size = (pool_out_num - 1) * stride - 2 * padding + 1 * (kernel - 1) + 1

    # Pad for the unsampled convolutions (depends on stride, and tensor size)
    output_padding = convolutions - un_pool_out_size

    if output_padding != 0:
        print("Cannot create full reconstruction, please choose different pool topology")

    pool_out_size = (batch_size, h_number, pool_out_num)
    reconstruction_size = (batch_size, h_number, un_pool_out_size)

    return {"pool_size": pool_out_size, "reconstruction_shape": reconstruction_size}


# used by crbm to initialize weight sizes
def conv2d_dim(input_shape, conv_topology):
    [batch_size, input_channels, v_num, q] = input_shape
    if type(v_num) is tuple and q == 1:
        v_num, q = v_num # 2d visible
    stride = conv_topology["stride"]
    padding = conv_topology["padding"]
    kernel = conv_topology["kernel"]
    dilation = conv_topology["dilation"]
    h_num = conv_topology["number"]

    # Copied From https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    convx_num = int(math.floor((v_num + padding[0]*2 - dilation[0] * (kernel[0]-1) - 1)/stride[0] + 1))
    convy_num = int(math.floor((q + padding[1] * 2 - dilation[1] * (kernel[1]-1) - 1)/stride[1] + 1))  # most configurations will set this to 1

    # Copied from https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html
    recon_x = (convx_num - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kernel[0] - 1) + 1
    recon_y = (convy_num - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kernel[1] - 1) + 1

    # Pad for the unsampled visible units (depends on stride, and tensor size)
    output_padding = (v_num - recon_x, q - recon_y)

    # Size of Convolution Filters
    weight_size = (h_num, input_channels, kernel[0], kernel[1])

    # Size of Hidden unit Inputs h_uk
    conv_output_size = (batch_size, h_num, convx_num, convy_num)

    return {"weight_shape": weight_size, "conv_shape": conv_output_size, "output_padding": output_padding}


# used by crbm to initialize weight sizes
def conv1d_dim(input_shape, conv_topology):
    [batch_size, input_channels, v_num] = input_shape

    stride = conv_topology["stride"]
    padding = conv_topology["padding"]
    kernel = conv_topology["kernel"]
    dilation = conv_topology["dilation"]
    h_num = conv_topology["number"]

    # Copied From https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    convx_num = int(math.floor((v_num + padding*2 - dilation * (kernel-1) - 1)/stride + 1))

    # Copied from https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html
    recon_x = (convx_num - 1) * stride - 2 * padding + dilation * (kernel - 1) + 1

    # Pad for the unsampled visible units (depends on stride, and tensor size)
    output_padding = v_num - recon_x

    # Size of Convolution Filters
    weight_size = (h_num, input_channels, kernel)

    # Size of Hidden unit Inputs h_uk
    conv_output_size = (batch_size, h_num, convx_num)

    return {"weight_shape": weight_size, "conv_shape": conv_output_size, "output_padding": output_padding}


##### Other model utility functions

# Get Model from checkpoint File with specified version and directory
# def get_checkpoint(version, dir=""):
#     checkpoint_dir = dir + "/version_" + str(version) + "/checkpoints/"
#
#     for file in os.listdir(checkpoint_dir):
#         if file.endswith(".ckpt"):
#             checkpoint_file = os.path.join(checkpoint_dir, file)
#     return checkpoint_file

def get_beta_and_W(model, hidden_key=None, include_gaps=False, separate_signs=False):
    name = model._get_name()
    if "CRBM" in name or "crbm" in name:
        if hidden_key is None:
            print("Must specify hidden key in get_beta_and_W for crbm")
            exit(-1)
        else:
            W = model.get_param(hidden_key + "_W").squeeze(1)
            if separate_signs:
                Wpos = np.maximum(W, 0)
                Wneg = np.minimum(W, 0)
                if include_gaps:
                    return np.sqrt((Wpos ** 2).sum(-1).sum(-1)), Wpos, np.sqrt((Wneg ** 2).sum(-1).sum(-1)), Wneg
                else:
                    return np.sqrt((Wpos[:, :, :-1] ** 2).sum(-1).sum(-1)), Wpos, np.sqrt((Wneg[:, :, :-1] ** 2).sum(-1).sum(-1)), Wneg
            else:
                if include_gaps:
                    return np.sqrt((W ** 2).sum(-1).sum(-1)), W
                else:
                    return np.sqrt((W[:, :, :-1] ** 2).sum(-1).sum(-1)), W
    elif "RBM" in name:
        W = model.get_param("W")
        if include_gaps:
            return np.sqrt((W ** 2).sum(-1).sum(-1)), W
        else:
            return np.sqrt((W[:, :, :-1] ** 2).sum(-1).sum(-1)), W


def all_weights(model, name=None, rows=5, order_weights=True):
    model_name = model._get_name()
    if name is None:
        name = model._get_name()

    if "CRBM" in model_name or "crbm" in model_name:
        if "cluster" in model_name:
            for cluster_indx in range(model.clusters):
                for key in model.hidden_convolution_keys[cluster_indx]:
                    wdim = model.convolution_topology[cluster_indx][key]["weight_dims"]
                    kernelx = wdim[2]
                    if kernelx <= 10:
                        ncols = 2
                    else:
                        ncols = 1
                    conv_weights(model, key, f"{name}_{key}" + key, rows, ncols, 7, 5, order_weights=order_weights)
        else:
            for key in model.hidden_convolution_keys:
                wdim = model.convolution_topology[key]["weight_dims"]
                kernelx = wdim[2]
                if kernelx <= 10:
                    ncols = 2
                else:
                    ncols = 1
                conv_weights(model, key, name + "_" + key, rows, ncols, 7, 5, order_weights=order_weights)
    elif "RBM" in model_name:
        beta, W = get_beta_and_W(model)
        if order_weights:
            order = np.argsort(beta)[::-1]
        else:
            order = np.arange(0, beta.shape[0], 1)
        wdim = W.shape[1]
        if wdim <= 20:
            ncols = 2
        else:
            ncols = 1
        fig = sequence_logo_all(W[order], data_type="weights", name=name + '.pdf', nrows=rows, ncols=ncols, figsize=(7, 5), ticks_every=10, ticks_labels_size=10, title_size=12, dpi=200, molecule=model.molecule)

    plt.close() # close all open figures


def conv_weights(crbm, hidden_key, name, rows, columns, h, w, order_weights=True):
    beta, W = get_beta_and_W(crbm, hidden_key)
    if order_weights:
        order = np.argsort(beta)[::-1]
    else:
        order = np.arange(0, beta.shape[0], 1)
    fig = sequence_logo_all(W[order], data_type="weights", name=name + '.pdf', nrows=rows, ncols=columns, figsize=(h,w) ,ticks_every=5,ticks_labels_size=10,title_size=12, dpi=200, molecule=crbm.molecule)
