import time
import pandas as pd
import math
import json
import numpy as np
import sys
from pytorch_lightning import LightningModule, Trainer
# from pytorch_lightning.profiler import SimpleProfiler, PyTorchProfiler
from pytorch_lightning.loggers import TensorBoardLogger


import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, AdamW, Adagrad, Adadelta  # Supported Optimizers
from multiprocessing import cpu_count # Just to set the worker number
from torch.autograd import Variable

from rbm_torch.utils.utils import Categorical, conv2d_dim
from torch.utils.data import WeightedRandomSampler
from rbm_torch.models.base import Base_drelu

class pool_crbm_relu(Base_drelu):
    def __init__(self, config, debug=False, precision="double", meminfo=False):
        super().__init__(config, debug=debug, precision=precision)

        self.mc_moves = config['mc_moves']  # Number of MC samples to take to update hidden and visible configurations

        # sample types control whether gibbs sampling, pcd, from the data or parallel tempering from random configs are used
        # Switches How the training of the RBM is performed
        self.sample_type = config['sample_type']

        assert self.sample_type in ['gibbs', 'pt', 'pcd']

        # Regularization Options #
        ###########################################
        self.l1_2 = config['l1_2']  # regularization on weights, ex. 0.25
        self.lf = config['lf']  # regularization on fields, ex. 0.001
        self.ld = config['ld']  # regularization on distance b/t aligned weights
        self.lgap = config['lgap'] # regularization on gaps
        self.lbs = config['lbs']  # regularization to promote using both sides of the weights
        self.lcorr = config['lcorr']  # regularization on correlation of weights
        ###########################################
        self.lkd = config['lkd']

        self.convolution_topology = config["convolution_topology"]

        if type(self.v_num) is int:
            # Normal dist. times this value sets initial weight values
            self.weight_initial_amplitude = np.sqrt(0.001 / self.v_num)
            self.register_parameter("fields", nn.Parameter(torch.zeros((self.v_num, self.q), device=self.device)))
            self.register_parameter("fields0", nn.Parameter(torch.zeros((self.v_num, self.q), device=self.device)))
        elif type(self.v_num) is tuple:  # Normal dist. times this value sets initial weight values
            self.weight_initial_amplitude = np.sqrt(0.01 / math.prod(list(self.v_num)))
            self.register_parameter("fields", nn.Parameter(torch.zeros((*self.v_num, self.q), device=self.device)))
            self.register_parameter("fields0", nn.Parameter(torch.zeros((*self.v_num, self.q), device=self.device)))

        self.hidden_convolution_keys = list(self.convolution_topology.keys())

        self.pools = []
        self.unpools = []

        for key in self.hidden_convolution_keys:
            # Set information about the convolutions that will be useful
            dims = conv2d_dim([self.batch_size, 1, self.v_num, self.q], self.convolution_topology[key])
            self.convolution_topology[key]["weight_dims"] = dims["weight_shape"]
            self.convolution_topology[key]["convolution_dims"] = dims["conv_shape"]
            self.convolution_topology[key]["output_padding"] = dims["output_padding"]

            # deal with pool and unpool initialization
            pool_input_size = dims["conv_shape"][:-1]
            pool_kernel = pool_input_size[2]

            self.pools.append(nn.MaxPool1d(pool_kernel, stride=1, return_indices=True, padding=0))
            self.unpools.append(nn.MaxUnpool1d(pool_kernel, stride=1, padding=0))

            # Convolution Weights
            self.register_parameter(f"{key}_W", nn.Parameter(self.weight_initial_amplitude * torch.randn(self.convolution_topology[key]["weight_dims"], device=self.device)))
            # hidden layer parameters
            self.register_parameter(f"{key}_theta", nn.Parameter(torch.zeros(self.convolution_topology[key]["number"], device=self.device)))
            self.register_parameter(f"{key}_gamma", nn.Parameter(torch.ones(self.convolution_topology[key]["number"], device=self.device)))
            # Used in PT Sampling / AIS
            self.register_parameter(f"{key}_0theta", nn.Parameter(torch.zeros(self.convolution_topology[key]["number"], device=self.device), requires_grad=False))
            self.register_parameter(f"{key}_0gamma", nn.Parameter(torch.ones(self.convolution_topology[key]["number"], device=self.device), requires_grad=False))

        self.ind_temp_schedule = self.init_temp_schedule(config["ind_temp"])
        self.seq_temp_schedule = self.init_temp_schedule(config["seq_temp"])


        # Saves Our hyperparameter options into the checkpoint file generated for Each Run of the Model
        # i.e. Simplifies loading a model that has already been run
        self.save_hyperparameters()

        # Initialize AIS/PT members
        self.data_sampler = data_sampler(self, device=self.device)
        self.log_Z_AIS = None
        self.log_Z_AIS_std = None

    @property
    def h_layer_num(self):
        return len(self.hidden_convolution_keys)

    ## Used in our Loss Function
    def free_energy(self, v):
        return self.energy_v(v) - self.logpartition_h(self.compute_output_v(v))

    def free_energy_ind(self, v):
        h_ind = self.logpartition_h_ind(self.compute_output_v(v))
        return (self.energy_v(v)/h_ind.shape[1]).unsqueeze(1) - h_ind

    ## Not used but may be useful
    def free_energy_h(self, h):
        return self.energy_h(h) - self.logpartition_v(self.compute_output_h(h))

    ## Total Energy of a given visible and hidden configuration
    def energy(self, v, h, remove_init=False, hidden_sub_index=-1):
        return self.energy_v(v, remove_init=remove_init) + self.energy_h(h, sub_index=hidden_sub_index, remove_init=remove_init) - self.bidirectional_weight_term(v, h, hidden_sub_index=hidden_sub_index)

    def energy_PT(self, v, h, N_PT, remove_init=False):
        E = torch.zeros((N_PT, v.shape[1]), device=self.device)
        for i in range(N_PT):
            E[i] = self.energy_v(v[i], remove_init=remove_init) + self.energy_h(h, sub_index=i, remove_init=remove_init) - self.bidirectional_weight_term(v[i], h, hidden_sub_index=i)
        return E

    def bidirectional_weight_term(self, v, h, hidden_sub_index=-1):
        conv = self.compute_output_v(v)
        E = torch.zeros((len(self.hidden_convolution_keys), conv[0].shape[0]), device=self.device)
        for iid, i in enumerate(self.hidden_convolution_keys):
            if hidden_sub_index != -1:
                h_uk = h[iid][hidden_sub_index]
            else:
                h_uk = h[iid]
            E[iid] = h_uk.mul(conv[iid]).sum(1)

        if E.shape[0] > 1:
            return E.sum(0)
        else:
            return E.squeeze(0)

    ############################################################# Individual Layer Functions
    def transform_v(self, I):
        return F.one_hot(torch.argmax(I + getattr(self, "fields").unsqueeze(0), dim=-1), self.q)

    def transform_h(self, I):
        output = []
        for kid, key in enumerate(self.hidden_convolution_keys):
            gamma = (getattr(self, f'{key}_gamma')).unsqueeze(0).unsqueeze(2)
            theta = (getattr(self, f'{key}_theta')).unsqueeze(0).unsqueeze(2)
            output.append(torch.maximum(I - theta, torch.tensor(0, device=self.device)) / gamma)
        return output

    ## Computes -g(si) term of potential
    def energy_v(self, config, remove_init=False):
        # config is a one hot vector
        v = config.type(torch.get_default_dtype())
        E = torch.zeros(config.shape[0], device=self.device)
        for i in range(self.q):
            if remove_init:
                E -= v[:, :, i].dot(getattr(self, "fields")[:, i] - getattr(self, "fields0")[:, i])
            else:
                E -= v[:, :, i].matmul(getattr(self, "fields")[:, i])

        return E

    ## Computes U(h) term of potential
    def energy_h(self, config, remove_init=False, sub_index=-1):
        # config is list of h_uks
        if sub_index != -1:
            E = torch.zeros((len(self.hidden_convolution_keys), config[0].shape[1]), device=self.device)
        else:
            E = torch.zeros((len(self.hidden_convolution_keys), config[0].shape[0]), device=self.device)

        for iid, i in enumerate(self.hidden_convolution_keys):
            if remove_init:
                gamma = getattr(self, f'{i}_gamma').sub(getattr(self, f'{i}_0gamma')).unsqueeze(0)
                theta = getattr(self, f'{i}_theta').sub(getattr(self, f'{i}_0theta')).unsqueeze(0)
            else:
                gamma = getattr(self, f'{i}_gamma').unsqueeze(0)
                theta = getattr(self, f'{i}_theta').unsqueeze(0)

            if sub_index != -1:
                con = config[iid][sub_index]
            else:
                con = config[iid]

            E[iid] = ((con.square() * gamma) / 2 + (con * theta)).sum(1)

        if E.shape[0] > 1:
            return E.sum(0)
        else:
            return E.squeeze(0)

    ## Random Config of Visible Potts States
    def random_init_config_v(self, custom_size=False, zeros=False):
        if custom_size:
            size = (*custom_size, self.v_num, self.q)
        else:
            size = (self.batch_size, self.v_num, self.q)

        if zeros:
            return torch.zeros(size, device=self.device)
        else:
            return self.sample_from_inputs_v(torch.zeros(size, device=self.device).flatten(0, -3), beta=0).reshape(size)

    ## Random Config of Hidden dReLU States
    def random_init_config_h(self, zeros=False, custom_size=False):
        config = []
        for iid, i in enumerate(self.hidden_convolution_keys):
            batch, h_num, convx_num, convy_num = self.convolution_topology[i]["convolution_dims"]

            if custom_size:
                size = (*custom_size, h_num)
            else:
                size = (self.batch_size, h_num)

            if zeros:
                config.append(torch.zeros(size, device=self.device))
            else:
                config.append(torch.randn(size, device=self.device))

        return config

    def clone_h(self, hidden_config, reduce_dims=[], expand_dims=[], sub_index=-1):
        new_config = []
        for hc in hidden_config:
            if sub_index != -1:
                new_h = hc[sub_index].clone()
            else:
                new_h = hc.clone()
            for dim in reduce_dims:
                new_h = new_h.squeeze(dim)
            for dim in expand_dims:
                new_h = new_h.unsqueeze(dim)
            new_config.append(new_h)
        return new_config

    ## Marginal over hidden units
    def logpartition_h(self, inputs, beta=1):
        # Input is list of matrices I_uk
        marginal = torch.zeros((len(self.hidden_convolution_keys), inputs[0].shape[0]), device=self.device)
        for iid, i in enumerate(self.hidden_convolution_keys):
            marginal[iid] = self.cgf_from_inputs_h(inputs[iid], hidden_key=i).sum(-1)
        return marginal.sum(0)

    def logpartition_h_ind(self, inputs, beta=1):
        # Input is list of matrices I_uk
        ys = []
        for iid, i in enumerate(self.hidden_convolution_keys):
            y = self.cgf_from_inputs_h(inputs[iid], hidden_key=i)
            ys.append(y)
        return torch.cat(ys, dim=1)

    ## Marginal over visible units
    def logpartition_v(self, inputs, beta=1):
        if beta == 1:
            return torch.logsumexp(getattr(self, "fields")[None, :, :] + inputs, 2).sum(1)
        else:
            return torch.logsumexp((beta * getattr(self, "fields") + (1 - beta) * getattr(self, "fields0"))[None, :] + beta * inputs, 2).sum(1)

    ## Mean of hidden layer specified by hidden_key
    def mean_h(self, psi, hidden_key=None, beta=1):
        if hidden_key is None:
            hidden_key = self.hidden_convolution_keys
        elif type(hidden_key) is str:
            hidden_key = [hidden_key]

        means = []
        for kid, key in enumerate(self.hidden_convolution_keys):
            if beta == 1:
                gamma = (getattr(self, f'{key}_gamma')).unsqueeze(0)
                theta = (getattr(self, f'{key}_theta')).unsqueeze(0)
            else:
                theta = (beta * getattr(self, f'{key}_theta') + (1 - beta) * getattr(self, f'{key}_0theta')).unsqueeze(0)
                gamma = (beta * getattr(self, f'{key}_gamma') + (1 - beta) * getattr(self, f'{key}_0gamma')).unsqueeze(0)
                psi[kid] *= beta

            sqrt_gamma = torch.sqrt(gamma)
            means.append((psi[kid] - theta) / gamma + 1. / self.erf_times_gauss((-psi[kid] + theta) / sqrt_gamma) / sqrt_gamma)

        if len(hidden_key) == 1:
            return means[0]

        return means


    ## Compute Input for Hidden Layer from Visible Potts, Uses one hot vector
    def compute_output_v(self, X):  # X is the one hot vector
        outputs = []
        self.max_inds = []
        self.min_inds = []
        for iid, i in enumerate(self.hidden_convolution_keys):
            weights = getattr(self, f"{i}_W")
            conv = F.conv2d(X.unsqueeze(1).type(torch.get_default_dtype()), weights, stride=self.convolution_topology[i]["stride"],
                            padding=self.convolution_topology[i]["padding"],
                            dilation=self.convolution_topology[i]["dilation"]).squeeze(3)

            max_pool, max_inds = self.pools[iid](conv.abs())

            flat_conv = conv.flatten(start_dim=2)
            max_conv_values = flat_conv.gather(2, index=max_inds.flatten(start_dim=2)).view_as(max_inds)
            max_pool *= max_conv_values/max_conv_values.abs()

            self.max_inds.append(max_inds)
            out = max_pool.flatten(start_dim=2)
            out.squeeze_(2)

            if self.dr > 0.:
                out = F.dropout(out, p=self.dr, training=self.training)

            outputs.append(out)
            if True in torch.isnan(out):
                print("hi")

        return outputs

    # Compute Input for Visible Layer from Hidden dReLU
    def compute_output_h(self, h):  # from h_uk (B, hidden_num)
        outputs = []
        for iid, i in enumerate(self.hidden_convolution_keys):
            reconst = self.unpools[iid](h[iid].view_as(self.max_inds[iid]), self.max_inds[iid])

            if reconst.ndim == 3:
                reconst.unsqueeze_(3)

            outputs.append(F.conv_transpose2d(reconst, getattr(self, f"{i}_W"),
                                              stride=self.convolution_topology[i]["stride"],
                                              padding=self.convolution_topology[i]["padding"],
                                              dilation=self.convolution_topology[i]["dilation"],
                                              output_padding=self.convolution_topology[i]["output_padding"]).squeeze(1))

        if len(outputs) > 1:
            return torch.sum(torch.stack(outputs), 0)

        return outputs[0]

    # Gibbs Sampling of Potts Visbile Layer
    def sample_from_inputs_v(self, psi, beta=1):  # Psi ohe (Batch_size, v_num, q)   fields (self.v_num, self.q)
        if beta == 1:
            cum_probas = psi + getattr(self, "fields").unsqueeze(0)
        else:
            cum_probas = beta * psi + beta * getattr(self, "fields").unsqueeze(0) + (1 - beta) * getattr(self, "fields0").unsqueeze(0)

        maxi, max_indices = cum_probas.max(-1)
        maxi.unsqueeze_(2)
        cum_probas -= maxi
        cum_probas.exp_()
        cum_probas[cum_probas > 1e9] = 1e9  # For numerical stability.

        dist = torch.distributions.categorical.Categorical(probs=cum_probas)
        return F.one_hot(dist.sample(), self.q)

    def sample_from_inputs_h(self, psi, nancheck=False, beta=1):  # psi is a list of hidden [input]
        """Gibbs Sampling of ReLU hidden layer"""
        h_uks = []
        for iid, i in enumerate(self.hidden_convolution_keys):
            if beta == 1:
                gamma = getattr(self, f'{i}_gamma').unsqueeze(0)
                theta = getattr(self, f'{i}_theta').unsqueeze(0)
            else:
                theta = (beta * getattr(self, f'{i}_theta') + (1 - beta) * getattr(self, f'{i}_0theta')).unsqueeze(0)
                gamma = (beta * getattr(self, f'{i}_gamma') + (1 - beta) * getattr(self, f'{i}_0gamma')).unsqueeze(0)
                psi[iid] *= beta

            if nancheck:
                nans = torch.isnan(psi[iid])
                if nans.max():
                    nan_unit = torch.nonzero(nans.max(0))[0]
                    print('NAN IN INPUT')
                    print('Hidden units', nan_unit)

            sqrt_gamma = torch.sqrt(gamma)
            I_plus = (-psi[iid] + theta) / sqrt_gamma
            rmin = torch.erf(I_plus / self.sqrt2)
            rmax = 1
            tmp = (rmax - rmin < 1e-14)
            h = (self.sqrt2 * torch.erfinv(rmin + (rmax - rmin) * torch.rand(psi[iid].shape, device=self.device)) - I_plus) / sqrt_gamma
            h = torch.clamp(h, min=0)  # Due to numerical error of erfinv, erf,  erfinv(rmin) is not exactly I_plus/sqrt(2).
            h[torch.isinf(h) | torch.isnan(h) | tmp] = 0
            h_uks.append(h)

        return h_uks

    ###################################################### Sampling Functions
    # Samples hidden from visible and vice versa, returns newly sampled hidden and visible
    def markov_step(self, v, beta=1):
        # Gibbs Sampler
        h = self.sample_from_inputs_h(self.compute_output_v(v), beta=beta)
        return self.sample_from_inputs_v(self.compute_output_h(h), beta=beta), h

    ######################################################### Pytorch Lightning Functions
    # def on_after_backward(self):
        # torch.nn.utils.clip_grad_norm_(self.parameters(), 10000, norm_type=2.0, error_if_nonfinite=True)

    # Clamps hidden potential values to acceptable range
    def on_before_zero_grad(self, optimizer):
        with torch.no_grad():
            for key in self.hidden_convolution_keys:
                getattr(self, f"{key}_gamma").data.clamp_(min=0.05)
                getattr(self, f"{key}_theta").data.clamp_(min=0.0)
                getattr(self, f"{key}_W").data.clamp_(-1.0, 1.0)

    ## Calls Corresponding Training Function
    def training_step(self, batch, batch_idx):
        # All other functions use self.W for the weights
        if self.sample_type == "gibbs":
            return self.training_step_CD_free_energy(batch, batch_idx)
        elif self.sample_type == "pt":
            return self.training_step_PT_free_energy(batch, batch_idx)
        elif self.sample_type == "pcd":
            return self.training_step_PCD_free_energy(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        inds, seqs, one_hot, seq_weights = batch

        free_energy = self.free_energy(one_hot)
        # free_energy_avg = (free_energy * seq_weights).sum() / seq_weights.sum()

        batch_out = {
            "val_free_energy": free_energy.mean().detach()
        }

        # logging on step, for whatever reason allocates 512 bytes on gpu after every epoch.
        self.log("ptl/val_free_energy", batch_out["val_free_energy"], on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=one_hot.shape[0])
        self.val_data_logs.append(batch_out)
        return

    def regularization_terms(self, distance_threshold=0.25):
        freg = self.lf / (2 * self.v_num * self.q) * getattr(self, "fields").square().sum((0, 1))
        wreg = torch.zeros((1,), device=self.device)
        dreg = torch.zeros((1,), device=self.device)  # discourages weights that are alike

        bs_loss = torch.zeros((1,), device=self.device)  # encourages weights to use both positive and negative contributions
        gap_loss = torch.zeros((1,), device=self.device)  # discourages high values for gaps

        for iid, i in enumerate(self.hidden_convolution_keys):
            W_shape = self.convolution_topology[i]["weight_dims"]  # (h_num,  input_channels, kernel0, kernel1)
            W = getattr(self, f"{i}_W")

            x = torch.sum(W.abs(), (3, 2, 1)).square()
            wreg += x.sum() * self.l1_2 / (W_shape[0] * W_shape[1] * W_shape[2] * W_shape[3])
            gap_loss += self.lgap * W[:, :, :, -1].abs().mean()

            Wpos = torch.clamp(W, min=0.)

            ws = Wpos.squeeze(1)

            with torch.no_grad():
                # compute positional differences for all pairs of weights
                pdiff = (ws.unsqueeze(0).unsqueeze(2) - ws.unsqueeze(1).unsqueeze(3)).sum(4)

                # concatenate it to itself to make full diagonals
                wdm = torch.concat([pdiff, pdiff.clone()], dim=2)

                # get stride to get matrix of all diagonals on separate row
                si, sj, v2, v = wdm.size()
                i_s, j_s, v2_s, v_s = wdm.stride()
                wdm_s = torch.as_strided(wdm, (si, sj, v, v), (i_s, j_s, v2_s, v2_s + 1))

                # get the best alignment position
                best_align = W_shape[2] - torch.argmin(wdm_s.abs().sum(3), -1)

                # get indices for all pairs of i <= j
                bat_x, bat_y = torch.triu_indices(si, sj, 1)

                # get their alignments
                bas = best_align[bat_x, bat_y]

                # create shifted weights
                vt_ind = torch.arange(len(bat_x), device=self.device)[:, None].expand(-1, v)
                v_ind = torch.arange(v, device=self.device)[None, :].expand(len(bat_y), -1)
                rolled_j = ws[bat_y][vt_ind, (v_ind + bas[:, None]) % v]

            # norms of all weights
            w_norms = torch.linalg.norm(ws, dim=(2, 1))

            # inner prod of weights i and shifted weights j
            inner_prod = torch.tensordot(ws[bat_x], rolled_j, dims=([2, 1], [2, 1]))[0]

            # angles between aligned weights
            angles = inner_prod/(w_norms[bat_x] * w_norms[bat_y] + 1e-6)

            # threshold
            angles = angles[angles > distance_threshold]

            dreg += angles.mean()

        dreg *= self.ld

        # Passed to training logger
        reg_dict = {
            "field_reg": freg.detach(),
            "weight_reg": wreg.detach(),
            "distance_reg": dreg.detach(),
            "gap_reg": gap_loss.detach(),
            "both_side_reg": bs_loss.detach()
        }

        return freg, wreg, dreg, bs_loss, gap_loss, reg_dict

    def training_step_PT_free_energy(self, batch, batch_idx):
        inds, seqs, one_hot, seq_weights = batch

        V_neg_oh, h_neg, V_pos_oh, h_pos = self(one_hot)

        # Calculate CD loss
        F_v = (self.free_energy(V_pos_oh) * seq_weights).sum() / seq_weights.sum()  # free energy of training data
        F_vp = (self.free_energy(V_neg_oh) * seq_weights).sum() / seq_weights.sum()  # free energy of gibbs sampled visible states
        cd_loss = F_v - F_vp

        # Regularization Terms
        freg, wreg, dreg, bs_loss, gap_loss, reg_dict = self.regularization_terms()

        # Calc loss
        loss = cd_loss + freg + wreg + dreg + bs_loss + gap_loss

        logs = {"loss": loss,
                "free_energy_diff": cd_loss.detach(),
                "train_free_energy": F_v.detach(),
                **reg_dict
                }

        self.log("ptl/train_free_energy", logs["train_free_energy"], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("ptl/train_loss", loss.detach(), on_step=True, on_epoch=True, prog_bar=True, logger=True)

        self.training_data_logs.append(logs)
        return logs["loss"]

    def training_step_CD_free_energy(self, batch, batch_idx):
        inds, seqs, one_hot, seq_weights = batch

        V_neg_oh, h_neg, V_pos_oh, h_pos = self(one_hot)
        F_v = (self.free_energy(V_pos_oh) * seq_weights / seq_weights.sum())  # free energy of training data
        F_vp = (self.free_energy(V_neg_oh) * seq_weights / seq_weights.sum()) # free energy of gibbs sampled visible states
        cd_loss = (F_v - F_vp).mean()

        free_energy_log = {
            "free_energy_pos": F_v.sum().detach(),
            "free_energy_neg": F_vp.sum().detach(),
            "free_energy_diff": cd_loss.sum().detach(),
            "cd_loss": cd_loss.detach(),
        }

        # Regularization Terms
        freg, wreg, dreg, bs_loss, gap_loss, reg_dict = self.regularization_terms()

        # Calculate Loss
        loss = cd_loss + freg + wreg + dreg + bs_loss + gap_loss

        logs = {"loss": loss,
                "train_free_energy": cd_loss.sum().detach(),
                **free_energy_log,
                **reg_dict
                }

        self.log("ptl/train_free_energy", logs["train_free_energy"], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("ptl/train_loss", loss.detach(), on_step=True, on_epoch=True, prog_bar=True, logger=True)

        self.training_data_logs.append(logs)
        return logs["loss"]

    def training_step_PCD_free_energy(self, batch, batch_idx):
        inds, seqs, one_hot, seq_weights = batch

        if self.current_epoch == 0 and batch_idx == 0:
            self.chain = torch.zeros((self.training_data.index.__len__(), *one_hot.shape[1:]), device=self.device)

        if self.current_epoch == 0:
            self.chain[inds] = one_hot.type(torch.get_default_dtype())

        V_oh_neg, h_neg = self.forward_PCD(inds)

        inputs_flat = torch.concat(self.compute_output_v(one_hot), 1)

        ### Normal Wayss
        F_v = self.free_energy(one_hot)
        F_vp = self.free_energy(V_oh_neg)

        kld_loss = self.kurtosis(F_v).clamp(min=0) * self.lkd
        cov_loss = (torch.cov(inputs_flat.T).triu().abs().sum())/((inputs_flat.shape[1]*(inputs_flat.shape[1]+1))/2) * self.lcorr

        cd_loss = (F_v - F_vp).mean()

        # Regularization Terms
        freg, wreg, dreg, bs_loss, gap_loss, reg_dict = self.regularization_terms()

        # Calculate Loss
        loss = cd_loss + freg + wreg + dreg + bs_loss + gap_loss + kld_loss + cov_loss

        if loss.isnan():
            print("okay")

        logs = {"loss": loss,
                "free_energy_diff": cd_loss.detach(),
                "free_energy_pos": F_v.mean().detach(),
                "free_energy_neg": F_vp.mean().detach(),
                "kl_loss": kld_loss.detach(),
                "cov_loss": cov_loss.detach(),
                #  "input_correlation_reg": input_loss.detach(),
                **reg_dict
                }

        self.log("ptl/free_energy_diff", logs["free_energy_diff"], on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=one_hot.shape[0])
        self.log("ptl/train_free_energy", logs["free_energy_pos"], on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=one_hot.shape[0])
        self.log("ptl/train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=one_hot.shape[0])

        self.training_data_logs.append(logs)
        return logs["loss"]

    def kurtosis(self, x):
        mean = torch.mean(x)
        diffs = (x - mean) + 1e-12
        std = torch.std(x) + 1e-12
        zscores = diffs/std
        skew = torch.mean(torch.pow(zscores, 3.0), 0)
        kurtosis = torch.mean(torch.pow(zscores, 4.0), 0)
        dm = diffs.max()
        # bimodality = (skew.square() + 1) / kurtosis
        return (std-2)  # + ((dm.abs()-5)/5).clamp(min=0) #(std-4).clamp(min=0) + skew.abs()*3 (kurtosis - 2).clamp(min=0)*3

    def forward_PCD(self, inds):
        """Gibbs sampling with Persistent Contrastive Divergence"""
        # Last sample that was saved to self.chain variable, initialized in training step
        fantasy_v = self.chain[inds]
        h_pos = self.sample_from_inputs_h(self.compute_output_v(fantasy_v))
        for _ in range(self.mc_moves - 1):
            fantasy_v, fantasy_h = self.markov_step(fantasy_v)

        V_neg, fantasy_h = self.markov_step(fantasy_v)
        h_neg = self.sample_from_inputs_h(self.compute_output_v(V_neg))
        # Save current sample for next iteration
        self.chain[inds] = V_neg.detach().type(torch.get_default_dtype())

        return V_neg, h_neg

    def forward_PT(self, V_pos_ohe, N_PT):
        # Initialize_PT is called before the forward function is called. Therefore, N_PT will be filled
        # Parallel Tempering
        n_chains = V_pos_ohe.shape[0]

        with torch.no_grad():
            fantasy_v = self.random_init_config_v(custom_size=(N_PT, n_chains))
            fantasy_h = self.random_init_config_h(custom_size=(N_PT, n_chains))
            fantasy_E = self.energy_PT(fantasy_v, fantasy_h, N_PT)

            for _ in range(self.mc_moves - 1):
                fantasy_v, fantasy_h, fantasy_E = self.data_sampler.markov_PT_and_exchange(fantasy_v, fantasy_h, fantasy_E, N_PT)
                self.data_sampler.update_betas(N_PT)

        fantasy_v, fantasy_h, fantasy_E = self.data_sampler.markov_PT_and_exchange(fantasy_v, fantasy_h, fantasy_E, N_PT)
        self.data_sampler.update_betas(N_PT)

        # V_neg, h_neg, V_pos, h_pos
        return fantasy_v[0], fantasy_h[0], V_pos_ohe, self.sample_from_inputs_h(self.compute_output_v(V_pos_ohe))

    def forward(self, V_pos_ohe):
        # Gibbs sampling
        fantasy_v, first_h = self.markov_step(V_pos_ohe)
        for _ in range(self.mc_moves - 1):
            fantasy_v, fantasy_h = self.markov_step(fantasy_v)

        return fantasy_v, fantasy_h, V_pos_ohe, first_h

    # X must be a pandas dataframe with the sequences in string format under the column 'sequence'
    # Returns the likelihood for each sequence in an array
    def predict(self, dataframe, column_key='sequence', individual_hiddens=False, device='cpu'):
        # Read in data
        reader = Categorical(dataframe, self.q, weights=None, max_length=self.v_num, alphabet=self.alphabet,
                             device=torch.device('cpu'), one_hot=True)
        # Put in Dataloader
        data_loader = torch.utils.data.DataLoader(
            reader,
            batch_size=self.batch_size,
            num_workers=self.worker_num,  # Set to 0 if debug = True
            pin_memory=False,
            shuffle=False
        )
        self.eval()
        with torch.no_grad():
            likelihood = []
            for i, batch in enumerate(data_loader):
                inds, seqs, one_hot, seq_weights = batch
                oh = one_hot.to(torch.device(self.device))
                likelihood += self.likelihood(one_hot, individual_hiddens=individual_hiddens).detach().cpu().tolist()

        return dataframe.sequence.tolist(), likelihood

    # Replace gap state contribution to 0 in all weights/biases
    def fill_gaps_in_parameters(self, fill=1e-6):
        with torch.no_grad():
            fields = getattr(self, "fields")
            fields[:, -1].fill_(fill)

            for iid, i in enumerate(self.hidden_convolution_keys):
                W = getattr(self, f"{i}_W")
                W[:, :, :, -1].fill_(fill)

    # X must be a pandas dataframe with the sequences in string format under the column 'sequence'
    # Returns the saliency map for all sequences in X
    def saliency_map(self, X):
        reader = Categorical(X, self.q, weights=None, max_length=self.v_num, molecule=self.alphabet, device=self.device, one_hot=True)
        data_loader = torch.utils.data.DataLoader(
            reader,
            batch_size=self.batch_size,
            num_workers=self.worker_num,  # Set to 0 if debug = True
            pin_memory=self.pin_mem,
            shuffle=False
        )
        saliency_maps = []
        self.eval()
        for i, batch in enumerate(data_loader):
            inds, seqs, one_hot, seq_weights = batch
            one_hot_v = Variable(one_hot.type(torch.get_default_dtype()), requires_grad=True)
            V_neg, h_neg, V_pos, h_pos = self(one_hot_v)
            weights = seq_weights
            F_v = (self.free_energy(V_pos) * weights).sum() / weights.sum()  # free energy of training data
            F_vp = (self.free_energy(V_neg) * weights).sum() / weights.sum()  # free energy of gibbs sampled visible states
            cd_loss = F_v - F_vp

            # Regularization Terms
            freg, wreg, dreg, bs_loss, gap_loss, reg_dict = self.regularization_terms()
            loss = cd_loss + freg + wreg + dreg + gap_loss + bs_loss
            loss.backward()

            saliency_maps.append(one_hot_v.grad.data.detach())

        return torch.cat(saliency_maps, dim=0)

    def likelihood(self, data, recompute_Z=False, individual_hiddens=False):
        if (self.log_Z_AIS is None) | recompute_Z:
            self.log_Z_AIS, self.log_Z_AIS_std = self.data_sampler.AIS()
        if individual_hiddens:
            return -self.free_energy_ind(data)
        else:
            return -self.free_energy(data) - self.log_Z_AIS

    def cgf_from_inputs_h(self, I, hidden_key, beta=1):
        B = I.shape[0]

        if hidden_key not in self.hidden_convolution_keys:
            print(f"Hidden Convolution Key {hidden_key} not found!")
            sys.exit(1)

        gamma = beta * getattr(self, f'{hidden_key}_gamma') + (1 - beta) * getattr(self, f'{hidden_key}_0gamma')
        theta = beta * getattr(self, f'{hidden_key}_theta') + (1 - beta) * getattr(self, f'{hidden_key}_0theta')

        sqrt_gamma = torch.sqrt(gamma).expand(B, -1)
        log_gamma = torch.log(gamma).expand(B, -1)
        out = self.log_erf_times_gauss((-I + theta) / sqrt_gamma) - 0.5 * log_gamma
        return out




class data_sampler:
    def __init__(self, pcrbm, device='cpu', **kwargs):

        self.model = pcrbm
        self.device = torch.device(device)

        self.record_acceptance = False
        self.record_swaps = False

        self.count_swaps = 0
        self.last_at_zero = None
        self.trip_duration = None

        for k, v in kwargs:
            setattr(self, k, v)

    def update_betas(self, N_PT, beta=1, update_betas_lr=0.1, update_betas_lr_decay=1):
        with torch.no_grad():
            stiffness = torch.maximum(1 - (self.mav_acceptance_rates / self.mav_acceptance_rates.mean()),
                                           torch.zeros_like(self.mav_acceptance_rates, device=self.device)) \
                                      + 1e-4 * torch.ones(N_PT - 1)
            diag = stiffness[0:-1] + stiffness[1:]
            offdiag_g = -stiffness[1:-1]
            offdiag_d = -stiffness[1:-1]
            M = torch.diag(offdiag_g, -1) + torch.diag(diag, 0) + torch.diag(offdiag_d, 1)

            B = torch.zeros(N_PT - 2, device=self.device)
            B[0] = stiffness[0] * beta
            self.betas[1:-1] = self.betas[1:-1] * (1 - update_betas_lr) + update_betas_lr * torch.linalg.solve(M, B)
            update_betas_lr *= update_betas_lr_decay

    def markov_PT_and_exchange(self, v, h, e, N_PT):
        for i, beta in zip(torch.arange(N_PT), self.betas):
            v[i], htmp = self.model.markov_step(v[i], beta=beta)
            for hid in range(self.model.h_layer_num):
                h[hid][i] = htmp[hid]
            e[i] = self.model.energy(v[i], h, hidden_sub_index=i)

        if self.record_swaps:
            particle_id = torch.arange(N_PT).unsqueeze(1).expand(N_PT, v.shape[1])

        betadiff = self.betas[1:] - self.betas[:-1]
        for i in np.arange(self.count_swaps % 2, N_PT - 1, 2):
            proba = torch.exp(betadiff[i] * e[i + 1] - e[i]).minimum(torch.ones_like(e[i]))
            swap = torch.rand(proba.shape[0], device=self.device) < proba
            if i > 0:
                v[i:i + 2, swap] = torch.flip(v[i - 1: i + 1], [0])[:, swap]
                for hid in range(self.model.h_layer_num):
                    h[hid][i:i + 2, swap] = torch.flip(h[hid][i - 1: i + 1], [0])[:, swap]

                e[i:i + 2, swap] = torch.flip(e[i - 1: i + 1], [0])[:, swap]
                if self.record_swaps:
                    particle_id[i:i + 2, swap] = torch.flip(particle_id[i - 1: i + 1], [0])[:, swap]
            else:
                v[i:i + 2, swap] = torch.flip(v[:i + 1], [0])[:, swap]
                for hid in range(self.model.h_layer_num):
                    h[hid][i:i + 2, swap] = torch.flip(h[hid][:i + 1], [0])[:, swap]
                e[i:i + 2, swap] = torch.flip(e[:i + 1], [0])[:, swap]
                if self.record_swaps:
                    particle_id[i:i + 2, swap] = torch.flip(particle_id[:i + 1], [0])[:, swap]

            if self.record_acceptance:
                self.acceptance_rates[i] = swap.type(torch.get_default_dtype()).mean()
                self.mav_acceptance_rates[i] = self.mavar_gamma * self.mav_acceptance_rates[i] + self.acceptance_rates[
                    i] * (1 - self.mavar_gamma)

        if self.record_swaps:
            self.particle_id.append(particle_id)

        self.count_swaps += 1
        return v, h, e

    def AIS(self, M=10, n_betas=10000, batches=None, verbose=0, beta_type='adaptive'):
        with torch.no_grad():
            if beta_type == 'linear':
                betas = torch.arange(n_betas, device=self.device) / torch.tensor(n_betas - 1, dtype=torch.float64,
                                                                                 device=self.device)
            elif beta_type == 'root':
                betas = torch.sqrt(torch.arange(n_betas, device=self.device)) / \
                        torch.tensor(n_betas - 1, dtype=torch.float64, device=self.device)
            elif beta_type == 'adaptive':
                Nthermalize = 200
                Nchains = 20
                N_PT = 11
                # adaptive_PT_lr = 0.05
                # adaptive_PT_decay = True
                # adaptive_PT_lr_decay = 10 ** (-1 / float(Nthermalize))
                if verbose:
                    t = time.time()
                    print('Learning betas...')
                self.gen_data(N_PT=N_PT, Nchains=Nchains, Lchains=1, Nthermalize=Nthermalize, update_betas=True)
                if verbose:
                    print('Elapsed time: %s, Acceptance rates: %s' % (time.time() - t, self.mav_acceptance_rates))
                betas = []
                sparse_betas = self.betas.flip(0)
                for i in range(N_PT - 1):
                    betas += list(
                        sparse_betas[i] + (sparse_betas[i + 1] - sparse_betas[i]) *
                        torch.arange(n_betas / (N_PT - 1),  device=self.device) /
                        (n_betas / (N_PT - 1) - 1))
                betas = torch.tensor(betas, device=self.device)
                n_betas = betas.shape[0]

            # Initialization.
            log_weights = torch.zeros(M, device=self.device)
            # config = self.gen_data(Nchains=M, Lchains=1, Nthermalize=0, beta=0)

            config = [self.model.sample_from_inputs_v(self.model.random_init_config_v(custom_size=(M,))),
                      self.model.sample_from_inputs_h(self.model.random_init_config_h(custom_size=(M,)))]

            log_Z_init = torch.zeros(1, device=self.device)
            log_Z_init += self.model.logpartition_h(self.model.random_init_config_h(custom_size=(1,), zeros=True), beta=0)
            log_Z_init += self.model.logpartition_v(self.model.random_init_config_v(custom_size=(1,), zeros=True), beta=0)

            if verbose:
                print(f'Initial evaluation: log(Z) = {log_Z_init.data}')

            for i in range(1, n_betas):
                if verbose:
                    if (i % 2000 == 0):
                        print(f'Iteration {i}, beta: {betas[i]}')
                        print('Current evaluation: log(Z)= %s +- %s' % (
                        (log_Z_init + log_weights).mean(), (log_Z_init + log_weights).std() / np.sqrt(M)))

                config[0], config[1] = self.model.markov_step(config[0], beta=betas[i])
                energy = self.model.energy(config[0], config[1])
                log_weights += -(betas[i] - betas[i - 1]) * energy

            log_Z_AIS = (log_Z_init + log_weights).mean()
            log_Z_AIS_std = (log_Z_init + log_weights).std() / np.sqrt(M)
            if verbose:
                print('Final evaluation: log(Z)= %s +- %s' % (log_Z_AIS, log_Z_AIS_std))
            return log_Z_AIS, log_Z_AIS_std

    def _gen_data(self, Nthermalize, Ndata, Nstep, N_PT=1, batches=1, reshape=True,
                   config_init=[], beta=1, record_replica=False, update_betas=False):

        if Ndata < 2:
            raise Exception("Ndata must be 2 or greater!")

        with torch.no_grad():
            if update_betas or len(self.betas) != N_PT:
                self.betas = torch.flip(torch.arange(N_PT, device=self.device) / (N_PT - 1) * beta, [0])

            self.acceptance_rates = torch.zeros(N_PT - 1, device=self.device)
            self.mav_acceptance_rates = torch.zeros(N_PT - 1, device=self.device)

            self.count_swaps = 0

            if self.record_swaps:
                self.particle_id = [torch.arange(N_PT, device=self.device)[:, None].repeat(batches, dim=1)]

            Ndata /= batches
            Ndata = int(Ndata)

            if config_init != []:
                config = config_init
            else:
                config = [self.model.random_init_config_v(custom_size=(N_PT, batches)),
                          self.model.random_init_config_h(custom_size=(N_PT, batches))]

            for _ in range(Nthermalize):
                energy = self.model.energy_PT(config[0], config[1], N_PT)
                config[0], config[1], energy = self.model.markov_PT_and_exchange(config[0], config[1], energy, N_PT)
                if update_betas:
                    self.update_betas(N_PT, beta=beta)

            # if record_replica:
            #     data = [config[0].clone().unsqueeze(0), self.model.clone_h(config[1], expand_dims=[0])]
            # else:
            #     data = [config[0][0].clone().unsqueeze(0), self.model.clone_h(config[1], expand_dims=[0], sub_index=0)]

            if record_replica:
                data_gen_v = self.model.random_init_config_v(custom_size=(Ndata, N_PT, batches), zeros=True)
                data_gen_h = self.model.random_init_config_h(custom_size=(Ndata, N_PT, batches), zeros=True)
                data_gen_v[0] = config[0].clone()

                clone = self.model.clone_h(config[1])
                for hid in range(self.model.h_layer_num):
                    data_gen_h[hid][0] = clone[hid]
            else:
                data_gen_v = self.model.random_init_config_v(custom_size=(Ndata, batches), zeros=True)
                data_gen_h = self.model.random_init_config_h(custom_size=(Ndata, batches), zeros=True)
                data_gen_v[0] = config[0][0].clone()

                clone = self.model.clone_h(config[1], sub_index=0)
                for hid in range(self.model.h_layer_num):
                    data_gen_h[hid][0] = clone[hid]

            for n in range(Ndata - 1):
                for _ in range(Nstep):
                    energy = self.model.energy_PT(config[0], config[1], N_PT)
                    config[0], config[1], energy = self.model.markov_PT_and_exchange(config[0], config[1], energy, N_PT)
                    if update_betas:
                        self.update_betas(N_PT, beta=beta)


                if record_replica:
                    data_gen_v[n + 1] = config[0].clone()

                    clone = self.model.clone_h(config[1])
                    for hid in range(self.model.h_layer_num):
                        data_gen_h[hid][n + 1] = clone[hid]


            data = [data_gen_v, data_gen_h]

            if self.record_swaps:
                print('cleaning particle trajectories')
                positions = torch.tensor(self.particle_id)
                invert = torch.zeros([batches, Ndata, N_PT], device=self.device)
                for b in range(batches):
                    for i in range(Ndata):
                        for k in range(N_PT):
                            invert[b, i, k] = torch.nonzero(positions[i, :, b] == k)[0]
                self.particle_id = invert
                self.last_at_zero = torch.zeros([batches, Ndata, N_PT], device=self.device)
                for b in range(batches):
                    for i in range(Ndata):
                        for k in range(N_PT):
                            tmp = torch.nonzero(self.particle_id[b, :i, k] == 0)[0]
                            if len(tmp) > 0:
                                self.last_at_zero[b, i, k] = i - 1 - tmp.max()
                            else:
                                self.last_at_zero[b, i, k] = 1000
                self.last_at_zero[:, 0, 0] = 0

                self.trip_duration = torch.zeros([batches, Ndata], device=self.device)
                for b in range(batches):
                    for i in range(Ndata):
                        self.trip_duration[b, i] = self.last_at_zero[b, i, torch.nonzero(invert[b, i, :] == 9)[0]]

            if reshape:
                data[0] = data[0].flatten(0, -3)
                data[1] = [hd.flatten(0, -3) for hd in data[1]]
            else:
                data[0] = data[0]
                data[1] = data[1]

            return data

    def gen_data(self, Nchains=10, Lchains=100, Nthermalize=0, Nstep=1, N_PT=2, config_init=[], beta=1, batches=None,
                 reshape=True, record_replica=False, record_acceptance=None, update_betas=None, record_swaps=False):
        """
        Generate Monte Carlo samples from the RBM. Starting from random initial conditions, Gibbs updates are performed to sample from equilibrium.
        Inputs :
            Nchains (10): Number of Markov chains
            Lchains (100): Length of each chain
            Nthermalize (0): Number of Gibbs sampling steps to perform before the first sample of a chain.
            Nstep (1): Number of Gibbs sampling steps between each sample of a chain
            N_PT (2): Number of Monte Carlo Exchange replicas to use. This==useful if the mixing rate==slow. Watch self.acceptance_rates_g to check that it==useful (acceptance rates about 0==useless)
            batches (10): Number of batches. Must divide Nchains. higher==better for speed (more vectorized) but more RAM consuming.
            reshape (True): If True, the output==(Nchains x Lchains, n_visibles/ n_hiddens) (chains mixed). Else, the output==(Nchains, Lchains, n_visibles/ n_hiddens)
            config_init ([]). If not [], a Nchains X n_visibles numpy array that specifies initial conditions for the Markov chains.
            beta (1): The inverse temperature of the model.
        """
        if N_PT < 2:
            raise Exception("N_PT must be greater than 1")

        with torch.no_grad():
            if batches == None:
                batches = Nchains
            n_iter = int(Nchains / batches)
            Ndata = Lchains * batches
            if record_replica:
                reshape = False

                if record_acceptance == None:
                    record_acceptance = True

                if update_betas == None:
                    update_betas = False

                if update_betas:
                    record_acceptance = True

            else:
                record_acceptance = False
                update_betas = False

            if record_replica:
                visible_data = self.model.random_init_config_v(custom_size=(Nchains, N_PT, Lchains), zeros=True)
                hidden_data = self.model.random_init_config_h(custom_size=(Nchains, N_PT, Lchains), zeros=True)
                data = [visible_data, hidden_data]

            if config_init is not []:
                if type(config_init) == torch.tensor:
                    h_layer = self.model.random_init_config_h()
                    config_init = [config_init, h_layer]

            for i in range(n_iter):
                if config_init != []:
                    config_init = [config_init[0][batches * i:batches * (i + 1)],
                                   config_init[1][batches * i:batches * (i + 1)]]

                config = self._gen_data(Nthermalize, Ndata, Nstep, N_PT=N_PT, batches=batches, reshape=False,
                                        beta=beta, record_replica=record_replica, update_betas=update_betas)


                if record_replica:
                    data[0][batches * i:batches * (i + 1)] = torch.swapaxes(config[0], 0, 2).clone()
                    for hid in range(self.model.h_layer_num):
                        data[1][hid][batches * i:batches * (i + 1)] = torch.swapaxes(config[1][hid], 0, 2).clone()

            if reshape:
                return [data[0].flatten(0, -3), [hd.flatten(0, -3) for hd in data[1]]]
            else:
                return data