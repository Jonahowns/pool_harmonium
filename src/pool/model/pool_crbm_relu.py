import math
import numpy as np
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.utils.data import DataLoader

from pool.dataset import Categorical
from pool.utils.model_utils import conv2d_dim
from pool.model.base import BaseRelu
from pool.model.generate_data import DataSampler


class PoolCRBMRelu(BaseRelu):
    def __init__(self, config, debug=False):
        super().__init__(config, debug=debug)

        mandatory_keys = ['mc_moves', 'sample_type', "l1_2", "lf", "ld", "lgap", "lcov", "ls", "convolution_topology"]

        for key in mandatory_keys:
            setattr(self, key, config[key])

        assert type(self.mc_moves) is int
        assert self.sample_type in ['gibbs', 'pt', 'pcd']
        assert type(self.l1_2) is float or type(self.l1_2) is int
        assert type(self.lf) is float or type(self.lf) is int
        assert type(self.ld) is float or type(self.ld) is int
        assert type(self.lgap) is float or type(self.lgap) is int
        assert type(self.lcov) is float or type(self.lcov) is int
        assert type(self.ls) is float or type(self.ls) is int
        assert type(self.convolution_topology) is dict

        # Set visible biases
        self.weight_initial_amplitude = np.sqrt(0.01 / math.prod(self.v_num))
        self.register_parameter("fields", nn.Parameter(torch.zeros((*self.v_num, self.q), device=self.device)))
        self.register_parameter("fields0", nn.Parameter(torch.zeros((*self.v_num, self.q), device=self.device)))

        # Container for pooling elements
        self.pools = []
        self.unpools = []

        self.hidden_convolution_keys = list(self.convolution_topology.keys())

        for key in self.hidden_convolution_keys:
            # Set information about the convolutions that will be useful
            dims = conv2d_dim([self.batch_size, 1, *self.v_num, self.q], self.convolution_topology[key])
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

        # Saves Our hyperparameter options into the checkpoint file generated for Each Run of the Model
        # i.e. Simplifies loading a model that has already been run
        self.save_hyperparameters()

        # Initialize AIS/PT members
        self.data_sampler = DataSampler(self)
        self.log_Z_AIS = None
        self.log_Z_AIS_std = None

        # Empty function, useful for inherited classes
        self.training_callback = lambda *args: None
        self.validation_callback = lambda *args: None


        # Set training Function
        if self.sample_type == "gibbs":
            self.training_step = self.training_step_CD_free_energy
        elif self.sample_type == "pt":
            self.training_step = self.training_step_PT_free_energy
        elif self.sample_type == "pcd":
            self.training_step = self.training_step_PCD_free_energy

    @property
    def h_layer_num(self):
        return len(self.hidden_convolution_keys)

    def on_train_start(self):
        super().on_train_start()
        self.data_sampler.set_device(self.device)

    def free_energy(self, v, beta=1):
        return self.energy_v(v, beta=beta) - self.logpartition_h(self.compute_output_v(v), beta=beta)

    def free_energy_ind(self, v, beta=1):
        """Free energy contribution frome each hidden node"""
        h_ind = self.logpartition_h_ind(self.compute_output_v(v), beta=beta)
        return (self.energy_v(v, beta=beta)/h_ind.shape[1]).unsqueeze(1) - h_ind

    def free_energy_h(self, h):
        return self.energy_h(h) - self.logpartition_v(self.compute_output_h(h))

    def energy(self, v, h, beta=1, remove_init=False, hidden_sub_index=-1):
        """Total Energy of a given visible and hidden configuration"""
        return self.energy_v(v, beta=beta, remove_init=remove_init) + \
            self.energy_h(h, sub_index=hidden_sub_index, beta=beta, remove_init=remove_init) \
            - self.bidirectional_weight_term(v, h, hidden_sub_index=hidden_sub_index)

    def energy_PT(self, v, h, N_PT, beta=1, remove_init=False):
        """Total Energy of N_PT given visible and hidden configurations"""
        E = torch.zeros((N_PT, v.shape[1]), device=self.device)
        for i in range(N_PT):
            E[i] = self.energy_v(v[i], beta=beta, remove_init=remove_init) + \
                   self.energy_h(h, sub_index=i, beta=beta, remove_init=remove_init) \
                   - self.bidirectional_weight_term(v[i], h, hidden_sub_index=i)
        return E

    def bidirectional_weight_term(self, v, h, hidden_sub_index=-1):
        conv = self.compute_output_v(v)
        E = torch.zeros((len(self.hidden_convolution_keys), conv[0].shape[0]), device=self.device)
        for iid, i in enumerate(self.hidden_convolution_keys):
            h_uk = h[iid]
            if hidden_sub_index != -1:
                h_uk = h_uk[hidden_sub_index]
            E[iid] = h_uk.mul(conv[iid]).sum(1)

        return E.sum(0)

    def transform_v(self, I):
        return F.one_hot(torch.argmax(I + getattr(self, "fields").unsqueeze(0), dim=-1), self.q)

    def transform_h(self, I):
        output = []
        for kid, key in enumerate(self.hidden_convolution_keys):
            gamma = (getattr(self, f'{key}_gamma')).unsqueeze(0).unsqueeze(2)
            theta = (getattr(self, f'{key}_theta')).unsqueeze(0).unsqueeze(2)
            output.append(torch.maximum(I - theta, torch.tensor(0, device=self.device)) / gamma)
        return output

    def energy_v(self, visible_config, beta=1, remove_init=False):
        """Computes -g(si) term of potential"""
        v = visible_config.type(torch.get_default_dtype())
        E = torch.zeros(visible_config.shape[0], device=self.device)
        if remove_init:
            E -= (v[:] * (getattr(self, "fields") - getattr(self, "fields0"))).sum((2, 1))
        else:
            E -= (v[:] * (beta * getattr(self, "fields") + (1-beta) * getattr(self, "fields0"))).sum((2, 1))
        return E

    def energy_h(self, hidden_config, beta=1, sub_index=-1, remove_init=False):
        """Computes U(h) term of potential"""
        # hidden_config is list of h_uks
        if sub_index != -1:
            E = torch.zeros((len(self.hidden_convolution_keys), hidden_config[0].shape[1]), device=self.device)
        else:
            E = torch.zeros((len(self.hidden_convolution_keys), hidden_config[0].shape[0]), device=self.device)

        for iid, i in enumerate(self.hidden_convolution_keys):
            if remove_init:
                gamma = getattr(self, f'{i}_gamma').sub(getattr(self, f'{i}_0gamma')).unsqueeze(0)
                theta = getattr(self, f'{i}_theta').sub(getattr(self, f'{i}_0theta')).unsqueeze(0)
            else:
                gamma = (beta * getattr(self, f'{i}_gamma') + (1 - beta) * getattr(self, f'{i}_0gamma')).unsqueeze(0)
                theta = (beta * getattr(self, f'{i}_theta') + (1 - beta) * getattr(self, f'{i}_0theta')).unsqueeze(0)

            if sub_index != -1:
                con = hidden_config[iid][sub_index]
            else:
                con = hidden_config[iid]

            E[iid] = (con.square() * gamma).sum(-1) / 2 + (con * theta).sum(-1)

        return E.sum(0)


    def random_init_config_v(self, custom_size=False, zeros=False):
        """Random Config of Visible Potts States"""
        if custom_size:
            size = (*custom_size, *self.v_num, self.q)
        else:
            size = (self.batch_size, *self.v_num, self.q)

        if zeros:
            return torch.zeros(size, device=self.device)
        else:
            return self.sample_from_inputs_v(torch.zeros(size, device=self.device).flatten(0, -3), beta=0).reshape(size)

    def random_init_config_h(self, zeros=False, custom_size=False):
        """Random Config of Hidden ReLU States"""
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
                config.append(self.sample_from_inputs_h(
                    [torch.zeros(size, device=self.device).flatten(0, -2)], beta=0)[0].reshape(size))

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

    def logpartition_h(self, inputs, beta=1):
        """Marginal over hidden units"""
        # Input is list of matrices I_uk
        marginal = torch.zeros((len(self.hidden_convolution_keys), inputs[0].shape[0]), device=self.device)
        for iid, i in enumerate(self.hidden_convolution_keys):
            marginal[iid] = self.cgf_from_inputs_h(inputs[iid], hidden_key=i, beta=beta).sum(-1)
        return marginal.sum(0)

    def logpartition_h_ind(self, inputs, beta=1):
        """Marginal over hidden units"""
        ys = []
        for iid, i in enumerate(self.hidden_convolution_keys):
            y = self.cgf_from_inputs_h(inputs[iid], hidden_key=i, beta=beta)
            ys.append(y)
        return torch.cat(ys, dim=1)

    def logpartition_v(self, inputs, beta=1):
        """Marginal over visible units"""
        return torch.logsumexp((beta * getattr(self, "fields") + (1 - beta) * getattr(self, "fields0"))[None, :] + beta * inputs, 2).sum(1)

    def mean_h(self, psi, hidden_key=None, beta=1):
        """Mean of hidden layer specified by hidden_key"""
        if hidden_key is None:
            hidden_key = self.hidden_convolution_keys
        elif type(hidden_key) is str:
            hidden_key = [hidden_key]

        means = []
        for kid, key in enumerate(self.hidden_convolution_keys):
            theta = (beta * getattr(self, f'{key}_theta') + (1 - beta) * getattr(self, f'{key}_0theta')).unsqueeze(0)
            gamma = (beta * getattr(self, f'{key}_gamma') + (1 - beta) * getattr(self, f'{key}_0gamma')).unsqueeze(0)
            psi[kid] *= beta

            sqrt_gamma = torch.sqrt(gamma)
            means.append((psi[kid] - theta) / gamma + 1. / self.erf_times_gauss((-psi[kid] + theta) / sqrt_gamma) / sqrt_gamma)

        if len(hidden_key) == 1:
            return means[0]

        return means

    def compute_output_v(self, X):
        """Compute Input for Hidden Layer from Visible Potts, Uses one hot vector"""
        outputs = []
        self.max_inds = []
        self.min_inds = []
        for iid, i in enumerate(self.hidden_convolution_keys):
            weights = getattr(self, f"{i}_W")
            conv = F.conv2d(X.unsqueeze(1).type(torch.get_default_dtype()), weights, stride=self.convolution_topology[i]["stride"],
                            padding=self.convolution_topology[i]["padding"],
                            dilation=self.convolution_topology[i]["dilation"]).squeeze(3)

            max_pool, max_inds = self.pools[iid](conv)

            self.max_inds.append(max_inds)
            out = max_pool.flatten(start_dim=2)
            out.squeeze_(2)

            if self.dr > 0.:
                out = F.dropout(out, p=self.dr, training=self.training)

            outputs.append(out)
            if True in torch.isnan(out):
                print("Nan in hidden unit input")

        return outputs

    def compute_output_h(self, h):  # from h_uk (B, hidden_num)
        """Compute Input for Visible Layer from Hidden dReLU"""
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

    def sample_from_inputs_v(self, psi, beta=1):  # Psi ohe (Batch_size, v_num, q)   fields (self.v_num, self.q)
        """Gibbs Sampling of Potts Visbile Layer"""
        cum_probas = beta * psi + beta * getattr(self, "fields").unsqueeze(0) + (1 - beta) * getattr(self, "fields0").unsqueeze(0)

        maxi, max_indices = cum_probas.max(-1)
        maxi.unsqueeze_(2)
        cum_probas -= maxi
        cum_probas.exp_()
        cum_probas[cum_probas > 1e9] = 1e9  # For numerical stability.

        dist = torch.distributions.categorical.Categorical(probs=cum_probas)
        return F.one_hot(dist.sample(), self.q)

    def sample_from_inputs_h(self, psi, beta=1):  # psi is a list of hidden [input]
        """Gibbs Sampling of ReLU hidden layer"""
        h_uks = []
        for iid, i in enumerate(self.hidden_convolution_keys):
            theta = (beta * getattr(self, f'{i}_theta') + (1 - beta) * getattr(self, f'{i}_0theta')).unsqueeze(0)
            gamma = (beta * getattr(self, f'{i}_gamma') + (1 - beta) * getattr(self, f'{i}_0gamma')).unsqueeze(0)
            psi[iid] *= beta

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

    def markov_step(self, v, beta=1):
        """Gibbs Sampler, Samples hidden from visible and vice versa, returns newly sampled hidden and visible"""
        h = self.sample_from_inputs_h(self.compute_output_v(v), beta=beta)
        return self.sample_from_inputs_v(self.compute_output_h(h), beta=beta), h

    ######################################################### Pytorch Lightning Functions
    # def on_after_backward(self):
        # torch.nn.utils.clip_grad_norm_(self.parameters(), 10000, norm_type=2.0, error_if_nonfinite=True)

    # Clamps hidden potential values to acceptable range
    # def on_before_zero_grad(self, optimizer):
    #     """ clip parameters to acceptable values """
    #     with torch.no_grad():
    #         for key in self.hidden_convolution_keys:
    #             getattr(self, f"{key}_gamma").data.clamp_(min=0.05)
    #             getattr(self, f"{key}_theta").data.clamp_(min=0.0)
    #             getattr(self, f"{key}_W").data.clamp_(-1.0, 1.0)

    def on_before_backward(self, loss):
        """ clip parameters to acceptable values """
        for key in self.hidden_convolution_keys:
            getattr(self, f"{key}_gamma").data.clamp_(min=0.05, max=2.0)
            getattr(self, f"{key}_theta").data.clamp_(min=0.0, max=2.0)
            getattr(self, f"{key}_W").data.clamp_(-1.0, 1.0)

    def validation_step(self, batch, batch_idx):
        self.validation_callback()
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

    def regularization_terms(self, distance_threshold=0.4):
        # regularizer on visible biases
        freg = self.lf / (2 * np.prod(self.v_num) * self.q) * getattr(self, "fields").square().sum((0, 1))
        # promotes sparsity on convolutional weights
        wreg = torch.zeros((1,), device=self.device)
        # discourages weights that are alike
        dreg = torch.zeros((1,), device=self.device)
        # discourages learning gaps
        gap_loss = torch.zeros((1,), device=self.device)

        for iid, i in enumerate(self.hidden_convolution_keys):
            W_shape = self.convolution_topology[i]["weight_dims"]  # (h_num,  input_channels, kernel0, kernel1)
            W = getattr(self, f"{i}_W")

            x = torch.sum(W.abs(), (3, 2, 1)).square()
            wreg += x.sum() * self.l1_2 / (W_shape[0] * W_shape[1] * W_shape[2] * W_shape[3])
            gap_loss += self.lgap * W[:, :, :, -1].abs().mean()

            # below is for distance reg
            ws = W.squeeze(1)
            with torch.no_grad():
                # compute positional differences for all pairs of weights
                pdiff = (ws.unsqueeze(0).unsqueeze(2) - ws.unsqueeze(1).unsqueeze(3)).sum(4)

                # concatenate it to itself to make full diagonals
                wdm = torch.concat([pdiff, pdiff.clone()], dim=2)

                # get stride to get matrix of all diagonals on separate rows
                #        xx xy xz xx xy xz
                #        yx yy yz yx yy yz
                #        zx zy zz zx zy zz

                # strided becomes
                # xx yy zz
                # xy yz zx
                # xz yx zy

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
            thresh = angles > distance_threshold
            if thresh.any():
                angles = angles[angles > distance_threshold]
                dreg += angles.mean()

        dreg *= self.ld

        # Passed to training logger
        reg_dict = {
            "field_reg": freg.detach(),
            "weight_reg": wreg.detach(),
            "distance_reg": dreg.detach(),
            "gap_reg": gap_loss.detach(),
        }

        return freg, wreg, dreg, gap_loss, reg_dict

    def training_step_PT_free_energy(self, batch, batch_idx):
        self.training_callback()
        inds, seqs, one_hot, seq_weights = batch

        V_neg_oh, h_neg, V_pos_oh, h_pos = self(one_hot)

        # Calculate CD loss
        F_v = (self.free_energy(V_pos_oh) * seq_weights).sum() / seq_weights.sum()  # free energy of training data
        F_vp = (self.free_energy(V_neg_oh) * seq_weights).sum() / seq_weights.sum()  # free energy of gibbs sampled visible states
        cd_loss = F_v - F_vp

        # Regularization Terms
        regs = self.regularization_terms()

        # Calc loss
        loss = cd_loss + sum(regs[:-1])

        logs = {"loss": loss,
                "free_energy_diff": cd_loss.detach(),
                "train_free_energy": F_v.detach(),
                **regs[-1]
                }

        self.log("ptl/train_free_energy", logs["train_free_energy"], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("ptl/train_loss", loss.detach(), on_step=True, on_epoch=True, prog_bar=True, logger=True)

        self.training_data_logs.append(logs)
        return logs["loss"]

    def training_step_CD_free_energy(self, batch, batch_idx):
        self.training_callback()
        inds, seqs, one_hot, seq_weights = batch

        V_neg_oh, h_neg = self(one_hot)
        F_v = (self.free_energy(one_hot) * seq_weights / seq_weights.sum())  # free energy of training data
        F_vp = (self.free_energy(V_neg_oh) * seq_weights / seq_weights.sum()) # free energy of gibbs sampled visible states
        cd_loss = (F_v - F_vp).mean()

        free_energy_log = {
            "free_energy_pos": F_v.sum().detach(),
            "free_energy_neg": F_vp.sum().detach(),
            "free_energy_diff": cd_loss.sum().detach(),
            "cd_loss": cd_loss.detach(),
        }

        # Regularization Terms
        regs = self.regularization_terms()

        # Calc loss
        loss = cd_loss + sum(regs[:-1])

        logs = {"loss": loss,
                "train_free_energy": cd_loss.sum().detach(),
                **free_energy_log,
                **regs[-1]
                }

        self.log("ptl/train_free_energy", logs["train_free_energy"], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("ptl/train_loss", loss.detach(), on_step=True, on_epoch=True, prog_bar=True, logger=True)

        self.training_data_logs.append(logs)
        return logs["loss"]

    def training_step_PCD_free_energy(self, batch, batch_idx):
        self.training_callback()
        inds, seqs, one_hot, seq_weights = batch

        if self.current_epoch == 0 and batch_idx == 0:
            self.chain = torch.zeros((self.training_data.index.__len__(), *one_hot.shape[1:]), device=self.device)

        if self.current_epoch == 0:
            self.chain[inds] = one_hot.type(torch.get_default_dtype())

        with torch.no_grad():
            V_oh_neg, h_neg = self.forward_PCD(inds)

        ### Normal Way
        F_v = self.free_energy(one_hot)
        F_vp = self.free_energy(V_oh_neg)

        # standard deviation of free energy
        std_loss = self.std(F_v).clamp(min=0) * self.ls

        # covariance loss on hidden unit input
        inputs_flat = torch.concat(self.compute_output_v(one_hot), 1)
        cov_loss = (torch.cov(inputs_flat.T).triu().abs().sum())/((inputs_flat.shape[1]*(inputs_flat.shape[1]+1))/2) * self.lcov

        # contrastive divergence
        cd_loss = (F_v - F_vp).mean()

        # Regularization Terms
        regs = self.regularization_terms()

        loss = cd_loss + sum(regs[:-1]) + std_loss + cov_loss

        if loss.isnan():
            print("okay")

        logs = {"loss": loss,
                "free_energy_diff": cd_loss.detach(),
                "free_energy_pos": F_v.mean().detach(),
                "free_energy_neg": F_vp.mean().detach(),
                "std_loss": std_loss.detach(),
                "cov_loss": cov_loss.detach(),
                **regs[-1]
                }

        # logging
        self.log("ptl/free_energy_diff", logs["free_energy_diff"], on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=one_hot.shape[0])
        self.log("ptl/train_free_energy", logs["free_energy_pos"], on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=one_hot.shape[0])
        self.log("ptl/train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=one_hot.shape[0])
        self.training_data_logs.append(logs)
        return logs["loss"]

    def std(self, x):
        # mean = torch.mean(x)
        # diffs = (x - mean) + 1e-12
        std = torch.std(x) + 1e-12
        # zscores = diffs/std
        # skew = torch.mean(torch.pow(zscores, 3.0), 0)
        # kurtosis = torch.mean(torch.pow(zscores, 4.0), 0)
        # dm = diffs.max()
        # bimodality = (skew.square() + 1) / kurtosis
        return (std-3).clamp(min=0)  # + ((dm.abs()-5)/5).clamp(min=0) #(std-4).clamp(min=0) + skew.abs()*3 (kurtosis - 2).clamp(min=0)*3

    def forward_PCD(self, inds):
        """Gibbs sampling with Persistent Contrastive Divergence"""
        # Last sample that was saved to self.chain variable, initialized in training step
        fantasy_v = self.chain[inds]
        for _ in range(self.mc_moves - 1):
            fantasy_v, fantasy_h = self.markov_step(fantasy_v)

        V_neg, fantasy_h = self.markov_step(fantasy_v)
        h_neg = self.sample_from_inputs_h(self.compute_output_v(V_neg))
        # Save current sample for next iteration
        self.chain[inds] = V_neg.detach().type(torch.get_default_dtype())

        return V_neg, h_neg

    def forward_PT(self, V_pos_ohe, N_PT):
        # Parallel Tempering
        n_chains = V_pos_ohe.shape[0]

        fantasy_v = self.random_init_config_v(custom_size=(N_PT, n_chains))
        fantasy_h = self.random_init_config_h(custom_size=(N_PT, n_chains))
        fantasy_E = self.energy_PT(fantasy_v, fantasy_h, N_PT)

        fantasy_v, fantasy_h, fantasy_E = self.data_sampler.markov_PT_and_exchange(fantasy_v, fantasy_h, fantasy_E, N_PT)
        self.data_sampler.update_betas(N_PT)

        for _ in range(self.mc_moves - 1):
            fantasy_v, fantasy_h, fantasy_E = self.data_sampler.markov_PT_and_exchange(fantasy_v, fantasy_h, fantasy_E, N_PT)
            self.data_sampler.update_betas(N_PT)

        return fantasy_v[0], fantasy_h[0]

    def forward(self, V_pos_ohe):
        # Gibbs sampling
        fantasy_v, fantasy_h = self.markov_step(V_pos_ohe)
        for _ in range(self.mc_moves - 1):
            fantasy_v, fantasy_h = self.markov_step(fantasy_v)

        return fantasy_v, fantasy_h

    # X must be a pandas dataframe with the sequences in string format under the column 'sequence'
    # Returns the likelihood for each sequence in an array
    def predict(self, dataframe, individual_hiddens=False):
        # Read in data
        reader = Categorical(dataframe, self.q, weights=None, max_length=self.v_num, alphabet=self.alphabet,
                             device=torch.device('cpu'), one_hot=True)
        # Put in Dataloader
        data_loader = DataLoader(
            reader,
            batch_size=self.batch_size,
            num_workers=self.data_worker_num,  # Set to 0 if debug = True
            pin_memory=False,
            shuffle=False
        )
        self.eval()
        with torch.no_grad():
            likelihood = []
            for i, batch in enumerate(data_loader):
                inds, seqs, one_hot, seq_weights = batch
                oh = one_hot.to(torch.device(self.device))
                likelihood += self.likelihood(oh, individual_hiddens=individual_hiddens).detach().cpu().tolist()

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
        I *= beta

        sqrt_gamma = torch.sqrt(gamma).expand(B, -1)
        log_gamma = torch.log(gamma).expand(B, -1)
        return self.log_erf_times_gauss((-I + theta) / sqrt_gamma) - 0.5 * log_gamma


class PCRSpecificity(PoolCRBMRelu):
    def __init__(self, config, debug=False):
        super().__init__(config, debug=debug)
        # specificity fraction
        mandatory_keys = ['sfrac', 'exps']
        for key in mandatory_keys:
            setattr(self, key, config[key])

        assert type(self.sfrac) is float and self.sfrac <= 1.
        assert type(self.exps) is float

        # sets our cutoffs prior to each training iteration
        self.input_filters = None
        self.input_cutoffs = []
        # self.training_callback = self.set_cutoffs
        # self.validation_callback = self.set_cutoffs

        self.save_hyperparameters()


    def set_cutoffs(self):
        self.input_cutoffs = []
        with torch.no_grad():
            for kid, key in enumerate(self.hidden_convolution_keys):
                ws = getattr(self, f"{key}_W").squeeze(1)
                in_max_val_pos = ws.clamp(min=0.).sum(2).sum(1)
                # in_max_val_pos, _ = ws.clamp(min=0.).max(2)
                # in_max_val_pos = in_max_val_pos.sum(1)
                new_cutoff = in_max_val_pos*self.sfrac
                # new_cutoff = in_max_val_pos*(self.sfrac*min(self.current_epoch /
                #                                          (self.epochs*self.sfrac_milestone), 1.))
                self.input_cutoffs.append(new_cutoff)

    def compute_output_v(self, X):
        """Compute Input for Hidden Layer from Visible Potts, Uses one hot vector"""
        self.set_cutoffs()  # sets cutoffs based off current weights
        outputs = []
        self.max_inds = []
        self.min_inds = []
        self.input_filters = []
        self.hidden_cutoffs = []
        for iid, i in enumerate(self.hidden_convolution_keys):
            weights = getattr(self, f"{i}_W")
            conv = F.conv2d(X.unsqueeze(1).type(torch.get_default_dtype()), weights, stride=self.convolution_topology[i]["stride"],
                            padding=self.convolution_topology[i]["padding"],
                            dilation=self.convolution_topology[i]["dilation"]).squeeze(3)

            max_pool, max_inds = self.pools[iid](conv)

            self.max_inds.append(max_inds)
            out = max_pool.flatten(start_dim=2)
            out.squeeze_(2)

            # self.input_filters.append(out < self.input_cutoffs[iid])
            # out[self.input_filters[-1]] /= self.input_cutoffs[iid]
            out = torch.pow((out/self.input_cutoffs[iid]).clamp(min=0.), self.exps)
            self.hidden_cutoffs.append(out.detach().clone())
            # out[self.input_filters[-1]] = 0.

            if self.dr > 0.:
                out = F.dropout(out, p=self.dr, training=self.training)

            outputs.append(out)
            if True in torch.isnan(out):
                print("Nan in hidden unit input")

        return outputs

    def compute_output_h(self, h):  # from h_uk (B, hidden_num)
        """Compute Input for Visible Layer from Hidden dReLU"""
        outputs = []
        for iid, i in enumerate(self.hidden_convolution_keys):
            # h[iid][self.input_filters[iid]] = 0.
            h[iid] = h[iid] * self.hidden_cutoffs[iid]

            reconst = self.unpools[iid](h[iid].view_as(self.max_inds[iid]), self.max_inds[iid])

            if reconst.ndim == 3:
                reconst.unsqueeze_(3)

            output = F.conv_transpose2d(reconst, getattr(self, f"{i}_W"),
                                              stride=self.convolution_topology[i]["stride"],
                                              padding=self.convolution_topology[i]["padding"],
                                              dilation=self.convolution_topology[i]["dilation"],
                                              output_padding=self.convolution_topology[i]["output_padding"]).squeeze(1)

            outputs.append(output)

        if len(outputs) > 1:
            return torch.sum(torch.stack(outputs), 0)

        return outputs[0]
