import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pool.model import PoolCRBMRelu


class DualCRBMRelu(PoolCRBMRelu):
    def __init__(self, config, debug=False):
        super().__init__(config, debug=debug)

        for key in self.hidden_convolution_keys:

            c1_shape = self.convolution_topology[key]["convolution_dims"]
            # conv_output_size = (batch_size, h_num, convx_num)

            # for 1d
            # weight_size = (out_channels, input_channels, kernel)
            self.convolution_topology[key]["weight2_dims"] = (c1_shape[1], c1_shape[1], c1_shape[2])

            self.register_parameter(f"{key}_W2", nn.Parameter(
                self.weight_initial_amplitude * torch.randn(self.convolution_topology[key]["weight2_dims"],
                                                            device=self.device)))

    def compute_output_v(self, X):
        """Compute Input for Hidden Layer from Visible Potts, Uses one hot vector"""
        outputs = []
        for iid, i in enumerate(self.hidden_convolution_keys):
            conv = F.conv2d(X.unsqueeze(1).type(torch.get_default_dtype()), getattr(self, f"{i}_W"),
                            stride=self.convolution_topology[i]["stride"],
                            padding=self.convolution_topology[i]["padding"],
                            dilation=self.convolution_topology[i]["dilation"]).squeeze(3)

            out = F.conv1d(conv, getattr(self, f"{i}_W2"), stride=1, padding=0, dilation=1)

            if self.dr > 0.:
                out = F.dropout(out, p=self.dr, training=self.training)

            out.squeeze_(2)

            outputs.append(out)
            if True in torch.isnan(out):
                print("hi")

            # batch_size, h_num, k convolutions
            # out.squeeze_(2)

        return outputs

    def compute_output_h(self, h):  # from h_uk (B, hidden_num)
        """Compute Input for Visible Layer from Hidden dReLU"""
        outputs = []
        for iid, i in enumerate(self.hidden_convolution_keys):
            reconst = F.conv_transpose1d(h[iid].unsqueeze(2), getattr(self, f'{i}_W2'), stride=1, padding=0,
                                         output_padding=0, dilation=1)

            # reconst = self.unpools[iid](h[iid].view_as(self.max_inds[iid]), self.max_inds[iid])

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

    def regularization_terms(self, distance_threshold=0.25):
        freg = self.lf / (2 * np.prod(self.v_num) * self.q) * getattr(self, "fields").square().sum((0, 1))
        wreg = torch.zeros((1,), device=self.device)
        dreg = torch.zeros((1,), device=self.device)  # discourages weights that are alike

        bs_loss = torch.zeros((1,),
                              device=self.device)  # encourages weights to use both positive and negative contributions
        gap_loss = torch.zeros((1,), device=self.device)  # discourages high values for gaps

        for iid, i in enumerate(self.hidden_convolution_keys):
            W_shape = self.convolution_topology[i]["weight_dims"]  # (h_num,  input_channels, kernel0, kernel1)
            W2_shape = self.convolution_topology[i]["weight2_dims"]  # (h_num, h_num, cx)
            W = getattr(self, f"{i}_W")
            W2 = getattr(self, f"{i}_W")

            x = torch.sum(W.abs(), (3, 2, 1)).square()
            y = torch.sum(W2.abs(), (2, 1)).square()
            wreg += x.sum() * self.l1_2 / (W_shape[0] * W_shape[1] * W_shape[2] * W_shape[3])
            wreg += y.sum() * self.l1_2 / (W_shape[0] * W_shape[1] * W_shape[2])
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
            angles = inner_prod / (w_norms[bat_x] * w_norms[bat_y] + 1e-6)

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
            "both_side_reg": bs_loss.detach()
        }

        return freg, wreg, dreg, bs_loss, gap_loss, reg_dict


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
        cov_loss = (torch.cov(inputs_flat.T).triu().abs().sum())/\
                   ((inputs_flat.shape[1]*(inputs_flat.shape[1]+1))/2) * self.lcorr

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

