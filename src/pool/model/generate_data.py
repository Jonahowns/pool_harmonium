import torch
import time
import types
import numpy as np
from copy import copy


class DataSampler:
    def __init__(self, pool_crbm, device=torch.device('cpu'), **kwargs):

        self.model = pool_crbm
        self.device = device

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


######### Data Generation Methods #########
def extract_cluster_crbm_pool(model, hidden_indices):
    tmp_model = copy.deepcopy(model)
    if "relu" in tmp_model._get_name():
        param_keys = ["gamma", "theta", "W", "0gamma", "0theta"]
    else:
        param_keys = ["gamma+", "gamma-", "theta+", "theta-", "W",  "0gamma+", "0theta+", "0gamma-", "0theta-"]

    assert len(hidden_indices) == len(model.hidden_convolution_keys)
    for kid, key in enumerate(tmp_model.hidden_convolution_keys):
        for pkey in param_keys:
            setattr(tmp_model, f"{key}_{pkey}", torch.nn.Parameter(getattr(tmp_model, f"{key}_{pkey}")[hidden_indices[kid]], requires_grad=False))
        tmp_model.max_inds[kid] = (tmp_model.max_inds[kid])[hidden_indices[kid]]

    # edit keys
    keys_to_del = []
    for kid, key in enumerate(tmp_model.hidden_convolution_keys):
        new_hidden_number = len(hidden_indices[kid])
        if new_hidden_number == 0:
            keys_to_del.append(key)
        else:
            tmp_model.convolution_topology[key]["number"] = new_hidden_number
            wdims = list(tmp_model.convolution_topology[key]["weight_dims"])
            wdims[0] = new_hidden_number
            tmp_model.convolution_topology[key]["weight_dims"] = tuple(wdims)
            cdims = list(tmp_model.convolution_topology[key]["convolution_dims"])
            cdims[1] = new_hidden_number
            tmp_model.convolution_topology[key]["convolution_dims"] = tuple(cdims)

    for key in keys_to_del:
        tmp_model.convolution_topology.pop(key)
        tmp_model.hidden_convolution_keys.pop(key)

    return tmp_model

def gen_data_biased_ih(model, ih_means, ih_stds, samples=500):
    if type(ih_means) is not list:
        ih_means = [ih_means]
        ih_stds = [ih_stds]

    v_out = []
    for i in range(len(ih_means)):
        v_out.append(ih_means[i][None, :] + torch.randn((samples, 1)) * ih_stds[i][None, :])

    hs = model.sample_from_inputs_h(v_out)
    return model.sample_from_inputs_v(model.compute_output_h(hs))


def gen_data_biased_h(model, h_means, h_stds, samples=500):
    if type(h_means) is not list:
        h_means = [h_means]
        h_stds = [h_stds]

    sampled_h = []
    for i in range(len(h_means)):
        sampled_h.append(h_means[i][None, :] + torch.randn((samples, 1)) * h_stds[i][None, :])

    return model.sample_from_inputs_v(model.compute_output_h(sampled_h))


def gen_data_lowT(model, beta=1, which = 'marginal' ,Nchains=10, Lchains=100, Nthermalize=0, Nstep=1, N_PT=1, reshape=True, update_betas=False, config_init=[]):
    tmp_model = copy.deepcopy(model)
    name = tmp_model._get_name()
    if "CRBM" in name:
        setattr(tmp_model, "fields", torch.nn.Parameter(getattr(tmp_model, "fields") * beta, requires_grad=False))
        if "class" in name:
            setattr(tmp_model, "y_bias", torch.nn.Parameter(getattr(tmp_model, "y_bias") * beta, requires_grad=False))

        if which == 'joint':
            if "relu" in "name":
                param_keys = ["gamma", "theta", "W"]
            else:
                param_keys = ["gamma+", "gamma-", "theta+", "theta-", "W"]
            if "class" in name:
                param_keys.append("M")
            for key in tmp_model.hidden_convolution_keys:
                for pkey in param_keys:
                    setattr(tmp_model, f"{key}_{pkey}", torch.nn.Parameter(getattr(tmp_model, f"{key}_{pkey}") * beta, requires_grad=False))
        elif which == "marginal":
            if "relu" in "name":
                param_keys = ["gamma+", "gamma-", "theta+", "theta-", "W", "0gamma+", "0gamma-", "0theta+", "0theta-"]
            else:
                param_keys = ["gamma", "0gamma", "theta", "0theta", "W"]
            if "class" in name:
                param_keys.append("M")
            new_convolution_keys = copy.deepcopy(tmp_model.hidden_convolution_keys)

            # Setup Steps for editing the hidden layer topology of our model
            setattr(tmp_model, "convolution_topology", copy.deepcopy(model.convolution_topology))
            tmp_model_conv_topology = getattr(tmp_model, "convolution_topology")  # Get and edit tmp_model_conv_topology

            if "pool" in name:
                tmp_model.pools = tmp_model.pools * beta
                tmp_model.unpools = tmp_model.unpools * beta
            else:
                # Also need to fix up parameter hidden_layer_W
                tmp_model.register_parameter("hidden_layer_W", torch.nn.Parameter(getattr(tmp_model, "hidden_layer_W").repeat(beta), requires_grad=False))

            # Add keys for new layers, add entries to convolution_topology for new layers, and add parameters for new layers
            for key in tmp_model.hidden_convolution_keys:
                for b in range(beta - 1):
                    new_key = f"{key}_{b}"
                    new_convolution_keys.append(new_key)
                    tmp_model_conv_topology[f"{new_key}"] = copy.deepcopy(tmp_model_conv_topology[f"{key}"])

                    for pkey in param_keys:
                        new_param_key = f"{new_key}_{pkey}"
                        # setattr(tmp_model, new_param_key, torch.nn.Parameter(getattr(tmp_model, f"{key}_{pkey}"), requires_grad=False))
                        tmp_model.register_parameter(new_param_key, torch.nn.Parameter(getattr(tmp_model, f"{key}_{pkey}"), requires_grad=False))

            tmp_model.hidden_convolution_keys = new_convolution_keys
    elif "RBM" in name:
        with torch.no_grad():
            if which == 'joint':
                tmp_model.params["fields"] *= beta
                tmp_model.params["W_raw"] *= beta
                tmp_model.params["gamma+"] *= beta
                tmp_model.params["gamma-"] *= beta
                tmp_model.params["theta+"] *= beta
                tmp_model.params["theta-"] *= beta
            elif which == 'marginal':
                if type(beta) == int:
                    tmp_model.params["fields"] *= beta
                    tmp_model.params["W_raw"] = torch.nn.Parameter(torch.repeat_interleave(model.params["W_raw"], beta, dim=0), requires_grad=False)
                    tmp_model.params["gamma+"] = torch.nn.Parameter(torch.repeat_interleave(model.params["gamma+"], beta, dim=0), requires_grad=False)
                    tmp_model.params["gamma-"] = torch.nn.Parameter(torch.repeat_interleave(model.params["gamma-"], beta, dim=0), requires_grad=False)
                    tmp_model.params["theta+"] = torch.nn.Parameter(torch.repeat_interleave(model.params["theta+"], beta, dim=0), requires_grad=False)
                    tmp_model.params["theta-"] = torch.nn.Parameter(torch.repeat_interleave(model.params["theta-"], beta, dim=0), requires_grad=False)
                    tmp_model.params["0gamma+"] = torch.nn.Parameter(torch.repeat_interleave(model.params["0gamma+"], beta, dim=0), requires_grad=False)
                    tmp_model.params["0gamma-"] = torch.nn.Parameter(torch.repeat_interleave(model.params["0gamma-"], beta, dim=0), requires_grad=False)
                    tmp_model.params["0theta+"] = torch.nn.Parameter(torch.repeat_interleave(model.params["0theta+"], beta, dim=0), requires_grad=False)
                    tmp_model.params["0theta-"] = torch.nn.Parameter(torch.repeat_interleave(model.params["0theta-"], beta, dim=0), requires_grad=False)
            tmp_model.prep_W()

    return tmp_model.data_sampler.gen_data(Nchains=Nchains,Lchains=Lchains,Nthermalize=Nthermalize,Nstep=Nstep,N_PT=N_PT,reshape=reshape,update_betas=update_betas,config_init = config_init)


def gen_data_lowT_cluster(model, cluster_indx, beta=1, which = 'marginal' ,Nchains=10, Lchains=100, Nthermalize=0, Nstep=1, N_PT=1, reshape=True, update_betas=False, config_init=[]):
    tmp_model = copy.deepcopy(model)
    name = tmp_model._get_name()
    if "CRBM" in name:
        setattr(tmp_model, "fields", torch.nn.Parameter(getattr(tmp_model, "fields") * beta, requires_grad=False))
        if "class" in name:
            setattr(tmp_model, "y_bias", torch.nn.Parameter(getattr(tmp_model, "y_bias") * beta, requires_grad=False))

        if which == 'joint':
            param_keys = ["gamma+", "gamma-", "theta+", "theta-", "W"]
            if "class" in name:
                param_keys.append("M")
            for key in tmp_model.hidden_convolution_keys[cluster_indx]:
                for pkey in param_keys:
                    setattr(tmp_model, f"{key}_{pkey}", torch.nn.Parameter(getattr(tmp_model, f"{key}_{pkey}") * beta, requires_grad=False))
        elif which == "marginal":
            param_keys = ["gamma+", "gamma-", "theta+", "theta-", "W", "0gamma+", "0gamma-", "0theta+", "0theta-"]
            if "class" in name:
                param_keys.append("M")
            new_convolution_keys = copy.deepcopy(tmp_model.hidden_convolution_keys[cluster_indx])

            # Setup Steps for editing the hidden layer topology of our model
            setattr(tmp_model, "convolution_topology", copy.deepcopy(model.convolution_topology))
            tmp_model_conv_topology = getattr(tmp_model, "convolution_topology")  # Get and edit tmp_model_conv_topology

            if "pool" in name or "pcrbm" in name:
                tmp_model.pools = tmp_model.pools[cluster_indx] * beta
                tmp_model.unpools = tmp_model.unpools[cluster_indx] * beta
            else:
                # Also need to fix up parameter hidden_layer_W
                tmp_model.register_parameter("hidden_layer_W", torch.nn.Parameter(getattr(tmp_model, "hidden_layer_W").repeat(beta), requires_grad=False))

            # Add keys for new layers, add entries to convolution_topology for new layers, and add parameters for new layers
            for key in tmp_model.hidden_convolution_keys[cluster_indx]:
                for b in range(beta - 1):
                    new_key = f"{key}_{b}"
                    new_convolution_keys.append(new_key)
                    tmp_model_conv_topology[cluster_indx][f"{new_key}"] = copy.deepcopy(tmp_model_conv_topology[f"{key}"])

                    for pkey in param_keys:
                        new_param_key = f"{new_key}_{pkey}"
                        # setattr(tmp_model, new_param_key, torch.nn.Parameter(getattr(tmp_model, f"{key}_{pkey}"), requires_grad=False))
                        tmp_model.register_parameter(new_param_key, torch.nn.Parameter(getattr(tmp_model, f"{key}_{pkey}"), requires_grad=False))

            tmp_model.hidden_convolution_keys[cluster_indx] = new_convolution_keys


    return tmp_model.data_sampler.gen_data_cluster(cluster_indx, Nchains=Nchains,Lchains=Lchains,Nthermalize=Nthermalize,Nstep=Nstep,N_PT=N_PT,reshape=reshape,update_betas=update_betas,config_init = config_init)


def gen_data_zeroT(model, which = 'marginal' ,Nchains=10,Lchains=100,Nthermalize=0,Nstep=1,N_PT=1,reshape=True,update_betas=False,config_init=[]):
    tmp_model = copy.deepcopy(model)
    if "class" in tmp_model._get_name():
        print("Zero Temp Generation of Data not available for classification CRBM")
    with torch.no_grad():
        if which == 'joint':
            tmp_model.markov_step = types.MethodType(markov_step_zeroT_joint, tmp_model)
        elif which == 'marginal':
            tmp_model.markov_step = types.MethodType(markov_step_zeroT_marginal, tmp_model)
        return tmp_model.data_sampler.gen_data(Nchains=Nchains,Lchains=Lchains,Nthermalize=Nthermalize,Nstep=Nstep,N_PT=N_PT,reshape=reshape,update_betas=update_betas,config_init = config_init)

def markov_step_zeroT_joint(self, v, beta=1):
    I = self.compute_output_v(v)
    h = self.transform_h(I)
    I = self.compute_output_h(h)
    nv = self.transform_v(I)
    return nv, h

def markov_step_zeroT_marginal(self, v,beta=1):
    I = self.compute_output_v(v)
    h = self.mean_h(I)
    I = self.compute_output_h(h)
    nv = self.transform_v(I)
    return nv, h

