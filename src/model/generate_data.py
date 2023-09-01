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

    return tmp_model.gen_data(Nchains=Nchains,Lchains=Lchains,Nthermalize=Nthermalize,Nstep=Nstep,N_PT=N_PT,reshape=reshape,update_betas=update_betas,config_init = config_init)

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


    return tmp_model.gen_data_cluster(cluster_indx, Nchains=Nchains,Lchains=Lchains,Nthermalize=Nthermalize,Nstep=Nstep,N_PT=N_PT,reshape=reshape,update_betas=update_betas,config_init = config_init)


def gen_data_zeroT(model, which = 'marginal' ,Nchains=10,Lchains=100,Nthermalize=0,Nstep=1,N_PT=1,reshape=True,update_betas=False,config_init=[]):
    tmp_model = copy.deepcopy(model)
    if "class" in tmp_model._get_name():
        print("Zero Temp Generation of Data not available for classification CRBM")
    with torch.no_grad():
        if which == 'joint':
            tmp_model.markov_step = types.MethodType(markov_step_zeroT_joint, tmp_model)
        elif which == 'marginal':
            tmp_model.markov_step = types.MethodType(markov_step_zeroT_marginal, tmp_model)
        return tmp_model.gen_data(Nchains=Nchains,Lchains=Lchains,Nthermalize=Nthermalize,Nstep=Nstep,N_PT=N_PT,reshape=reshape,update_betas=update_betas,config_init = config_init)

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

