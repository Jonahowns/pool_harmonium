

# adapted from https://github.com/pytorch/pytorch/issues/7359
class WeightedSubsetRandomSampler:
    r"""Samples elements from a given list of indices with given probabilities (weights), with replacement.

    Arguments:
        weights (sequence)   : a 2ce of weights, not necessary summing up to one
        num_samples (int): number of samples to draw
    """

    def __init__(self, weights, labels, group_fraction, batch_size, batches, per_sample_replacement=False):
        if not isinstance(batch_size, int):
            raise ValueError("num_samples should be a non-negative integer "
                             "value, but got num_samples={}".format(batch_size))
        if not isinstance(batches, int):
            raise ValueError("num_samples should be a non-negative integer "
                             "value, but got num_samples={}".format(batches))

        self.batch_size = batch_size
        self.n_batches = batches

        self.weights = torch.tensor(weights, dtype=torch.double)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.indices = torch.arange(0, self.weights.shape[0], 1)

        self.replacement = per_sample_replacement  # can't think of a good reason to have this on

        self.label_set = list(set(labels))
        self.sample_per_label = [math.floor(x*self.batch_size) for x in group_fraction]
        # self.sample_per_label = self.num_samples // len(self.label_set)

        self.label_weights, self.label_indices = [], []
        for i in self.label_set:
            lm = labels == i
            self.label_indices.append(self.indices[lm])
            self.label_weights.append(self.weights[lm])

    def __iter__(self):
        batch_samples = []
        for i in range(self.n_batches):
            samples = []
            for j in range(len(self.label_set)):
                samples.append(self.label_indices[j][torch.multinomial(self.label_weights[j], self.sample_per_label[j], self.replacement)])

            batch_samples.append(torch.cat(samples, dim=0))

        for bs in batch_samples:
            yield bs

    def __len__(self):
        return self.n_batches


# from https://discuss.pytorch.org/t/how-to-enable-the-dataloader-to-sample-from-each-class-with-equal-probability/911/6
class StratifiedBatchSampler:
    """Stratified batch sampling
    Provides equal representation of target classes in each batch
    """
    def __init__(self, y, batch_size, shuffle=True, seed=38):
        if torch.is_tensor(y):
            y = y.cpu().numpy()
        assert len(y.shape) == 1, 'label array must be 1D'
        self.n_batches = int(len(y) / batch_size)
        self.skf = StratifiedKFold(n_splits=self.n_batches, shuffle=shuffle)
        self.X = torch.randn(len(y), 1).numpy()
        self.y = y
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        if self.shuffle:
            self.skf.random_state = self.seed  # should be governed by globally set seed

        for train_idx, test_idx in self.skf.split(self.X, self.y):
            yield test_idx

    def __len__(self):
        # return len(self.y)
        return self.n_batches


def label_samples(weights, label_spacing, label_groups):
    if type(label_spacing) is list:
        bin_edges = label_spacing
    else:
        if label_spacing == "log":
            bin_edges = np.geomspace(np.min(weights), np.max(weights), label_groups + 1)
        elif label_spacing == "lin":
            bin_edges = np.linspace(np.min(weights), np.max(weights), label_groups + 1)
        else:
            print(f"Label spacing option {label_spacing} not supported! Supported options are log, lin, "
                  f"and user provided spacing")
            exit(1)

    bin_edges = bin_edges[1:]

    def assign_label(x):
        bin_edge = bin_edges[0]
        idx = 0
        while x > bin_edge:
            idx += 1
            bin_edge = bin_edges[idx]

        return idx

    labels = list(map(assign_label, weights))
    return labels


class Categorical(Dataset):

    # Takes in pd dataframe with 2ces and weights of sequences (key: "sequences", weights: "sequence_count")
    # Also used to calculate the independent fields for parameter fields initialization
    def __init__(self, dataframe, q, weights=None, max_length=20, alphabet='protein', device='cpu',
                 one_hot=False, labels=False, additional_data=None):
        # Drop Duplicates/ Reset Index from most likely shuffled sequences
        # self.dataset = dataset.reset_index(drop=True).drop_duplicates("sequence")
        self.dataset = dataframe.reset_index(drop=True)

        # dictionaries mapping One letter code to integer for all macro molecule types
        if type(alphabet) is str:
            try:
                self.base_to_id = letter_to_int_dicts[alphabet]
            except:
                print(f"Molecule {alphabet} not supported. Please use protein, dna, or rna")
        elif type(alphabet) is dict:
            self.base_to_id = alphabet

        # number of possible bases
        self.n_bases = q

        # Makes sure everything is on correct device
        self.device = device

        # Number of Visible Nodes
        self.max_length = max_length

        # Set Sequence Data from dataframe
        self.seq_data = self.dataset.sequence.to_numpy()

        # Return one hot representation or ordinal? (T or F)
        self.oh = one_hot

        if self.oh:
            self.train_data = self.one_hot(self.categorical(self.seq_data))
        else:
            self.train_data = self.categorical(self.seq_data)

        # add anything else? labels or whatever
        self.additional_data = None
        if additional_data:
            self.additional_data = additional_data

        # default all equally weighted
        self.train_weights = np.asarray([1. for x in range(len(self))])
        if weights is not None:
            if type(weights) is list:
                self.train_weights = np.asarray(weights)
            elif type(weights) is np.ndarray:
                self.train_weights = weights

    def __getitem__(self, index):
        seq = self.seq_data[index]  # str of sequence
        model_input = self.train_data[index]  # either vector of integers for categorical or one hot vector
        weight = self.train_weights[index]

        return_arr = [index, seq, model_input, weight]

        if self.additional_data is not None:
            data = self.additional_data[index]
            return_arr.append(data)

        return return_arr

    def categorical(self, seq_dataset):
        return torch.tensor(list(map(lambda x: [self.base_to_id[y] for y in x], seq_dataset)), dtype=torch.long)

    def one_hot(self, cat_dataset):
        return F.one_hot(cat_dataset, num_classes=self.n_bases)

    def __len__(self):
        return self.train_data.shape[0]

    # def field_init(self):
    #     out = torch.zeros((self.max_length, self.n_bases), device=self.device)
    #     position_index = torch.arange(0, self.max_length, 1, device=self.device)
    #     if self.oh:
    #         cat_tensor = self.train_data.argmax(-1)
    #     else:
    #         cat_tensor = self.train_data
    #     for b in range(self.total):
    #         # out[position_index, cat_tensor[b]] += self.train_weights[b]
    #         out[position_index, cat_tensor[b]] += 1
    #     out.div_(self.total)  # in place
    #
    #     # invert softmax
    #     eps = 1e-6
    #     fields = torch.log((1 - eps) * out + eps / self.n_bases)
    #     fields -= fields.sum(1).unsqueeze(1) / self.n_bases
    #     return fields

    # def distance(self, MSA):
    #     B = MSA.shape[0]
    #     N = MSA.shape[1]
    #     distance = np.zeros([B, B])
    #     for b in range(B):
    #         distance[b] = ((MSA[b] != MSA).mean(1))
    #         distance[b, b] = 2.
    #     return distance
    #
    # def count_neighbours(self, MSA, threshold=0.1):  # Compute reweighting
    #     # works but is quite slow, should probably move this eventually
    #     # msa_long = MSA.long()
    #     B = MSA.shape[0]
    #     neighs = np.zeros((B,), dtype=float)
    #     for b in range(B):
    #         if self.oh:
    #             # pairwise_dist = torch.logical_and(MSA[b].unsqueeze(0))
    #             pairwise_dists = (MSA[b].unsqueeze(0) * MSA).sum(-1).sum(-1) / self.max_length
    #             neighs[b] = (pairwise_dists > threshold).float().sum().item()
    #         else:
    #             pairwise_dist = (MSA[b].unsqueeze(0) - MSA)
    #             dists = (pairwise_dist == 0).float().sum(1) / self.max_length
    #             neighs[b] = (dists > threshold).float().sum()
    #
    #         if b % 10000 == 0:
    #             print("Progress:", round(b/B, 3) * 100)
    #
    #     # N = MSA.shape[1]
    #     # num_neighbours = np.zeros(B)
    #     #
    #     # for b in range(B):
    #     #     num_neighbours[b] = 1 + ((MSA[b] != MSA).float().mean(1) < threshold).sum()
    #     return neighs



    # def on_epoch_end(self):
    #     self.count = 0
    #     if self.shuffle:
    #         self.dataset = self.dataset.sample(frac=1).reset_index(drop=True)

