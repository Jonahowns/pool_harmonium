import os
import math
import json
import sys
import torch
import pandas as pd
import numpy as np

from pytorch_lightning import LightningModule
from sklearn.model_selection import GroupShuffleSplit

from torch.utils.data import WeightedRandomSampler
from pool.utils.model_utils import process_weight_selection, weight_transform, configure_optimizer
from pool.dataset import StratifiedBatchSampler, WeightedSubsetRandomSampler, Categorical, label_samples
from pool.utils.io import fasta_read, fasta_read_basic


# class that takes care of generic methods for all models
class Base(LightningModule):
    def __init__(self, config, debug=False):
        super().__init__()

        mandatory_keys = ['seed',  'gpus', 'precision', 'batch_size', 'epochs', 'fasta_file', 'v_num', 'q', 'alphabet',
                          'data_worker_num', 'optimizer', 'lr', 'lr_final', 'weight_decay', 'decay_after', 'dr',
                          'sampling_strategy', 'validation_set_size', 'test_set_size', 'dataset_directory',
                          'sequence_weights_selection',  'sampling_weights_selection']
        for key in mandatory_keys:
            setattr(self, key, config[key])

        # basic options
        assert type(self.dr) is int or type(self.dr) is float
        assert type(self.batch_size) is int
        assert type(self.epochs) is int
        assert type(self.fasta_file) is list
        assert type(self.v_num) is int or type(self.v_num) is list
        assert type(self.q) is int
        assert type(self.alphabet) is str or type(self.alphabet) is dict
        assert type(self.data_worker_num) is int
        assert type(self.sequence_weights_selection) is dict or type(self.sequence_weights_selection) is str
        assert type(self.sampling_weights_selection) is dict or type(self.sampling_weights_selection) is str

        self.sequence_weights = process_weight_selection(self.sequence_weights_selection, self.dataset_directory)
        self.sampling_weights = process_weight_selection(self.sampling_weights_selection, self.dataset_directory)

        if type(self.v_num) is int:
            self.v_num = [self.v_num]

        # optimizer options
        ######################
        self.optimizer = configure_optimizer(self.optimizer)
        assert type(self.lr) is float or type(self.lr) is int
        assert type(self.lr_final) is float or self.lr_final == 'None' or type(self.lr) is int
        assert type(self.weight_decay) is float or type(self.lr) is int

        if self.lr_final == 'None':
            self.lr_final = self.lr * 1e-2

        # Dataset Options
        ####################
        assert self.validation_set_size < 1.0
        assert self.test_set_size < 1.0

        # Dataset Sampling
        ########################
        sampling_strategy_keys = {"random": [], "weighted": ["sampling_weights", "sample_multiplier"],
                                  "stratified": ["label_spacing", "label_fraction"],
                                  "stratified_weighted": ["sampling_weights", "sample_multiplier",
                                                          "label_spacing", "label_fraction"]}

        assert self.sampling_strategy in sampling_strategy_keys

        for key in sampling_strategy_keys[self.sampling_strategy]:
            setattr(self, key, config[key])

        # fill default for all
        for key in ["sampling_weights", "sample_multiplier", "label_spacing", "label_fraction"]:
            if not hasattr(self, key):
                if key == "label_spacing":
                    setattr(self, key, [0, 0])
                elif key == "label_fraction":
                    setattr(self, key, [1])
                elif key == "sampling_weights":
                    setattr(self, key, 'None')
                elif key == "sample_multiplier":
                    setattr(self, key, 0)

        self.label_groups = 1
        if "stratified" in self.sampling_strategy:
            self.label_groups = len(self.label_spacing) - 1
            assert len(self.label_fraction) == self.label_groups

        if "weighted" in self.sampling_strategy:
            assert self.sample_multiplier > 0

        # Pytorch Basic Options
        ################################
        assert type(self.seed) is int
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(self.seed)  # For reproducibility
        supported_precisions = {"double": torch.float64, "single": torch.float32}
        try:
            torch.set_default_dtype(supported_precisions[self.precision])
        except KeyError:
            print(f"Precision {self.precision} not supported.")
            sys.exit(-1)

        matmul_precisions = {"single": "medium", "double": "high"}
        torch.set_float32_matmul_precision(matmul_precisions[self.precision])

        if debug:
            self.data_worker_num = 0

        self.pin_mem = False
        if self.gpus > 0:
            self.pin_mem = True


        # Sequence Weighting
        # Not pretty but necessary to either provide the weights or to import from the fasta file
        # To import from the provided fasta file weights="fasta" in initialization of RBM

        self.training_data_logs = []
        self.val_data_logs = []

        self.training_data = None
        self.validation_data = None
        self.test_data = None
        self.dataset_indices = None

    def setup(self, stage=None):
        """Loads Data to be trained from provided fasta file. Splits into sets and manages weights."""
        data_pds = []

        if self.data_worker_num == 0:
            threads = 1
        else:
            threads = self.data_worker_num

        for file in self.fasta_file:
            filepath = os.path.join(self.dataset_directory, file)
            print(filepath)
            try:
                # seqs, seq_read_counts, all_chars, q_data = fasta_read(filepath,
                #                                                       self.alphabet, drop_duplicates=False,
                #                                                       threads=threads)

                seqs, seq_read_counts = fasta_read_basic(filepath, seq_read_counts=True, drop_duplicates=True)
                q_data = 5
                all_chars = []

            except IOError:
                print(f"Provided Fasta File '{filepath}' Not Found", file=sys.stderr)
                print(f"Current Directory '{os.getcwd()}'", file=sys.stderr)
                sys.exit(1)

            if q_data != self.q:
                print(
                    f"State Number mismatch! Expected q={self.q}, in dataset q={q_data}. "
                    f"All observed chars: {all_chars}", file=sys.stderr)
                sys.exit(1)

            data = pd.DataFrame(data={'sequence': seqs, 'fasta_count': seq_read_counts})

            if type(self.sequence_weights) == str and "fasta" in self.sequence_weights:
                weights = np.asarray(seq_read_counts)
                weights = weight_transform(weights, self.sequence_weights)
                data["seq_count"] = weights

            data_pds.append(data)

        all_data = pd.concat(data_pds)
        if type(self.sequence_weights) is np.ndarray:
            all_data["seq_count"] = self.sequence_weights

        print(all_data["sequence"][0], self.v_num)
        assert len(all_data["sequence"][0]) == np.prod(self.v_num)  # make sure v_num is same as data_length

        labels = 0  # default fill value
        if "stratified" in self.sampling_strategy:
            labels = label_samples(all_data["fasta_count"], self.label_spacing, self.label_groups)

        all_data["label"] = labels

        train_sets, val_sets, test_sets = [], [], []
        for i in range(self.label_groups):
            label_df = all_data[all_data["label"] == i]
            if self.test_set_size > 0.:
                # Split label df into train and test sets, taking into account duplicates
                not_test_inds, test_inds = next(GroupShuffleSplit(test_size=self.test_set_size, n_splits=1,
                                                                  random_state=self.seed).split(label_df, groups=label_df['sequence']))
                test_sets += label_df.index[test_inds].to_list()

                # Further split training set into train and test set
                train_inds, val_inds = next(GroupShuffleSplit(test_size=self.validation_set_size, n_splits=1,
                                                              random_state=self.seed).split(label_df.iloc[not_test_inds], groups=label_df.iloc[not_test_inds]['sequence']))
                train_sets += label_df.iloc[not_test_inds].index[train_inds].to_list()
                val_sets += label_df.iloc[not_test_inds].index[val_inds].to_list()

            else:
                # Split label df into train and validation sets, taking into account duplicates
                train_inds, val_inds = next(GroupShuffleSplit(test_size=self.validation_set_size, n_splits=1,
                                                              random_state=self.seed).split(label_df, groups=label_df['sequence']))
                train_sets += label_df.index[train_inds].to_list()
                val_sets += label_df.index[val_inds].to_list()

        self.training_data = all_data.iloc[train_sets]
        self.validation_data = all_data.iloc[val_sets]

        if "fasta" in self.sampling_weights:
            self.sampling_weights = weight_transform(torch.tensor(all_data["fasta_count"].values), self.sampling_weights)
            print(self.sampling_weights)

        if self.sampling_weights is not None:
            self.sampling_weights = self.sampling_weights[train_sets]

        self.dataset_indices = {"train_indices": train_sets, "val_indices": val_sets}
        if self.test_set_size > 0:
            self.test_data = all_data.iloc[test_sets]
            self.dataset_indices["test_indices"] = test_sets

    def on_train_start(self):
        # Log which sequences belong to each dataset, same order as fasta file
        with open(self.logger.log_dir + "/dataset_indices.json", "w") as f:
            json.dump(self.dataset_indices, f)

    # Sets Up Optimizer as well as Exponential Weight Decay
    def configure_optimizers(self):
        optim = self.optimizer(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # Exponential Weight Decay after set amount of epochs (set by decay_after)
        decay_gamma = (self.lr_final / self.lr) ** (1 / (self.epochs * (1 - self.decay_after)))
        decay_milestone = math.floor(self.decay_after * self.epochs)
        my_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optim,
                                                               milestones=[decay_milestone], gamma=decay_gamma)
        optim_dict = {"lr_scheduler": my_lr_scheduler,
                      "optimizer": optim}
        return optim_dict

    # Loads Training Data
    def train_dataloader(self, init_fields=False):
        training_weights = None
        if "seq_count" in self.training_data.columns:
            training_weights = self.training_data["seq_count"].tolist()

        train_reader = Categorical(self.training_data, self.q, weights=training_weights, max_length=self.v_num,
                                   alphabet=self.alphabet, device=self.device, one_hot=True)
        # Init Fields
        if init_fields:
            if hasattr(self, "fields"):
                with torch.no_grad():
                    initial_fields = train_reader.field_init()
                    self.fields += initial_fields
                    self.fields0 += initial_fields

        else:
            if hasattr(self, "fields"):
                with torch.no_grad():
                    initial_fields = torch.randn((*self.v_num, self.q), device=self.device)*0.01
                    self.fields += initial_fields
                    self.fields0 += initial_fields

        # Sampling
        if self.sampling_strategy == "stratified":
            return torch.utils.data.DataLoader(
                train_reader,
                batch_sampler=StratifiedBatchSampler(self.training_data["label"].to_numpy(), batch_size=self.batch_size,
                                                     shuffle=True, seed=self.seed),
                num_workers=self.data_worker_num,
                pin_memory=self.pin_mem
            )
        elif self.sampling_strategy == "weighted":
            return torch.utils.data.DataLoader(
                train_reader,
                sampler=WeightedRandomSampler(weights=self.sampling_weights,
                                              num_samples=self.batch_size * self.sample_multiplier, replacement=True),
                num_workers=self.data_worker_num,  # Set to 0 if debug = True
                batch_size=self.batch_size,
                pin_memory=self.pin_mem
            )
        elif self.sampling_strategy == "stratified_weighted":
            return torch.utils.data.DataLoader(
                train_reader,
                batch_sampler=WeightedSubsetRandomSampler(self.sampling_weights, self.training_data["label"].to_numpy(),
                                                          self.label_fraction, self.batch_size, self.sample_multiplier),
                num_workers=self.data_worker_num,
                pin_memory=self.pin_mem
            )
        else:
            self.sampling_strategy = "random"
            return torch.utils.data.DataLoader(
                train_reader,
                batch_size=self.batch_size,
                num_workers=self.data_worker_num,
                pin_memory=self.pin_mem,
                shuffle=True
            )

    def val_dataloader(self):
        # Get Correct Validation weights
        validation_weights = None
        if "seq_count" in self.validation_data.columns:
            validation_weights = self.validation_data["seq_count"].tolist()

        val_reader = Categorical(self.validation_data, self.q, weights=validation_weights, max_length=self.v_num,
                                 alphabet=self.alphabet, device=self.device, one_hot=True, additional_data=None)

        if self.sampling_strategy == "stratified":
            return torch.utils.data.DataLoader(
                val_reader,
                batch_sampler=StratifiedBatchSampler(self.validation_data["label"].to_numpy(),
                                                     batch_size=self.batch_size, shuffle=False),
                num_workers=self.data_worker_num,  # Set to 0 if debug = True
                pin_memory=self.pin_mem
            )
        else:
            return torch.utils.data.DataLoader(
                val_reader,
                batch_size=self.batch_size,
                num_workers=self.data_worker_num,  # Set to 0 to view tensors while debugging
                pin_memory=self.pin_mem,
                shuffle=False
            )

    def on_validation_epoch_end(self):
        # log values to tensorboard
        result_dict = {}
        for key in self.val_data_logs[0].keys():
            result_dict[key] = torch.stack([x[key] for x in self.val_data_logs]).mean()

        self.logger.experiment.add_scalars("Val Scalars", result_dict, self.current_epoch)

        self.val_data_logs.clear()

    def on_train_epoch_end(self):
        """On Epoch End Collects Scalar Statistics and Distributions of Parameters for the Tensorboard Logger"""
        result_dict = {}
        for key in self.training_data_logs[0].keys():
            if key == "loss":
                result_dict[key] = torch.stack([x[key].detach() for x in self.training_data_logs]).mean()
            else:
                try:
                    result_dict[key] = torch.stack([x[key] for x in self.training_data_logs]).mean()
                except RuntimeError:
                    print('Logging Problems in "on_train_epoch_end"')

        # log values to tensorboard
        self.logger.experiment.add_scalars("Train Scalars", result_dict, self.current_epoch)

        # log model parameters
        for name, p in self.named_parameters():
            self.logger.experiment.add_histogram(name, p.detach(), self.current_epoch)

        self.training_data_logs.clear()

    def get_param(self, param_name):
        """Return param as a numpy array"""
        try:
            tensor = getattr(self, param_name).clone()
            return tensor.cpu().detach().numpy()
        except KeyError:
            print(f"Key {param_name} not found")
            sys.exit(1)


class BaseRelu(Base):
    """class that extends base class for any of our rbm/crbm models"""
    def __init__(self, config, debug=False):

        super().__init__(config, debug=debug)

        # Constants for faster math
        self.logsqrtpiover2 = torch.tensor(0.2257913526, device=self.device, requires_grad=False)
        self.pbis = torch.tensor(0.332672, device=self.device, requires_grad=False)
        self.a1 = torch.tensor(0.3480242, device=self.device, requires_grad=False)
        self.a2 = torch.tensor(- 0.0958798, device=self.device, requires_grad=False)
        self.a3 = torch.tensor(0.7478556, device=self.device, requires_grad=False)
        self.invsqrt2 = torch.tensor(0.7071067812, device=self.device, requires_grad=False)
        self.sqrt2 = torch.tensor(1.4142135624, device=self.device, requires_grad=False)

    def erf_times_gauss(self, X):
        """This is the "characteristic" function phi, used for ReLU and dReLU"""
        m = torch.zeros_like(X, device=self.device)
        tmp1 = X < -6
        m[tmp1] = 2 * torch.exp(X[tmp1] ** 2 / 2)

        tmp2 = X > 0
        t = 1 / (1 + self.pbis * X[tmp2])
        m[tmp2] = t * (self.a1 + self.a2 * t + self.a3 * t ** 2)

        tmp3 = torch.logical_and(~tmp1, ~tmp2)
        t2 = 1 / (1 - self.pbis * X[tmp3])
        m[tmp3] = -t2 * (self.a1 + self.a2 * t2 + self.a3 * t2 ** 2) + 2 * torch.exp(X[tmp3] ** 2 / 2)
        return m

    def log_erf_times_gauss(self, X):
        """Used in Cumulant Generating Function for ReLU and dReLU """
        m = torch.zeros_like(X, device=self.device)
        tmp = X < 4
        m[tmp] = 0.5 * X[tmp] ** 2 + torch.log(1 - torch.erf(X[tmp] / self.sqrt2)) + self.logsqrtpiover2
        m[~tmp] = - torch.log(X[~tmp]) + torch.log(1 - 1 / X[~tmp] ** 2 + 3 / X[~tmp] ** 4)
        return m
