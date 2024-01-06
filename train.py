import os
import argparse
import glob
from copy import copy
from numpy import linspace
from numpy.random import randint
import numpy as np

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer

from pool.utils.model_utils import load_run
from pool.utils.data_prep import generate_enrichment_fasta
from pool.analysis.analysis_methods import fetch_data
from pool.utils.io import dataframe_to_fasta
import pool.model


def get_model(model_type, config, debug_flag=False):
    return getattr(pool.model, model_type)(config, debug=debug_flag)


def get_trainer(config):
    logger = TensorBoardLogger(config["save_directory"], name=config["name"])
    if config["gpus"] > 1:
        # distributed data parallel, multi-gpus on single machine or across multiple machines
        return Trainer(max_epochs=config['epochs'], logger=logger, gpus=config["gpus"], accelerator="cuda",
                       strategy="ddp")
    else:
        if config['gpus'] == 0:
            return Trainer(max_epochs=config['epochs'], logger=logger, accelerator="cpu")
        else:
            return Trainer(max_epochs=config['epochs'], logger=logger, devices=config["gpus"], accelerator="cuda")


def train(config, debug_arg=None):
    debug_flag = False
    if debug_arg is not None:
        debug_flag = True
        config['gpus'] = 0
        if debug_arg == 'dg':
            config['gpus'] = 1

    if config["training_instances"] > 1:
        configs = parse_config_multiple_runs(config)
    else:
        configs = [config]

    for conf in configs:
        model = get_model(config["model_type"], conf, debug_flag=debug_flag)
        plt = get_trainer(config)
        plt.fit(model)


def parse_config_multiple_runs(config):
    # currently only supports single valued parameters!!!
    # assumes random search, may change later
    number = config["training_instances"]
    # mark which parameters are being altered
    config_diffs = {}
    for param in config['model'].keys():
        if type(config['model'][param]) is list:
            range_b, range_e = config['model'][param][:]
            config_diffs[param] = linspace(range_b, range_e, number).tolist()

    # make random selections of parameters
    # must be unique
    param_number = len(config.keys())
    param_configs = []
    while len(param_configs) < number:
        p_sel = []
        for p in range(param_number):
            p_sel.append(randint(0, p))
        if p_sel not in param_configs:
            param_configs.append(p_sel)

    final_configs = []
    for pid, pconf in enumerate(param_configs):
        instance_config = copy(config)
        for idx, key in enumerate(config_diffs.keys()):
            instance_config[key] = config_diffs[key][pconf[idx]]
        final_configs.append(instance_config)
    return final_configs

def iterative_enrichment_train(config, debug_arg=None):
    '''train model over course of selex experiment using the calculated common seqs and enrichment b/t each consecutive round'''
    rounds = config['rounds']

    directory = os.path.join(config['dataset_directory'], '')
    inter_round_num = len(rounds) - 1

    assert type(config['epochs']) is list
    assert type(config['lr']) is list

    # number of epochs for each
    inter_round_epochs = np.linspace(*config['epochs'], inter_round_num).tolist()
    inter_round_epochs = [int(x) for x in inter_round_epochs]
    # lr for each
    inter_round_lr = np.linspace(*config['lr'], inter_round_num).tolist()

    # make enrichment data_directory in parent directory of raw data
    data_dir = os.path.join(os.path.dirname(os.path.dirname(directory)), 'enrichment_data')
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

        # load data into dataframe
        raw_data = fetch_data(rounds, directory=directory, threads=6, alphabet='dna', normalize_counts=True)

        # generate data
        file_names = []
        for i in range(inter_round_num):
            file_names.append(generate_enrichment_fasta(raw_data, rounds[i], rounds[i+1], data_dir))

    else:
        file_names = glob.glob(os.path.join(data_dir, '*.fasta'))
        file_names = [os.path.basename(x) for x in file_names]  # drop
        # this ugly line sorts the filenames in ascending order according to the order given by the rounds list
        file_names.sort(key=lambda x: rounds.index(os.path.splitext(x)[0].split('_')[0]))


    model_home_dir = config['save_directory'] + config['name'] + '_iterative'
    config['save_directory'] = model_home_dir
    config['dataset_directory'] = data_dir

    config['name'] += '_-1'
    if debug_arg is not None:
        file_names = file_names[:2]
        inter_round_epochs = [1, 1]
        inter_round_num = 2

    for i in range(inter_round_num):
        config['name'] = config['name'].split('_')[0] + f'_{rounds[i + 1]}_{rounds[i]}'
        config['fasta_file'] = [file_names[i]]
        config['epochs'] = inter_round_epochs[i]
        config['lr'] = inter_round_lr[i]
        config['lr_final'] = config['lr']*1e-2

        if i == 0:
            model = get_model(config['model_type'], config)
            train(config, debug_arg=debug_arg)
        else:
            # change data, seqs and sampling weights
            model.fasta_file = [file_names[i]]
            model.reset_seq_weights()
            # loads new data and weights
            # model.setup()  # load new data
            model.lr = inter_round_lr[i]
            model.lr_final = model.lr*1e-2
            model.epochs = inter_round_epochs[i]

            trainer = get_trainer(config)
            trainer.fit(model)





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Runs Training Procedure from a json run file")
    parser.add_argument('runfile', type=str, help="json file containing all info needed to run models")
    parser.add_argument('-dg', type=str, nargs="?", help="debug flag for gpu, pass true to be able to inspect tensor values", default="False")
    parser.add_argument('-dc', type=str, nargs="?", help="debug flag for cpu, pass true to be able to inspect tensor values", default="False")
    parser.add_argument('-s', type=str, nargs="?", help="seed number", default=None)
    args = parser.parse_args()

    os.environ["SLURM_JOB_NAME"] = "bash"  # server runs crash without this line (yay raytune)
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1" # For debugging of cuda errors

    config = load_run(args.runfile)

    debug_arg = None
    if args.dg in ["true", "True"]:
        debug_arg = 'dg'

    if args.dc in ["true", "True"]:
        debug_arg = 'dc'

    if 'rounds' in config:
        iterative_enrichment_train(config, debug_arg=debug_arg)
    else:
        train(config, debug_arg=debug_arg)
