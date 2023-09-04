import os
import argparse

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer

from pool.utils.model_utils import load_run
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


# callable from jupyter notebooks etc.
def train(config, debug_arg=None):
    debug_flag = False
    if debug_arg is not None:
        debug_flag = True
        config['gpus'] = 0
        if debug_arg == 'dg':
            config['gpus'] = 1
    model = get_model(config["model_type"], config, debug_flag=debug_flag)
    if debug_flag:
        config['gpus']
    plt = get_trainer(config)
    plt.fit(model)



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

    train(config, debug_arg=debug_arg)

    # debug_flag = False
    # if args.dg in ["true", "True"]:
    #     debug_flag = True
    #     config['gpus'] = 1
    #
    # if args.dc in ["true", "True"]:
    #     debug_flag = True
    #     config["gpus"] = 0
    #
    # # Training Code
    # model = get_model(config["model_type"], config, debug_flag=debug_flag)
    # plt = get_trainer(config)
    # plt.fit(model)
