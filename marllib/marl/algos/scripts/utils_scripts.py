import json
from typing import Dict

from ray import tune
from ray.tune import CLIReporter

from marllib.marl.algos.utils.log_dir_util import available_local_dir


def generate_running_name(exp):
    algorithm = exp["algorithm"]
    map_name = exp["env_args"]["map_name"]
    arch = exp["model_arch_args"]["core_arch"]
    return '_'.join([algorithm, arch, map_name])


def restore_model(restore: Dict, exp: Dict):
    if restore is not None:
        with open(restore["params_path"], 'r') as JSON:
            raw_exp = json.load(JSON)
            raw_exp = raw_exp["model"]["custom_model_config"]['model_arch_args']
            check_exp = exp['model_arch_args']
            if check_exp != raw_exp:
                raise ValueError("is not using the params required by the checkpoint model")
        model_path = restore["model_path"]
    else:
        model_path = None

    return model_path


def run_tune_alg(name, config, run, exp, restore, stop, trainer):
    config.update(run)
    model_path = restore_model(restore, exp)
    results = tune.run(trainer,
                       name=name,
                       checkpoint_at_end=exp['checkpoint_end'],
                       checkpoint_freq=exp['checkpoint_freq'],
                       restore=model_path,
                       stop=stop,
                       config=config,
                       verbose=1,
                       trial_dirname_creator=exp.get('trial_dirname_creator'),
                       trial_name_creator=exp.get('trial_name_creator'),
                       progress_reporter=CLIReporter(),
                       local_dir=available_local_dir if exp["local_dir"] == "" else exp["local_dir"])
    return results
