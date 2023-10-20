# MIT License

# Copyright (c) 2023 Replicable-MARL

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import Any, Dict

from ray.rllib.models import ModelCatalog
from ray.tune.analysis import ExperimentAnalysis
from ray.tune.utils import merge_dicts

from marllib.marl.algos.core.VD.vda2c import VDA2CTrainer
from marllib.marl.algos.scripts.utils_scripts import run_tune_alg, generate_running_name
from marllib.marl.algos.utils.setup_utils import AlgVar


def run_vda2c(model: Any, exp: Dict, run: Dict, env: Dict,
             stop: Dict, restore: Dict) -> ExperimentAnalysis:
    """ This script runs the Value Decomposition Advantage Actor-Critic (VDA2C) algorithm using Ray RLlib.
    Args:
        :params model (str): The name of the model class to register.
        :params exp (dict): A dictionary containing all the learning settings.
        :params run (dict): A dictionary containing all the environment-related settings.
        :params env (dict): A dictionary specifying the condition for stopping the training.
        :params restore (bool): A flag indicating whether to restore training/rendering or not.

    Returns:
        ExperimentAnalysis: Object for experiment analysis.

    Raises:
        TuneError: Any trials failed and `raise_on_failed_trial` is True.
    """

    ModelCatalog.register_custom_model(
        "Value_Decomposition_Model", model)

    _param = AlgVar(exp)

    train_batch_size = _param["batch_episode"] * env["episode_limit"]
    if "fixed_batch_timesteps" in exp:
        train_batch_size = exp["fixed_batch_timesteps"]
    episode_limit = env["episode_limit"]

    batch_mode = _param["batch_mode"]
    lr = _param["lr"]
    use_gae = _param["use_gae"]
    gae_lambda = _param["lambda"]
    vf_loss_coeff = _param["vf_loss_coeff"]
    entropy_coeff = _param["entropy_coeff"]
    back_up_config = merge_dicts(exp, env)
    back_up_config.pop("algo_args")  # clean for grid_search

    config = {
        "train_batch_size": train_batch_size,
        "batch_mode": batch_mode,
        "use_gae": use_gae,
        "lambda": gae_lambda,
        "vf_loss_coeff": vf_loss_coeff,
        "entropy_coeff": entropy_coeff,
        "lr": lr if restore is None else 1e-10,
        "model": {
            "custom_model": "Value_Decomposition_Model",
            "max_seq_len": episode_limit,
            "custom_model_config": back_up_config,
        },
    }

    RUNNING_NAME = generate_running_name(exp)

    return run_tune_alg(RUNNING_NAME, config, run, exp, restore, stop, VDA2CTrainer)
