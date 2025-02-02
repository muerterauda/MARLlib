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
from ray.rllib.utils.framework import try_import_torch
from ray.tune.analysis import ExperimentAnalysis
from ray.tune.utils import merge_dicts

from marllib.marl.algos.core.IL.trpo import TRPOTrainer
from marllib.marl.algos.scripts.utils_scripts import run_tune_alg, generate_running_name
from marllib.marl.algos.utils.setup_utils import AlgVar
from marllib.marl.algos.utils.trust_regions import TrustRegionUpdator

torch, nn = try_import_torch()


def run_itrpo(model: Any, exp: Dict, run: Dict, env: Dict,
              stop: Dict, restore: Dict) -> ExperimentAnalysis:
    """ This script runs the Independent Trust Region Policy Optimisation (ITPRO) algorithm using Ray RLlib.
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
    """
    for bug mentioned https://github.com/ray-project/ray/pull/20743
    make sure sgd_minibatch_size > max_seq_len
    """
    ModelCatalog.register_custom_model(
        "Base_Model", model)

    _param = AlgVar(exp)

    kl_threshold = _param['kl_threshold']
    accept_ratio = _param['accept_ratio']
    critic_lr = _param['critic_lr']

    TrustRegionUpdator.kl_threshold = kl_threshold
    TrustRegionUpdator.accept_ratio = accept_ratio
    TrustRegionUpdator.critic_lr = critic_lr

    train_batch_size = _param["batch_episode"] * env["episode_limit"]
    if "fixed_batch_timesteps" in exp:
        train_batch_size = exp["fixed_batch_timesteps"]
    sgd_minibatch_size = train_batch_size
    episode_limit = env["episode_limit"]
    while sgd_minibatch_size < episode_limit:
        sgd_minibatch_size *= 2

    batch_mode = _param["batch_mode"]
    clip_param = _param["clip_param"]
    grad_clip = _param["grad_clip"]
    use_gae = _param["use_gae"]
    gae_lambda = _param["lambda"]
    kl_coeff = _param["kl_coeff"]
    num_sgd_iter = _param["num_sgd_iter"]
    vf_loss_coeff = _param["vf_loss_coeff"]
    entropy_coeff = _param["entropy_coeff"]
    vf_clip_param = _param["vf_clip_param"]
    back_up_config = merge_dicts(exp, env)
    back_up_config.pop("algo_args")  # clean for grid_search

    config = {
        "batch_mode": batch_mode,
        "use_gae": use_gae,
        "lambda": gae_lambda,
        "kl_coeff": kl_coeff,
        "vf_loss_coeff": vf_loss_coeff,
        "entropy_coeff": entropy_coeff,
        "vf_clip_param": vf_clip_param,
        "num_sgd_iter": num_sgd_iter,
        "train_batch_size": train_batch_size,
        "sgd_minibatch_size": sgd_minibatch_size,
        "grad_clip": grad_clip,
        "clip_param": clip_param,
        "model": {
            "custom_model": "Base_Model",
            "max_seq_len": episode_limit,
            "custom_model_config": back_up_config,
        },
    }

    RUNNING_NAME = generate_running_name(exp)

    return run_tune_alg(RUNNING_NAME, config, run, exp, restore, stop, TRPOTrainer)
