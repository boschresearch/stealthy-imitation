# Copyright 2024 Robert Bosch GmbH
# Copyright 2021 OpenDILab Contributors.
#
# Parts of the script below are adapted from [DI-engine](https://github.com/opendilab/DI-engine)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Optional, List, Any
import time
import pickle
from functools import partial
import numpy as np
import torch
from ditk import logging
from tensorboardX import SummaryWriter
from ding.envs import get_vec_env_setting, create_env_manager
from ding.worker import BaseLearner,BaseSerialCommander, create_buffer, \
    create_serial_collector, create_serial_evaluator
from ding.policy import create_policy
from ding.utils import set_pkg_seed
from ding.entry.utils import random_collect
from ding.policy.common_utils import default_preprocess_learn
from ding.policy import Policy
from easydict import EasyDict

def serial_pipeline(
        cfg: EasyDict,
        env_setting: Optional[List[Any]] = None,
        model: Optional[torch.nn.Module] = None,
        max_train_iter: Optional[int] = int(1e10),
        max_env_step: Optional[int] = int(1e10),
        log_obs_boundary: int=0
) -> Policy:
    """
    Overview:
        Serial pipeline entry for off-policy RL.
    Arguments:
        - input_cfg (:obj:`Union[str, Tuple[dict, dict]]`): Config in dict type. \
            ``str`` type means config file path. \
            ``Tuple[dict, dict]`` type means [user_config, create_cfg].
        - seed (:obj:`int`): Random seed.
        - env_setting (:obj:`Optional[List[Any]]`): A list with 3 elements: \
            ``BaseEnv`` subclass, collector env config, and evaluator env config.
        - model (:obj:`Optional[torch.nn.Module]`): Instance of torch.nn.Module.
        - max_train_iter (:obj:`Optional[int]`): Maximum policy update iterations in training.
        - max_env_step (:obj:`Optional[int]`): Maximum collected environment interaction steps.
    Returns:
        - policy (:obj:`Policy`): Converged policy.
    """

    if env_setting is None:
        env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)
    else:
        env_fn, collector_env_cfg, evaluator_env_cfg = env_setting
    collector_env = create_env_manager(cfg.env.manager,
                                       [partial(env_fn, cfg=c) for c in collector_env_cfg])
    evaluator_env = create_env_manager(cfg.env.manager,
                                       [partial(env_fn, cfg=c) for c in evaluator_env_cfg])
    collector_env.seed(cfg.seed)
    evaluator_env.seed(cfg.seed, dynamic_seed=False)
    set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)
    policy = create_policy(cfg.policy, model=model,
                           enable_field=['learn', 'collect', 'eval', 'command'])

    # Initialize the running max and min tensors.
    # We assume that `cfg.policy.obs_shape` is the shape of the observation.
    # You may need to adjust this according to your actual observation shape.
    if log_obs_boundary:
        running_max_obs = torch.full((cfg.policy.model.obs_shape,), float('-inf'))
        running_min_obs = torch.full((cfg.policy.model.obs_shape,), float('inf'))

    # Create worker components: 
    # learner, collector, evaluator, replay buffer, commander.
    tb_logger = SummaryWriter(f'./{cfg.exp_name}/log/serial')
    learner = BaseLearner(cfg.policy.learn.learner,
                          policy.learn_mode,
                          tb_logger,
                          exp_name=cfg.exp_name)
    collector = create_serial_collector(
        cfg.policy.collect.collector,
        env=collector_env,
        policy=policy.collect_mode,
        tb_logger=tb_logger,
        exp_name=cfg.exp_name
    )
    evaluator = create_serial_evaluator(
        cfg.policy.eval.evaluator,
        env=evaluator_env,
        policy=policy.eval_mode,
        tb_logger=tb_logger,
        exp_name=cfg.exp_name
    )
    replay_buffer = create_buffer(cfg.policy.other.replay_buffer,
                                  tb_logger=tb_logger, exp_name=cfg.exp_name)
    commander = BaseSerialCommander(
        cfg.policy.other.commander,
        learner,
        collector,
        evaluator,
        replay_buffer,
        policy.command_mode
    )
    # ==========
    # Main loop
    # ==========
    # Learner's before_run hook.
    learner.call_hook('before_run')

    # Accumulate plenty of data at the beginning of training.
    if cfg.policy.get('random_collect_size', 0) > 0:
        random_collect(cfg.policy, policy, collector, collector_env, commander, replay_buffer)
    while True:
        collect_kwargs = commander.step()
        # Evaluate policy performance
        if evaluator.should_eval(learner.train_iter):
            stop, eval_info = evaluator.eval(learner.save_checkpoint,
                                             learner.train_iter,
                                             collector.envstep)
            if stop:
                break
        # Collect data by default config n_sample/n_episode
        new_data = collector.collect(train_iter=learner.train_iter, policy_kwargs=collect_kwargs)
        replay_buffer.push(new_data, cur_collector_envstep=collector.envstep)
        # Learn policy from collected data
        for i in range(cfg.policy.learn.update_per_collect):
            # Learner will train ``update_per_collect`` times in one iteration.
            train_data = replay_buffer.sample(learner.policy.get_attribute('batch_size'),
                                              learner.train_iter)
            if train_data is None:
                # It is possible that replay buffer's data count is too few to
                # train ``update_per_collect`` times
                logging.warning(
                    f"Replay buffer's data can only train for {i} steps. " +
                    "You can modify data collect config, e.g. increasing n_sample, n_episode."
                )
                break

            if log_obs_boundary:

                test_data = default_preprocess_learn(
                    train_data
                )
                # Update the running max and min
                max_obs = test_data['obs'].max(dim=0)[0]
                min_obs = test_data['obs'].min(dim=0)[0]
                running_max_obs = torch.max(running_max_obs, max_obs)
                running_min_obs = torch.min(running_min_obs, min_obs)

                # Log each variable's max and min value
                for i, (max_obs, min_obs) in enumerate(zip(running_max_obs, running_min_obs)):
                    tb_logger.add_scalar(f'obs_boundary/max_var_{i+1}',
                                         max_obs.item(), learner.train_iter)
                    tb_logger.add_scalar(f'obs_boundary/min_var_{i+1}',
                                         min_obs.item(), learner.train_iter)

            learner.train(train_data, collector.envstep)
            if learner.policy.get_attribute('priority'):
                replay_buffer.update(learner.priority_info)
        if collector.envstep >= max_env_step or learner.train_iter >= max_train_iter:
            break

    # Learner's after_run hook.
    learner.call_hook('after_run')

    with open(os.path.join(cfg.exp_name, 'result.pkl'), 'wb') as f:
        eval_value_raw = [d['eval_episode_return'] for d in eval_info]
        final_data = {
            'stop': stop,
            'env_step': collector.envstep,
            'train_iter': learner.train_iter,
            'eval_value': np.mean(eval_value_raw),
            'eval_value_raw': eval_value_raw,
            'finish_time': time.ctime(),
        }
        pickle.dump(final_data, f)
    return policy
