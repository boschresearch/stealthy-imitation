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


from easydict import EasyDict

halfcheetah_sac_config = dict(
    exp_name='halfcheetah_sac_seed0',
    env=dict(
        env_id='HalfCheetah-v3',
        norm_obs=dict(use_norm=False, ),
        norm_reward=dict(use_norm=False, ),
        collector_env_num=1,
        evaluator_env_num=8,
        n_evaluator_episode=8,
        stop_value=12000,
    ),
    policy=dict(
        cuda=True,
        random_collect_size=10000,
        model=dict(
            twin_critic=True,
            action_space='reparameterization',
            obs_shape=17,
            action_shape=6,
            actor_head_hidden_size=256,
            actor_head_layer_num=1,
            critic_head_hidden_size=256,
        ),
        learn=dict(
            update_per_collect=1,
            batch_size=256,
            learning_rate_q=1e-3,
            learning_rate_policy=1e-3,
            learning_rate_alpha=3e-4,
            ignore_done=True,
            target_theta=0.005,
            discount_factor=0.99,
            alpha=0.2,
            reparameterization=True,
            auto_alpha=False,
        ),
        collect=dict(
            n_sample=1,
            unroll_len=1,
        ),
        command=dict(),
        eval=dict(),
        other=dict(replay_buffer=dict(replay_buffer_size=1000000, ), ),
    ),
)

halfcheetah_sac_config = EasyDict(halfcheetah_sac_config)
main_config = halfcheetah_sac_config

halfcheetah_sac_create_config = dict(
    env=dict(
        type='mujoco',
        import_names=['dizoo.mujoco.envs.mujoco_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='sac',
        import_names=['ding.policy.sac'],
    ),
    replay_buffer=dict(type='naive', ),
)
halfcheetah_sac_create_config = EasyDict(halfcheetah_sac_create_config)
create_config = halfcheetah_sac_create_config