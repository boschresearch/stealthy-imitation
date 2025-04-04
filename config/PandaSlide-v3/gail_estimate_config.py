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

# for panda, the environment information is not used, but only the policy and reward model
obs_shape = 24
act_shape = 3
exp_name='exp/hopper/gail_estimate_seed4'
main_config = EasyDict(
    dict(
        exp_name=exp_name,
        env=dict(
            env_id='Hopper-v3',
            norm_obs=dict(use_norm=False, ),
            norm_reward=dict(use_norm=False, ),
            collector_env_num=1,
            evaluator_env_num=8,
            n_evaluator_episode=8,
            stop_value=4000,
        ),
        reward_model=dict(
            input_size=obs_shape + act_shape,
            hidden_size=256,
            target_new_data_count=64,
            batch_size=64,
            learning_rate=1e-3,
            update_per_collect=100,
            collect_count=100000,
        ),
        policy=dict(
            model=dict(
                twin_critic=True,
                action_space='reparameterization',
                obs_shape=obs_shape,
                action_shape=3,
                actor_head_hidden_size=256,
                critic_head_hidden_size=256,
            ),
            learn=dict(
                learner=dict(
                    train_iterations=1000000000,
                    dataloader=dict(
                        num_workers=0,
                    ),
                    log_policy=True,
                    hook=dict(
                        load_ckpt_before_run='',
                        log_show_after_iter=100,
                        save_ckpt_after_iter=10000,
                        save_ckpt_after_run=True,
                    ),
                    cfg_type='BaseLearnerDict',
                ),
                multi_gpu=False,
                update_per_collect=1,
                batch_size=256,
                learning_rate_q=0.001,
                learning_rate_policy=0.001,
                learning_rate_value=0.0003,
                learning_rate_alpha=0.0003,
                target_theta=0.005,
                discount_factor=0.99,
                alpha=0.2,
                auto_alpha=False,
                log_space=True,
                ignore_done=False,
                init_w=0.003,
                reparameterization=True,
            ),
            collect=dict(
                collector=dict(
                    deepcopy_obs=False,
                    transform_obs=False,
                    collect_print_freq=100,
                    cfg_type='SampleSerialCollectorDict',
                    type='sample',
                ),
                collector_logit=False,
                n_sample=64,
                unroll_len=1,
            ),
            eval=dict(
                evaluator=dict(
                    eval_freq=500,
                    render={'render_freq': -1, 'mode': 'train_iter'},
                    cfg_type='InteractionSerialEvaluatorDict',
                    n_episode=8,
                    stop_value=4000,
                ),
            ),
            other=dict(
                replay_buffer=dict(
                    type='naive',
                    replay_buffer_size=1000000,
                    deepcopy=False,
                    enable_track_used_data=False,
                    periodic_thruput_seconds=60,
                    cfg_type='NaiveReplayBufferDict',
                ),
            ),
            cuda=True,
            on_policy=False,
            priority=False,
            priority_IS_weight=False,
            random_collect_size=25000,
            transition_with_policy_data=True,
            multi_agent=False,
            cfg_type='SACCommandModePolicyDict',
            command={},
        ),
    )
)


create_config = EasyDict(
    dict(
        env=dict(
            type='mujoco',
            import_names=['dizoo.mujoco.envs.mujoco_env'],
        ),
        env_manager=dict(type='subprocess'),
        policy=dict(
            type='sac',
        ),
        replay_buffer=dict(type='naive', ),
        reward_model=dict(type='custom_gail',import_names=['model.custom_gail']),
    )
)

