# Copyright (C) 2024 Robert Bosch GmbH

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import argparse
import os
from ding.config import read_config, compile_config
from pipeline.serial_pipeline import serial_pipeline

def setup_config(args, cfg, create_cfg):
    """
    Set up the configuration for the experiment based on the provided arguments.

    This function modifies the provided configuration objects based on the command-line arguments. 
    It sets up the experiment name, policy type, and various policy parameters.

    Args:
        args: A Namespace object that contains the command-line arguments.
        cfg: A Config object that contains the current configuration.
        create_cfg: A Config object that contains the configuration to be created.

    Returns:
        A tuple containing the modified cfg and create_cfg objects.
    """
    # Set up experiment name
    cfg.exp_name = f"exp/{args.env}/{args.policy}_seed{args.seed}"
    if args.exp_name is not None:
        cfg.exp_name += f"_{args.exp_name}"
    if args.debug:
        cfg.exp_name += "_debug"

    create_cfg.policy.type = create_cfg.policy.type + '_command'

    # Set policy parameters
    if args.priority is not None:
        cfg.policy.priority = bool(args.priority)
        cfg.policy.priority_IS_weight = bool(args.priority)
    if args.batch_size is not None:
        cfg.policy.learn.batch_size = args.batch_size
    if args.learning_rate is not None:
        cfg.policy.learn.learning_rate = args.learning_rate
    if args.entropy_weight is not None:
        cfg.policy.learn.entropy_weight = args.entropy_weight
    if args.clip_ratio is not None:
        cfg.policy.learn.clip_ratio = args.clip_ratio
    if args.epoch_per_collect is not None:
        cfg.policy.learn.epoch_per_collect = args.epoch_per_collect
    if args.gae_lambda is not None:
        cfg.policy.collect.gae_lambda = args.gae_lambda
    if args.discount_factor is not None:
        cfg.policy.collect.discount_factor = args.discount_factor
    if args.bound_type is not None:
        cfg.policy.model.bound_type = args.bound_type
    if args.auto_alpha is not None:
        cfg.policy.learn.auto_alpha = bool(args.auto_alpha)
    if args.update_per_collect is not None:
        cfg.policy.learn.update_per_collect = args.update_per_collect
    if args.collect_n_sample is not None:
        cfg.policy.collect.n_sample = args.collect_n_sample

    cfg.policy.cuda = bool(args.cuda)
    if args.load_ckpt_before_run is not None:
        cfg.policy.learn.learner.hook.load_ckpt_before_run = args.load_ckpt_before_run

    return cfg, create_cfg

def main(args):
    """
    Main function to run the training pipeline.

    This function reads the configuration file based on the environment and policy name,
    sets up the configuration, compiles the configuration, and then runs the pipeline.

    Args:
        args: A Namespace object that contains the command-line arguments.
    """
    # Read config file by environment and policy name
    current_path = os.getcwd()
    config_path = os.path.join(
        current_path, f"config/{args.env}/{args.policy}_config.py")
    cfg, create_cfg = read_config(config_path)

    # Setup the config
    cfg, create_cfg = setup_config(args, cfg, create_cfg)

    # Compile the config
    cfg = compile_config(cfg, seed=args.seed, auto=True,
                         create_cfg=create_cfg, save_cfg=True)

    # Run the pipeline
    serial_pipeline(cfg, log_obs_boundary=args.log_obs_boundary)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a reinforcement learning model.")
    parser.add_argument("--debug", type=int, default=0,
                        help="Enable debug mode")
    parser.add_argument("--env", type=str, default="hopper",
                        help="Environment to use")
    parser.add_argument("--policy", type=str,
                        default="sac", help="Policy to use")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--exp_name", type=str, default=None,
                        help="Name of the experiment")
    parser.add_argument("--load_ckpt_before_run", type=str,
                        default=None, help="Load checkpoint before running")
    parser.add_argument("--cuda", type=int, default=1,
                        help="Use CUDA for computation")
    parser.add_argument("--log_obs_boundary", type=int,
                        default=0, help="Log observation boundary")
    parser.add_argument("--priority", type=int, default=None,
                        help="Priority of the policy")
    parser.add_argument("--batch_size", type=int,
                        default=None, help="Batch size for learning")
    parser.add_argument("--learning_rate", type=float,
                        default=None, help="Learning rate")
    parser.add_argument("--entropy_weight", type=float,
                        default=None, help="Weight for entropy")
    parser.add_argument("--clip_ratio", type=float,
                        default=None, help="Clip ratio for policy gradient")
    parser.add_argument("--gae_lambda", type=float, default=None,
                        help="Lambda for Generalized Advantage Estimation")
    parser.add_argument("--epoch_per_collect", type=int,
                        default=None, help="Number of epochs per data collection")
    parser.add_argument("--discount_factor", type=float,
                        default=None, help="Discount factor for rewards")
    parser.add_argument("--bound_type", type=str, default=None,
                        help="Type of boundary for the policy")
    parser.add_argument("--auto_alpha", type=int, default=None,
                        help="Automatically adjust alpha")
    parser.add_argument("--update_per_collect", type=int,
                        default=None, help="Number of updates per data collection")
    parser.add_argument("--collect_n_sample", type=int,
                        default=None, help="Number of samples to collect")

    args = parser.parse_args()

    main(args)
