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

import sys
import argparse
from functools import partial

import torch
from ding.config import read_config, compile_config
from ding.envs import get_vec_env_setting, create_env_manager
from ding.policy import create_policy
from ding.policy.common_utils import default_preprocess_learn
from ding.utils import set_pkg_seed
from ding.worker import create_serial_collector, create_serial_evaluator
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import logging
logger = logging.getLogger(__name__)

def load_policy(args):
    """
    Load the model based on the provided configuration and arguments.

    Args:
        args: Command line arguments.

    Returns:
        The loaded model.
    """
    policy_path = f"{args.policy_path}/formatted_total_config.py"
    seed = args.seed

    cfg, create_cfg = read_config(policy_path)

    create_cfg.policy.type = create_cfg.policy.type + '_command'
    cfg.policy.collect.n_sample = args.n_sample
    cfg.policy.collect.collector_logit = True
    cfg = compile_config(cfg, seed=seed, env=None, auto=True, create_cfg=create_cfg, save_cfg=False)
    set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)

    policy = create_policy(cfg.policy, model=None, enable_field=['eval', 'learn', 'collect', 'command'])
    if args.attacker:
        model_path = f"{args.policy_path}/ckpt/best_val_loss.pth.tar"
    else:
        model_path = f"{args.policy_path}/ckpt/ckpt_best.pth.tar"
    policy._model.load_state_dict(torch.load(model_path, map_location='cpu')['model'], strict=True)

    return policy, cfg


def create_collector_evaluator(args, cfg, policy):
    """
    Creates an evaluation environment and a collector environment for the given policy.

    This function sets up the environments based on the provided configuration and policy. 
    It also sets up a serial evaluator for the evaluation environment and a serial collector for the collector environment.
    If the 'record_video' argument is set, it configures the evaluator environment to save a replay video.

    Args:
        args (argparse.Namespace): Parsed command line arguments.
        cfg (dict): Configuration dictionary for the model and environment.
        policy (Policy): The policy to be evaluated.

    Returns:
        collector (SerialCollector): A serial collector for the collector environment.
        evaluator (SerialEvaluator): A serial evaluator for the evaluation environment.
    """
    if args.record_video:
        cfg.env.evaluator_env_num = 1
        cfg.env.n_evaluator_episode = 1
        cfg.policy.eval.evaluator.n_episode = 1
    env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)
    evaluator_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg])
    if args.record_video:
        evaluator_env.enable_save_replay(f"{args.policy_path}/video")
    evaluator_env.seed(cfg.seed, dynamic_seed=False)
    evaluator = create_serial_evaluator(
        cfg.policy.eval.evaluator,
        env=evaluator_env,
        policy=policy.eval_mode)
    
    collector_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in collector_env_cfg])
    collector_env.seed(cfg.seed)
    collector = create_serial_collector(
        cfg.policy.collect.collector,
        env=collector_env,
        policy=policy.collect_mode
    )
    return collector, evaluator


def main(args):
    """
    Main function to load the model, create the evaluation environment, and perform the analysis.

    Args:
        args: Command line arguments.
    """
    policy, cfg = load_policy(args)
    collector, evaluator = create_collector_evaluator(args, cfg, policy)

    # Configure logging to output DEBUG and higher level messages to both a file and the console
    logging.basicConfig(filename=f'{args.policy_path}/logger.log', filemode='a', level=logging.DEBUG, force=True)
    console_handler = logging.StreamHandler(sys.stdout)
    logging.getLogger().addHandler(console_handler)

    policy._model.eval()

    with torch.no_grad():
        stop, eval_info = evaluator.eval(None, 0, 0)


    logger.info(f"Episode Reward mean = {torch.tensor(eval_info['eval_episode_return']).mean().item():.2f}")
    logger.info(f"Episode Reward std = {torch.tensor(eval_info['eval_episode_return']).std().item():.2f}")
    logger.info(f"Episode Reward max = {torch.tensor(eval_info['eval_episode_return']).max().item():.2f}")
    logger.info(f"Episode Reward min = {torch.tensor(eval_info['eval_episode_return']).min().item():.2f}")


    if args.record_video:
        return
    

    new_data = collector.collect(train_iter=0)
    data = default_preprocess_learn(new_data)
    def tensor_to_string(tensor):
        return ', '.join(f"{value:.2f}" for value in tensor)
    

    obs_data = data['obs']
    if args.visualize:
        # Pair plot
        obs_data_df = pd.DataFrame(obs_data.numpy())
        pair_plot = sns.pairplot(obs_data_df,kind="kde")
        pair_plot.savefig(f'{args.policy_path}/pairplot_kde_{args.n_sample}.png')
        plt.close()
        return
    # Covariance
    obs_data_mean_subtracted = obs_data - obs_data.mean(dim=0)
    covariance = obs_data_mean_subtracted.t().mm(obs_data_mean_subtracted) / (obs_data.shape[0] - 1)

    # Save mean, covariance, PCA model and transformed data
    torch.save({
        'obs_data':obs_data,
        'mean': obs_data.mean(dim=0),
        'std':obs_data.std(dim=0),
        'covariance': covariance
    }, args.policy_path+f'/obs_data_{args.n_sample}.pt')

    logger.info("covariance")
    logger.info(covariance)

    mu=data['logit'][0]
    sigma = data['logit'][0]


    logger.info(f"policy_logit_mu_mean: [{tensor_to_string(mu.mean(dim=0))}]")
    logger.info(f"policy_logit_mu_max: [{tensor_to_string(mu.max(dim=0)[0])}]")
    logger.info(f"policy_logit_mu_min: [{tensor_to_string(mu.min(dim=0)[0])}]")


    logger.info(f"policy_logit_sigma_mean: [{tensor_to_string(sigma.mean(dim=0))}]")
    logger.info(f"policy_logit_sigma_max: [{tensor_to_string(sigma.max(dim=0)[0])}]")
    logger.info(f"policy_logit_sigma_min: [{tensor_to_string(sigma.min(dim=0)[0])}]")

    logger.info(f"policy_action_action_mean: [{tensor_to_string(data['action'].mean(dim=0))}]")
    logger.info(f"policy_action_action_std: [{tensor_to_string(data['action'].std(dim=0))}]")
    logger.info(f"policy_action_action_max: [{tensor_to_string(data['action'].max(dim=0)[0])}]")
    logger.info(f"policy_action_action_min: [{tensor_to_string(data['action'].min(dim=0)[0])}]")

    logger.info(f"obs_state_std: [{tensor_to_string(data['obs'].std(dim=0))}]")
    logger.info(f"obs_state_mean: [{tensor_to_string(data['obs'].mean(dim=0))}]")
    logger.info(f"obs_state_min: [{tensor_to_string(data['obs'].min(dim=0)[0])}]")
    logger.info(f"obs_state_max: [{tensor_to_string(data['obs'].max(dim=0)[0])}]")

    logger.info(f"reward_mean: {data['reward'].mean():.2f}")
    logger.info(f"reward_max: {data['reward'].max():.2f}")
    logger.info(f"reward_min: {data['reward'].min():.2f}")

    collector.close()
    evaluator.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This script is used to analyze a policy in a reinforcement learning environment.")

    parser.add_argument("--policy_path", type=str, default="exp/hopper/sac_seed0",
                        help="Path to the directory where the policy and its configuration are stored.")

    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for random number generation to ensure reproducibility.")

    parser.add_argument("--n_sample", type=int, default=100,
                        help="Number of samples to collect for analysis.")

    parser.add_argument("--visualize", type=int, default=0,
                        help="Set to 1 to visualize the obs_data in kde, 0 otherwise.")

    parser.add_argument("--record_video", type=int, default=0,
                        help="Set to 1 to record a video of the policy's behavior, 0 otherwise.")

    parser.add_argument("--attacker", type=int, default=0,
                        help="Set to 1 if the policy is for an attacker, 0 otherwise.")

    args = parser.parse_args()

    main(args)