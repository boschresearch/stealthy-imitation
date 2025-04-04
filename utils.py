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

from datetime import datetime
from functools import partial
import io
import os

from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from PIL import Image
import torch
from torch.distributions import Independent, Normal, Distribution
from torchvision.transforms import ToTensor
from tensorboardX import SummaryWriter

from ding.config import read_config, compile_config
from ding.envs import get_vec_env_setting, create_env_manager
from ding.policy import create_policy
from ding.utils import set_pkg_seed
from ding.worker import create_serial_evaluator

def create_evaluator_and_logger(cfg, adversary_policy):
    """
    Create an evaluator and a logger for the given configuration and adversary policy.

    Parameters:
    cfg (Config): The configuration object containing the experiment settings.
    adversary_policy (Policy): The policy object for the adversary.

    Returns:
    tb_logger (SummaryWriter): The tensorboard logger for the experiment.
    evaluator (Evaluator): The evaluator object for the experiment.
    """
    tb_logger = SummaryWriter(f'./{cfg.exp_name}/log/serial')
    env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)
    evaluator_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg])
    evaluator_env.seed(cfg.seed, dynamic_seed=False)
    evaluator = create_serial_evaluator(
        cfg.policy.eval.evaluator,
        env=evaluator_env,
        policy=adversary_policy.eval_mode,
        tb_logger=tb_logger,
        exp_name=cfg.exp_name
    )

    return tb_logger, evaluator
    
def construct_transfer_dataset(global_budget, vic_model,query_budget, estimate_dist,args, train_val_ratio=0.8,sampler=None,cfg=None):
    # Increase the global budget by the query budget
    global_budget += query_budget

    # If no sampler is provided, sample from the estimate distribution
    # Otherwise, update the sampler's distribution and get samples from it
    if(sampler==None):
        query = estimate_dist.sample((query_budget,))
    else:
        sampler.update_distribution(estimate_dist)
        query = sampler.get_samples(query_budget)

    # Split the query into 20 equal-sized chunks
    chunks = torch.chunk(query, 20)

    # Initialize a list to store the labels for each chunk
    label_list = []
    # Set the vic_model to evaluation mode
    vic_model.eval()
    with torch.no_grad():
        # For each chunk, move it to the GPU, compute the logits with the vic_model,
        # apply the tanh function to the mean of the logits to get the labels,
        # and add the labels to the label list
        for chunk in chunks:
            chunk = chunk.cuda()
            logit = vic_model(chunk, 'compute_actor')['logit']
            mu,sigma = logit
            label_chunk = torch.tanh(mu)
            label_list.append(label_chunk.cpu())

    # Concatenate all the labels together
    label = torch.cat(label_list, dim=0)

    # If defense is enabled, apply the defense mechanism
    if(args.defense==True):
        # Define the minimum and maximum observation values for each environment
        if(args.env=="hopper"):
            obs_min = torch.tensor([0,-1,-2,-2,-1,-2,-6,-7,-10,-10,-10])
            obs_max = torch.tensor([2,1,1,1,1,6,3,8,10,10,10])
        elif(args.env=="halfcheetah"):
            obs_min = torch.tensor([-1,-2,-1,-1,-1,-2,-2,-1,-3,-6,-8,-29,-36,-26,-31,-27,-26])
            obs_max = torch.tensor([2,20,2,1,1,1,2,1,16,7,15,25,32,26,29,31,27])
        elif(args.env=="walker2d"):
            obs_min = torch.tensor([0,-1,-3,-3,-2,-3,-3,-2,-5,-7,-10,-10,-10,-10,-10,-10,-10])
            obs_max = torch.tensor([2,1,1,1,2,1,1,2,9,3,10,10,10,10,10,10,10])

        # Create a mask for samples that are out of range
        mask_out_of_range = (query < obs_min) | (query > obs_max)
        mask_out_of_range = mask_out_of_range.any(dim=1)

        # Create a tensor of random values between -1 and 1
        random_values = 2 * torch.rand_like(label) - 1

        # Replace label values where mask_out_of_range is True with random values
        label[mask_out_of_range] = random_values[mask_out_of_range]

    # Split the query and label data into training and validation sets
    train_size = int(train_val_ratio * query_budget)
    current_data = {"train_query": query[:train_size],
                    "train_label": label[:train_size],
                    "val_query": query[train_size:],
                    "val_label": label[train_size:]}

    # Return the current data and the updated global budget
    return current_data,global_budget

def load_victim_policy_eval(victim_cfg, seed=0):
    """
    This function loads the victim model, sets it to evaluation mode, evaluates it, and returns the victim policy.

    Parameters:
    victim_cfg (str): Path to the victim model's configuration
    seed (int): Seed for random number generation. Default is 0.

    Returns:
    victim_policy: The victim policy after evaluation
    cfg: The configuration used for creating the victim policy
    """

    # Import the configuration of the victim model
    victim_cfg_file = f"{victim_cfg}/formatted_total_config.py"
    env_setting = None

    # Read and compile the configuration
    cfg, create_cfg = read_config(victim_cfg_file)
    create_cfg.policy.type = create_cfg.policy.type + '_command'
    cfg = compile_config(cfg, seed=seed, env=None, auto=True, create_cfg=create_cfg, save_cfg=False)
    set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)

    # Create the victim policy and load the pre-trained model
    victim_policy = create_policy(cfg.policy, model=None, enable_field=['eval', 'learn', 'collect', 'command'])
    victim_model_path = f"{victim_cfg}/ckpt/ckpt_best.pth.tar"
    victim_policy._model.load_state_dict(torch.load(victim_model_path, map_location='cpu')['model'], strict=True)

    # Create the evaluation environment
    env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)
    evaluator_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg])
    evaluator = create_serial_evaluator(cfg.policy.eval.evaluator, env=evaluator_env, policy=victim_policy.eval_mode)
    evaluator_env.seed(cfg.seed, dynamic_seed=False)

    # Set the victim model to evaluation mode
    victim_policy._model.eval()

    # Test the victim model and print the mean reward
    print("Reward mean of victim model")
    with torch.no_grad():
        stop, eval_info = evaluator.eval(None, 0, 0)

    return victim_policy, cfg


