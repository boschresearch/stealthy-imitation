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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.distributions as tdist

from ding.config import read_config, compile_config
from ding.policy import create_policy

from ding.worker import BaseLearner
from ding.utils import set_pkg_seed

from utils import create_evaluator_and_logger, load_victim_policy_eval
from utils_all import EmpiricalDistribution, KernelDensityDistribution



class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
def prepare_data_loaders(args, estimate_dist, vic_model):
    """
    Prepare the training and validation data loaders.

    Parameters:
    args: Command-line arguments.
    estimate_dist: The distribution to sample from.
    vic_model: The victim model.

    Returns:
    train_loader, val_loader: The training and validation data loaders.
    """

    bc_query = estimate_dist.sample((args.bc_budget,)).to(torch.float32)
    chunks = torch.chunk(bc_query, 1000) # split data into chunks to query the victim model, otherwise it is too large
    bc_label_list = []  
    vic_model.eval()

    with torch.no_grad():
        for chunk in chunks:
            chunk = chunk.cuda()
            logit = vic_model(chunk, 'compute_actor')['logit']
            mu, sigma = logit 
            bc_label_chunk = torch.tanh(mu)
            bc_label_list.append(bc_label_chunk.cpu()) 

    bc_label = torch.cat(bc_label_list, dim=0) 
    train_size = int(0.8 * bc_query.shape[0])
    train_bc_query = bc_query[:train_size]
    train_bc_label = bc_label[:train_size]
    val_bc_query = bc_query[train_size:]
    val_bc_label = bc_label[train_size:]

    train_dataset = MyDataset(train_bc_query, train_bc_label)
    val_dataset = MyDataset(val_bc_query, val_bc_label)

    train_loader = DataLoader(train_dataset, batch_size=args.bc_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.bc_batch_size, shuffle=False)

    return train_loader, val_loader

def train_bc(global_budget, global_epoch, adv_model, vic_model, estimate_dist, args,criterion, optimizer, num_epochs, evaluator, eval_freq_epoch=5, save_checkpoint=None, tb_logger=None):
    
    adv_model.eval()
    with torch.no_grad():
        stop, eval_info = evaluator.eval(save_checkpoint, global_epoch, global_budget)
    adv_model.train()
    global_budget=global_budget+args.bc_budget
    min_valloss_eval_info = eval_info

    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = args.patience

    train_loader, val_loader = prepare_data_loaders(args, estimate_dist, vic_model)


    for epoch in range(num_epochs):
        global_epoch+=1
        epoch_loss = 0.0
        data_count = 0
        for inputs, targets in train_loader:
            inputs=inputs.cuda()
            targets=targets.cuda()
            optimizer.zero_grad()

            logit = adv_model(inputs, 'compute_actor')['logit']

            mu,sigma = logit
            outputs = torch.tanh(mu)


            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(inputs)
            data_count += len(inputs)

        tb_logger.add_scalar("train_loss", epoch_loss / data_count, global_epoch)


        if (global_epoch) % eval_freq_epoch == 0:
            adv_model.eval()
            with torch.no_grad():
                val_loss = 0.0
                val_data_count = 0
                for val_inputs, val_targets in val_loader:
                    val_inputs = val_inputs.cuda()
                    val_targets = val_targets.cuda()


                    val_logit = adv_model(val_inputs, 'compute_actor')['logit']

                    val_mu,val_sigma = val_logit

                    val_outputs = torch.tanh(val_mu)


                    loss = criterion(val_outputs, val_targets)
                    val_loss += loss.item() * len(val_inputs)
                    val_data_count += len(val_inputs)

                avg_val_loss = val_loss / val_data_count
                tb_logger.add_scalar("val_loss", avg_val_loss, global_epoch)
                with torch.no_grad():
                    stop, eval_info = evaluator.eval(save_checkpoint, global_epoch, global_budget)
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    epochs_no_improve = 0
                    min_valloss_eval_info = eval_info
                # early stopping check
                if args.early_stopping:
                    epochs_no_improve += 1
                    if epochs_no_improve == patience:
                        print('Early stopping!')
                        return global_epoch, global_budget, min_valloss_eval_info
            adv_model.train()
   
    return global_epoch,global_budget,min_valloss_eval_info

def setup_cfg(args):
    """
    Setup the configuration for the experiment.

    Parameters:
    args (argparse.Namespace): The command-line arguments.

    Returns:
    cfg (Config): The configuration for the experiment.
    """
    adv_cfg = f"config/{args.env}/{args.policy}_config.py"
    cfg, create_cfg = read_config(adv_cfg)
    create_cfg.policy.type += '_command'

    # set up exp name
    cfg.exp_name = f"exp/{args.env}/{args.policy}/"
    cfg.exp_name += f"{args.exp_dir}/" if args.exp_dir else ""
    cfg.exp_name += f"{args.exp_name}_" if args.exp_name else ""
    cfg.exp_name += "debug_" if args.debug else ""

    cfg = compile_config(cfg, seed=args.seed, env=None, auto=True, create_cfg=create_cfg, save_cfg=True)
    set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)
    return cfg

def get_sample_dist(cfg, args):
    obs_shape = cfg.policy.model.obs_shape
    real_obs_data = torch.load(f"{args.victim_cfg}/obs_data_1000000.pt") 

    if(args.sample_dist=="normal"):
        means = torch.zeros(obs_shape,dtype=torch.float32)
        stds = torch.ones(obs_shape,dtype=torch.float32)*args.std
        cov_matrix = torch.diag(stds ** 2)
        sample_dist = tdist.MultivariateNormal(means, cov_matrix)
    elif(args.sample_dist=="real_normal"):
        means = real_obs_data['mean']
        stds = real_obs_data['std']
        cov_matrix = torch.diag(stds ** 2)
        sample_dist = tdist.MultivariateNormal(means, cov_matrix)
    elif(args.sample_dist=="real_multinormal"):
        sample_dist = tdist.MultivariateNormal(real_obs_data['mean'], real_obs_data['covariance'])
    elif(args.sample_dist=="real_states"):
        sample_dist = EmpiricalDistribution(real_obs_data['obs_data'])
    elif(args.sample_dist=="real_kde"):
        sample_dist = KernelDensityDistribution(real_obs_data['obs_data'],multivariate=False)
    elif(args.sample_dist=="real_multikde"):
        sample_dist = KernelDensityDistribution(real_obs_data['obs_data'],multivariate=True)
    elif(args.sample_dist=="real_mean_wrong_std"):
        means = real_obs_data['mean']
        stds = real_obs_data['std']*args.scale_std
        cov_matrix = torch.diag(stds ** 2)
        sample_dist = tdist.MultivariateNormal(means, cov_matrix)
    elif(args.sample_dist=="real_std_wrong_mean"):
        means = real_obs_data['mean'] + args.scale_std * (torch.randint(0, 2, (obs_shape,))* 2 - 1) * real_obs_data['std']
        stds = real_obs_data['std']
        cov_matrix = torch.diag(stds ** 2)
        sample_dist = tdist.MultivariateNormal(means, cov_matrix)
    
    return sample_dist

def main(args):

    # Setup the configuration for the attacker policy and reward model
    cfg = setup_cfg(args)

    # Load and evaluate the trained victim policy
    victim_policy,victim_cfg = load_victim_policy_eval(args.victim_cfg,seed=0)
    victim_policy._model.eval()

    # Create the adversary policy
    adversary_policy = create_policy(cfg.policy, model=None, enable_field=['eval', 'learn','collect'])
    
    # Create helper classes like evaluator, optimizer etc.
    tb_logger, evaluator = create_evaluator_and_logger(cfg, adversary_policy)

    # just used for save_ckpt function
    learner = BaseLearner(cfg.policy.learn.learner, adversary_policy.learn_mode, tb_logger, exp_name=cfg.exp_name)
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(adversary_policy._model.actor.parameters(), lr=args.bc_lr)


    # Initialize the global variables
    global_epoch = 0
    global_budget = 0 # consumed budget

    sample_dist = get_sample_dist(cfg, args)

    global_epoch,global_budget,min_valloss_eval_info = train_bc(
                    global_budget=global_budget,
                    global_epoch=global_epoch,
                    adv_model=adversary_policy._model, 
                    vic_model=victim_policy._model, 
                    estimate_dist=sample_dist,
                    args=args, 
                    criterion=criterion, 
                    optimizer=optimizer, 
                    num_epochs=args.bc_num_epochs, 
                    evaluator=evaluator, 
                    eval_freq_epoch=args.eval_freq_epoch, 
                    save_checkpoint=learner.save_checkpoint, 
                    tb_logger=tb_logger)
    tb_logger.add_scalar("final_test/return_mean", min_valloss_eval_info['eval_episode_return_mean'].item(), global_epoch)
    tb_logger.add_scalar("final_test/return_std", torch.tensor(min_valloss_eval_info['eval_episode_return']).std().item(), global_epoch)
    learner.call_hook('after_run')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--debug", type=int, default=1, help="Enable or disable debug mode")
    parser.add_argument("--victim_cfg", type=str, default="exp/hopper/sac_seed0", help="Path to the victim configuration file")
    parser.add_argument("--env", type=str, default="hopper", help="Environment to use for the experiment")
    parser.add_argument("--policy", type=str, default="gail_estimate", help="Policy to use for the experiment")
    parser.add_argument("--seed", type=int, default=4, help="Random seed for reproducibility")
    parser.add_argument("--exp_name", type=str, default="test", help="Name of the experiment")

    parser.add_argument("--bc_budget", type=int, default=100000, help="Budget for behavioral cloning")
    parser.add_argument("--bc_batch_size", type=int, default=512, help="Batch size for behavioral cloning")
    parser.add_argument("--bc_lr", type=float, default=0.001, help="Learning rate for behavioral cloning")
    parser.add_argument("--bc_num_epochs", type=int, default=5, help="Number of epochs for behavioral cloning")
    parser.add_argument("--eval_freq_epoch", type=int, default=2, help="Frequency of evaluation during training")
    parser.add_argument("--early_stopping", type=int, default=0, help="Enable or disable early stopping")
    parser.add_argument("--patience", type=int, default=20, help="Patience for early stopping")

    parser.add_argument("--exp_dir", type=str, help="Directory to save experiment results")

    parser.add_argument("--sample_dist", type=str, default="normal", help="Distribution to sample from")
    parser.add_argument("--std", type=float, default=1, help="Standard deviation for the sampling distribution")
    parser.add_argument("--scale_std", type=float, default=1, help="Scaling factor for the standard deviation")

    args = parser.parse_args()
    print(args)
    main(args)