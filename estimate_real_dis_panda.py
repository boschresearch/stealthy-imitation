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

import copy
import argparse


import torch
from torch import nn, optim
import torch.distributions as tdist
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR

from ding.policy import create_policy
from ding.config import read_config, compile_config
from ding.utils import set_pkg_seed
from ding.reward_model import create_reward_model
from ding.worker import BaseLearner

from tensorboardX import SummaryWriter

import panda_gym
from huggingface_sb3 import EnvironmentName

from utils_panda import (
    add_noise_to_zero_std_dims, 
    panda_evaluator, 
    construct_transfer_dataset_panda, 
    load_victim_policy_eval_panda
)
from utils_all import calculate_budgets, plot_dists, p_test_analysis, create_reward_model_and_collector

val_loss_list= []
variance_of_label = []
mean_distances = []
kl_distances = []



def train_model_from_buf(num_iter,current_data,monitor_budget, model,cfg,args,criterion, optimizer, num_epochs=1, tb_logger=None,global_budget=0):

    model.train()
    
    data = current_data

    monitor_budget = int(monitor_budget * args.monitor_budget_ratio)

    train_budget = int(monitor_budget*0.8)
    val_budget = monitor_budget - train_budget

    train_bc_query = data['train_query'][:train_budget]
    train_bc_label = data['train_label'][:train_budget]
    val_bc_query = data['val_query'][:val_budget]
    val_bc_label = data['val_label'][:val_budget]

    train_dataset = TensorDataset(train_bc_query, train_bc_label)
    val_dataset = TensorDataset(val_bc_query, val_bc_label)

    train_loader = DataLoader(train_dataset, batch_size=args.bc_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.bc_batch_size, shuffle=False)

        
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        data_count = 0
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            inputs = inputs.cuda()
            targets = targets.cuda()



            logit = model(inputs, 'compute_actor')['logit']
            mu,sigma = logit
            outputs = torch.tanh(mu)


            loss = criterion(outputs, targets).mean()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(inputs)
            data_count += len(inputs)
        if(tb_logger!=None):
            tb_logger.add_scalar("monitor/train_loss", epoch_loss / data_count, num_iter)
            tb_logger.add_scalar("monitor/train_loss_budget", epoch_loss / data_count, global_budget)

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            val_data_count = 0

            for val_inputs, val_targets in val_loader:

                val_inputs = val_inputs.cuda()
                val_targets = val_targets.cuda()

                val_logit = model(val_inputs, 'compute_actor')['logit']
                val_mu,val_sigma = val_logit
                val_outputs = torch.tanh(val_mu)

                persample_loss = criterion(val_outputs, val_targets).mean(dim=-1)
                loss = persample_loss.mean()
                val_loss += loss.item() * len(val_inputs)
                val_data_count += len(val_inputs)


            avg_val_loss = val_loss / val_data_count

            if(tb_logger is not None):
                tb_logger.add_scalar("monitor/val_loss", avg_val_loss, num_iter)
                tb_logger.add_scalar("monitor/val_loss_budget", avg_val_loss, global_budget)
                
    return avg_val_loss

def train_bc(global_budget, global_epoch, current_data, bc_budget,train_val_ratio,adv_model, cfg,args,criterion, optimizer, num_epochs,evaluator, eval_freq_epoch=5, save_checkpoint=None, tb_logger=None,epoch_lr_scheduler=None,early_stopping=False):
    
    if early_stopping:
        # early stopping variables
        best_val_loss = float('inf')
        epochs_no_improve = 0
        patience = args.bc_patience


    adv_model.train()
    if(bc_budget!=-1):
        train_amount = int(bc_budget*train_val_ratio)
        val_amount = bc_budget - train_amount
    else:
        train_amount = current_data['train_query'].shape[0]
        val_amount = current_data['val_query'].shape[0]

    train_bc_query = current_data['train_query'][:train_amount]
    train_bc_label = current_data['train_label'][:train_amount]
    val_bc_query = current_data['val_query'][:val_amount]
    val_bc_label = current_data['val_label'][:val_amount]

    train_dataset = TensorDataset(train_bc_query, train_bc_label)
    val_dataset = TensorDataset(val_bc_query, val_bc_label)

    train_loader = DataLoader(train_dataset, batch_size=args.bc_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.bc_batch_size, shuffle=False)


    if(tb_logger is not None):
        val_mask = ((val_bc_label != 1) & (val_bc_label != -1)).all(dim=1) 
        clean_data_ratio = val_mask.sum().item()/val_amount
        tb_logger.add_scalar("val_victim_clean_data_ratio", clean_data_ratio, global_epoch)
        
    for epoch in range(num_epochs):
        global_epoch+=1
        epoch_loss = 0.0
        data_count = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            inputs = inputs.cuda()
            targets = targets.cuda()

            logit = adv_model(inputs, 'compute_actor')['logit']
            mu,sigma = logit
            outputs = torch.tanh(mu)

            loss = criterion(outputs, targets).mean()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(inputs)
            data_count += len(inputs)
        if(tb_logger is not None):
            tb_logger.add_scalar("train_loss", epoch_loss / data_count, global_epoch)
        if(epoch_lr_scheduler is not None):
            epoch_lr_scheduler.step()

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

                    persample_loss = criterion(val_outputs, val_targets).mean(dim=-1)
                    loss = persample_loss.mean()
                    val_loss += loss.item() * len(val_inputs)
                    val_data_count += len(val_inputs)

                avg_val_loss = val_loss / val_data_count
                if(args.estimate_dist=="p_test"):
                    val_loss_list.append(avg_val_loss)
                if(tb_logger is not None):
                    tb_logger.add_scalar("val_loss", avg_val_loss, global_epoch)

                if(evaluator is not None):
                    evaluator.eval(all_timestep=1000,model=adv_model,type='ding',tb_logger=tb_logger,name="adv",global_epoch=global_epoch,global_budget=global_budget)
                
                if early_stopping:
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        epochs_no_improve = 0
                        save_checkpoint("best_val_loss.pth.tar")
                    else:
                        epochs_no_improve += 1
                        if epochs_no_improve == patience:
                            print('Early stopping!')
                            return global_epoch, val_dataset   
            adv_model.train()

    return global_epoch,val_dataset



def setup_cfg(args):
    """
    This function sets up the configuration for the adversary policy and reward model.
    
    Parameters:
    args: ArgumentParser
        Contains all the command line arguments.
    
    Returns:
    cfg: Config
        The configuration for the adversary policy and reward model.
    """
    
    # Read the configuration file
    adv_cfg = f"config/{args.env}/{args.policy}_config.py"
    cfg, create_cfg = read_config(adv_cfg)
    create_cfg.policy.type = create_cfg.policy.type + '_command'

    # Set up experiment name
    cfg.exp_name = f"exp/{args.env}/{args.policy}/"
    cfg.exp_name += f"{args.exp_dir}/" if args.exp_dir else ""
    cfg.exp_name += f"{args.exp_name}_" if args.exp_name else ""
    cfg.exp_name += "debug_" if args.debug else ""
    
    # Setup policy model
    if args.actor_head_layer_num != -1:
        cfg.policy.model.actor_head_layer_num = args.actor_head_layer_num
    if args.actor_head_hidden_size != -1:
        cfg.policy.model.actor_head_hidden_size = args.actor_head_hidden_size

    # Set reward model hyperparameters
    if args.reward_batch_size is not None:
        cfg.reward_model.batch_size = args.reward_batch_size
    if args.reward_target_new_data_count is not None:
        cfg.reward_model.target_new_data_count = args.reward_target_new_data_count
    if args.reward_update_per_collect != 0:
        cfg.reward_model.update_per_collect = args.reward_update_per_collect

    # Set policy collect sample size
    if args.collect_n_sample: 
        cfg.policy.collect.n_sample = args.collect_n_sample

    # Compile the configuration
    cfg = compile_config(cfg, seed=args.seed, env=None, auto=True, create_cfg=create_cfg, save_cfg=True)
    cfg.reward_model.data_path = cfg.exp_name + "/query_label.pt"
    
    # Set the seed for reproducibility
    set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)
    
    return cfg

def initialize_estimate_distribution(cfg, args):
    obs_shape = cfg.policy.model.obs_shape
    real_obs_data = torch.load(f"./exp/panda/{args.algo}/{str(args.env)}_1/obs_data_1000000.pt") 
    real_obs_data = add_noise_to_zero_std_dims(real_obs_data)
    real_mean = real_obs_data['mean'].float()
    real_std = real_obs_data['std'].float()
    real_dist = tdist.Normal(real_mean, real_std)

    # Initialize distribution
    means = torch.zeros(obs_shape, dtype=torch.float32)
    stds = torch.ones(obs_shape, dtype=torch.float32) * args.init_std
    estimate_dist = tdist.Normal(means, stds)

    args.multi_normal = False
    if args.estimate_dist == "full_gaussian":
        covariance = torch.diag(stds**2)
        estimate_dist = tdist.MultivariateNormal(means, covariance)
        real_dist = tdist.MultivariateNormal(real_mean, real_obs_data['covariance'].cpu())
        args.multi_normal = True
    elif args.estimate_dist == "real_dist":
        estimate_dist = real_dist
    elif args.estimate_dist == "real_dist_shift":
        means = real_mean + 3*real_std
        stds = real_std
        estimate_dist = tdist.Normal(means, stds)
    elif args.estimate_dist == "wrong_dist":
        means = torch.ones(obs_shape, dtype=torch.float32)*(-3.0)
        estimate_dist = tdist.Normal(means, stds)
    elif args.estimate_dist == "uniform_dist":
        estimate_range = 3
        estimate_mean = means
        estimate_dist = tdist.Uniform(estimate_mean-estimate_range, estimate_mean+estimate_range)
    elif args.estimate_dist == "p_test":
        mean_distance = torch.rand(1) * args.p_test_range * (torch.randint(0, 2, (obs_shape,))* 2 - 1)
        mean = real_std*mean_distance+real_mean
        estimate_dist = tdist.Normal(mean, real_std)
        args.reward_list = []

    return real_dist, estimate_dist, args, real_mean, real_std

def main(args):

    # Setup the configuration for the attacker policy and reward model
    cfg = setup_cfg(args)
        
    # Load and evaluate the trained victim policy
    vic_model, success_rate,return_mean,return_std,env,log_path = load_victim_policy_eval_panda(args.env,args.algo,args.folder,seed=0)
    # Create the adversary policy
    adversary_policy = create_policy(cfg.policy, model=None, enable_field=['eval', 'learn','collect'])
    init_state_dict = copy.deepcopy(adversary_policy._model.state_dict())
    
    # If monitoring is enabled, create a monitor policy, i.e. distribution evaluator in the paper
    if args.monitor:
        monitor_policy = create_policy(cfg.policy, model=None, enable_field=['eval', 'learn','collect'])
        init_monitor_state_dict = copy.deepcopy(monitor_policy._model.state_dict())
        monitor_max_val_loss = float('-inf')
        monitor_optimizer = optim.Adam(monitor_policy._model.actor.parameters(), lr=args.bc_lr)
        

    # Create evaluator, tb_logger
    evaluator = panda_evaluator(env)
    tb_logger = SummaryWriter(f'./{cfg.exp_name}/log/serial')


    # If a checklist is provided, create a checklist evaluator
    if args.check_list is not None:
        monitor_dic={'monitor_best_estimate_dist':[],
                     'monitor_max_val_loss':[],
                     'num_iter':[],
                     'kl_divergence':[],
                     'global_budget':[],
                     'monitor_data':[],
                     'last_bc_budget':[],
                     'total_budget':args.check_list}
        


    # Create a learner for the save_ckpt function
    learner = BaseLearner(cfg.policy.learn.learner, adversary_policy.learn_mode, tb_logger, exp_name=cfg.exp_name)
    criterion = nn.SmoothL1Loss(reduction='none')
    optimizer = optim.Adam(adversary_policy._model.actor.parameters(), lr=args.bc_lr)

    # Initialize the real and estimated distributions
    real_dist, estimate_dist, args, real_mean, real_std = initialize_estimate_distribution(cfg, args)
    obs_shape = cfg.policy.model.obs_shape

    # Create reward_model and collector for training reward model
    reward_model, init_reward_state_dict, collector = create_reward_model_and_collector(cfg, adversary_policy, tb_logger, obs_shape, estimate_dist, args)
    
    # Initialize the global variables
    global_epoch = 0
    num_iter = 0
    global_budget = 0 # consumed budget
    bc_budget, monitor_budget, query_budget = calculate_budgets(args, estimate_dist)

    # Evaluate the adversary policy
    adversary_policy._model.eval()
    successes, episode_rewards,episode_lengths,_ = evaluator.eval(all_timestep=1000,model=adversary_policy._model,type='ding',tb_logger=tb_logger,name="adv",global_epoch=global_epoch,global_budget=global_budget)

    # Set the ratio of training to validation data
    train_val_ratio = 0.8
    check_list_idx = 0

    # Check if the initial required budget exceeds the first checkpoint total budget
    if(global_budget + query_budget + args.last_bc_budget > monitor_dic['total_budget'][check_list_idx]):
        raise ValueError("The initial required budget exceeds the first checkpoint total budget")
    
    # Check if the total budget (global budget + query budget + last BC budget) is less than the total budget at the current checkpoint
    # If it is, continue the loop
    while(global_budget + query_budget + args.last_bc_budget <= monitor_dic['total_budget'][check_list_idx]):
        num_iter += 1


        # Step I: Transfer dataset construction
        current_data,global_budget = construct_transfer_dataset_panda(global_budget, vic_model,query_budget, estimate_dist,args, train_val_ratio,cfg=cfg)
        kl_divergence = plot_dists(real_dist, estimate_dist, num_iter,tb_logger,global_budget)

        # evaluate current estimated distribution
        if args.monitor:
            monitor_val_loss = train_model_from_buf(num_iter,current_data, monitor_budget,monitor_policy._model,cfg,args,criterion, monitor_optimizer, num_epochs=1, tb_logger=tb_logger,global_budget=global_budget)
            # reinitialize the optimizer and monitor model to keep it only afftected with current estimated distribution
            monitor_optimizer = optim.Adam(monitor_policy._model.actor.parameters(), lr=args.bc_lr)
            monitor_policy._model.load_state_dict(init_monitor_state_dict)
            # update parameters if current loss exceeds max loss
            if(monitor_val_loss>monitor_max_val_loss):
                monitor_max_val_loss = monitor_val_loss
                monitor_best_estimate_dist = copy.deepcopy(estimate_dist)
                monitor_kl_divergence = copy.deepcopy(kl_divergence)
                monitor_data = current_data # save the queried data of best estimate distribution for further training

        # Step II: Behavioral cloning
        global_epoch, val_dataset = train_bc(global_budget, 
                        global_epoch, 
                        current_data, 
                        bc_budget,
                        train_val_ratio,
                        adversary_policy._model, 
                        cfg,
                        args,
                        criterion, 
                        optimizer, 
                        args.bc_num_epochs, 
                        evaluator, 
                        eval_freq_epoch=args.eval_freq_epoch, 
                        save_checkpoint=learner.save_checkpoint, 
                        tb_logger=tb_logger,
                        epoch_lr_scheduler=None)
        
        if args.with_reward:
            # Step III: Reward model training
            # query adversary policy to construct attacker dataset
            new_data_count, target_new_data_count = 0, cfg.reward_model.get('target_new_data_count', 1)
            while new_data_count < target_new_data_count:
                new_data = collector.collect()
                new_data_count += len(new_data)
                reward_model.collect_data(new_data)

            reward_model.load_expert_data_from_buf(current_data,bc_budget,train_val_ratio,prune=args.prune) # load victim transfer dataset
            reward_model.train_with_saved_data() # update reward_model

            # Step IV: Reward-guided distribution refinement
            if args.estimate_new:
                new_bc_estimate_dist=None
 
                new_estimate_dist = reward_model.estimate_dist(top_percentage=args.top_percentage,multi_normal=args.multi_normal,args=args,new_bc_estimate_dist=new_bc_estimate_dist,collector=collector)

                # when p_test is enable,
                # this script is usedto analyse correlation between difficulty of imitation and distribution divergence instead of stealing.
                if args.estimate_dist == "p_test":
                    # Save the z-score and variance of label
                    mean_distances.append(abs(mean_distance[0]).item())  # z-score
                    kl_distances.append(torch.distributions.kl_divergence(real_dist, estimate_dist).numpy().mean())
                    variance_of_label.append(torch.var(current_data["val_label"], dim=0).mean().item())

                    # Initialize a new z-score
                    # The z-score is a measure of how many standard deviations an element is from the mean.
                    # Here, we randomly generate a new z-score within the range specified by args.p_test_range.
                    mean_distance = torch.rand(1) * args.p_test_range * (torch.randint(0, 2, (obs_shape,)) * 2 - 1)

                    # Re-initialize another distribution with the new mean and the same standard deviation as the real distribution
                    mean = real_std * mean_distance + real_mean
                    estimate_dist = tdist.Normal(mean, real_std)

                    # Re-initialize the adversary policy and reward model with their initial states
                    adversary_policy._model.load_state_dict(init_state_dict)
                    reward_model.reward_model.load_state_dict(init_reward_state_dict)

                    # Reset the distribution of the collector for training reward
                    collector.set_dist(estimate_dist)

                    # Re-initialize the optimizer for the adversary policy
                    optimizer = optim.Adam(adversary_policy._model.actor.parameters(), lr=args.bc_lr)
                    continue

                estimate_dist = new_estimate_dist
                collector.set_dist(estimate_dist) # update collector with new_estimate_dist
            else:
                # do not update estimate_dist but executed to draw reward distribution
                reward_model.estimate_dist(top_percentage=args.top_percentage,multi_normal=args.multi_normal,estimate_dist=args.estimate_dist)

            # clear the data of reward model
            reward_model.clear_data()

        # Calculate the budgets for the next round
        bc_budget, monitor_budget, query_budget = calculate_budgets(args, estimate_dist)

        # Check if the total of the next query, the global budget, and the last BC budget exceeds the total budget
        total_budget_exceeded = (global_budget + query_budget + args.last_bc_budget) >= monitor_dic['total_budget'][check_list_idx]

        if total_budget_exceeded:
            # If the total budget is exceeded, save the current data as a checkpoint
            monitor_dic['monitor_best_estimate_dist'].append(monitor_best_estimate_dist)
            monitor_dic['monitor_max_val_loss'].append(monitor_max_val_loss)
            monitor_dic['num_iter'].append(num_iter)
            monitor_dic['kl_divergence'].append(monitor_kl_divergence)
            monitor_dic['global_budget'].append(global_budget)
            monitor_dic['monitor_data'].append(monitor_data)
            
            # Calculate the remaining budget for the last BC
            remaining_bc_budget = args.check_list[check_list_idx] - global_budget
            monitor_dic['last_bc_budget'].append(remaining_bc_budget)
            
            # Move to the next checkpoint
            check_list_idx += 1 

            # If all checkpoints have been processed, break the loop
            if check_list_idx >= len(monitor_dic['total_budget']):
                break

    if(args.estimate_dist=="p_test"):
        p_test_analysis(kl_distances, 
                        val_loss_list=val_loss_list, 
                        variance_of_label=variance_of_label, 
                        reward_list=args.reward_list,
                        save_dir=cfg.exp_name)
        
    # train the attacker policy only with the best distribution evaluated by monitor_policy(distribution evaluator) for each checkpoint
    if args.check_list is not None:
        # Iterate over all the budgets in the monitor dictionary
        for check_idx in range(len(monitor_dic['total_budget'])):

            # Extract the necessary data from the monitor dictionary
            query_budget = monitor_dic['last_bc_budget'][check_idx]
            estimate_dist = monitor_dic['monitor_best_estimate_dist'][check_idx]
            save_data = monitor_dic['monitor_data'][check_idx]
            total_budget = monitor_dic['total_budget'][check_idx]
            num_iter = monitor_dic['num_iter'][check_idx]
            
            # Construct a new transfer dataset and concatenate it with the saved data
            bc_data, _ = bc_data,_ = construct_transfer_dataset_panda(0, vic_model,query_budget, estimate_dist,args, train_val_ratio,cfg=cfg)
            for key in ['train_query', 'train_label', 'val_query', 'val_label']:
                bc_data[key] = torch.cat((bc_data[key], save_data[key]), dim=0)
            
            # Initialize the adversary policy model
            adversary_policy._model.load_state_dict(init_state_dict) 

            # Initialize the optimizer
            optimizer = optim.Adam(adversary_policy._model.actor.parameters(), lr=args.bc_lr)
            
            # Set up the learning rate scheduler based on the last_lr_decay argument
            epoch_lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.retrain_bc_epochs) if args.last_lr_decay else None
            
            # Set the evaluation frequency based on the early_stopping argument
            # if do early_stopping, then evaluate the model at each epoch
            eval_freq_epoch = 1 if args.early_stopping else args.retrain_bc_epochs + 1
            
            # Train the attacker policy with the best distribution evaluated by the monitor policy
            train_bc(0, 
                    0, 
                    bc_data, 
                    -1,
                    train_val_ratio,
                    adversary_policy._model, 
                    cfg,
                    args,
                    criterion, 
                    optimizer, 
                    args.retrain_bc_epochs, 
                    None, 
                    eval_freq_epoch=eval_freq_epoch, 
                    save_checkpoint=learner.save_checkpoint, 
                    tb_logger=None,
                    epoch_lr_scheduler=epoch_lr_scheduler,
                    early_stopping=args.early_stopping)
            
            # Evaluate the model at the current checkpoint
            if args.early_stopping: # if it is early_stopping, then the best model should be reloaded
                best_model_path = f"{cfg.exp_name}/ckpt/best_val_loss.pth.tar"
                adversary_policy._model.load_state_dict(torch.load(best_model_path, map_location='cpu')['model'], strict=True)
            
            adversary_policy._model.eval()
            with torch.no_grad():
                evaluator.eval(all_timestep=1000,model=adversary_policy._model,type='ding',tb_logger=tb_logger,name="adv_checklist",global_epoch=num_iter,global_budget=total_budget)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--debug", type=int, default=1)
    parser.add_argument("--folder", type=str,default="exp/panda")
    parser.add_argument("--algo", type=str,default="tqc")
    parser.add_argument("--env", help="environment ID", type=EnvironmentName, default="PandaPickAndPlace-v1")
    parser.add_argument("--seed", type=int,default=4)
    parser.add_argument("--policy", type=str,default='gail_estimate')
    parser.add_argument("--exp_name", type=str,default="test")

    parser.add_argument("--bc_budget", type=int,default=100000)# 100k 
    parser.add_argument("--bc_batch_size", type=int, default=512) # 512
    parser.add_argument("--bc_lr", type=float, default=0.001)
    parser.add_argument("--bc_num_epochs", type=int, default=1)

    parser.add_argument("--last_bc_budget", type=int, default=1000000, help="Last budget for behavioral cloning. Default is 1M.")
    parser.add_argument("--check_list", nargs='+', type=int, default=None, help="Checklist of items. Default is None.")

    parser.add_argument("--exp_dir", type=str, help="Directory for the experiment.")
    parser.add_argument("--only_mean", type=int,default=1, help="Only use mean. Default is 1.")
    parser.add_argument("--fix_states", type=int,default=0, help="Fix states. Default is 0.")
    parser.add_argument("--all_states", type=int,default=1000000, help="All states. Default is 1000000.")
    parser.add_argument("--reward_batch_size", type=int,default=640, help="Batch size for reward. Default is 640.")
    parser.add_argument("--collect_n_sample", type=int,default=640, help="Number of samples to collect. Default is 640.")
    parser.add_argument("--reward_target_new_data_count", type=int,default=640, help="Target new data count for reward. Default is 640.")
    parser.add_argument("--with_reward", type=int,default=1, help="With reward. Default is 1.")
    
    parser.add_argument("--retrain_bc_epochs", type=int,default=200, help="Number of epochs for retraining behavioral cloning. Default is 200.")
    parser.add_argument("--multi_normal", type=int,default=0, help="Use multi normal. Default is 0.")

    parser.add_argument("--top_percentage", type=float,default=1, help="Top percentage of data to train reward model. Default is 1, i.e. 100%.")

    parser.add_argument("--estimate_dist", type=str,default=None, help="Estimate distribution. Default is None.")
    parser.add_argument("--init_std", type=float,default=1.0, help="Initial standard deviation. Default is 1.0.")
    parser.add_argument("--estimate_new", type=int,default=1, help="Estimate new. Default is 1.")

    parser.add_argument("--reward_update_per_collect", type=int,default=400, help="Reward update per collect. Default is 400.")

    parser.add_argument("--eval_freq_epoch", type=int,default=5, help="Frequency of evaluation per epoch. Default is 5.")
    parser.add_argument("--prune", type=int,default=1, help="Prune. Default is 1.")

    parser.add_argument("--actor_head_layer_num", type=int,default=-1, help="Number of layers in actor head. Default is -1.")
    parser.add_argument("--actor_head_hidden_size", type=int,default=-1, help="Hidden size of actor head. Default is -1.")

    parser.add_argument("--defense", type=int,default=0, help="Defense. Default is 0.")
    parser.add_argument("--monitor", type=int,default=0, help="Monitor. Default is 0.")

    parser.add_argument("--adapting_bc_budget", type=float,default=0, help="Adapting budget for behavioral cloning. Default is 0.")
    parser.add_argument("--monitor_budget_ratio", type=float,default=1.0, help="Budget ratio for monitor. Default is 1.0.")
    parser.add_argument("--bc_patience", type=int,default=20, help="Patience for behavioral cloning. Default is 20.")
    parser.add_argument("--last_lr_decay", type=int,default=1, help="Last learning rate decay. Default is 1.")
    parser.add_argument("--early_stopping", type=int,default=0, help="Early stopping. Default is 0.")
    parser.add_argument("--both_dynamic", type=int,default=0, help="Both dynamic. Default is 0.")
    parser.add_argument("--p_test_range", type=int,default=10, help="Test range for p. Default is 10.")

    parser.add_argument("--reward_on", type=str,choices=['vic','adv','both'], default='vic', help="Reward on. Default is 'vic'.")

    args = parser.parse_args()
    print(args)

    main(args)
