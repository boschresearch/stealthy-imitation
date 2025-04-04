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
import copy
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
from ding.reward_model import create_reward_model

class RandomCollector():
    def __init__(self, 
                 vic_rm=0, 
                 adv_model=None,
                 reward_model=None,
                 obs_shape=10,
                 n_sample=640,
                 bound_type=None,
                 kl_coef=0,

                 only_mean=True,
                 all_states=50000,
                 fix_states=1,
                 estimate_dist=None,
                 cfg=None):
        # Initialize models and configurations
        self.adv_model = adv_model
        self.reward_model = reward_model
        self.cfg = cfg

        # Initialize parameters for sampling and distribution estimation
        self.estimate_dist = estimate_dist
        self.obs_shape = obs_shape
        self.n_sample = n_sample
        self.collect_iter = 0
        self.envstep = 0
        self.vic_rm = vic_rm
        self.bound_type = bound_type
        self.kl_coef = kl_coef

        self.only_mean = only_mean

        self.fix_states=fix_states

        # If fix_states is True, initialize a fixed set of states for sampling
        if(fix_states==True):
            self.all_states = self.estimate_dist.sample((all_states,))


    def collect(self,n_sample=-1,state=None):
        # Set the number of samples to collect
        if(n_sample == -1):
            n_sample = self.n_sample
        
        # Set models to evaluation mode
        if self.adv_model!=None:
            self.adv_model.eval()


        # Initialize data dictionary
        data={}
        # Determine the state for sampling
        if(state==None):
            if(self.fix_states==True):
                indices = torch.randperm(self.all_states.shape[0])[:n_sample]
                state = self.all_states[indices].cuda()
            else:
                state = self.estimate_dist.sample((n_sample,))
                state = state.cuda()
        else:
            state  = state.cuda()
            n_sample = state.shape[0]

        # Update iteration and environment step counters
        self.collect_iter += 1
        self.envstep += n_sample

        # Collect data
        with torch.no_grad():
            # Compute action logits with the adversarial model
            adv_logit = self.adv_model(state,'compute_actor')['logit']
            adv_mu,adv_sigma = adv_logit 

            # Create a distribution for the adversarial model
            dist_adv = Independent(Normal(adv_mu, adv_sigma), 1)
            # Sample actions from the distribution
            if(self.only_mean!=True): 
                adv_actions = dist_adv.sample()
            else:
                adv_actions = adv_mu 

            adv_actions = torch.tanh(adv_actions)


            # Store the collected data
            data["obs"] = state.cpu()
            data["next_obs"] = torch.zeros_like(data["obs"])
            data["action"] = adv_actions.cpu()
            data["reward"] = torch.zeros([n_sample,1])
            data["done"] = torch.ones(n_sample, dtype=torch.bool)
            data["collect_iter"] = torch.full([n_sample], self.collect_iter)

        # Convert the data dictionary to a list of dictionaries
        list_data=[]
        for i in range(n_sample):
            list_data.append({
                "obs": data["obs"][i],
                "next_obs": data["next_obs"][i],
                "action": data["action"][i],
                "reward": data["reward"][i],
                "done": data["done"][i],
                "collect_iter": data["collect_iter"][i],
            })

        return list_data
    
    def set_dist(self, estimate_dist):
        """ Update the distribution for sampling """
        self.estimate_dist = estimate_dist




class EmpiricalDistribution(torch.distributions.Distribution):
    def __init__(self, tensor):
        super().__init__()

        self.tensor = tensor
        self.num_samples = tensor.size(0)
        self.dim = tensor.size(1)

    def sample(self, sample_shape=torch.Size()):
        indices = torch.randint(low=0, high=self.num_samples, size=sample_shape + torch.Size([1]))
        return torch.index_select(self.tensor, 0, indices.squeeze(-1))

    def log_prob(self, value):
        raise NotImplementedError("Empirical distribution does not support computing a log probability.")

    def entropy(self):
        raise NotImplementedError("Empirical distribution does not support computing entropy.")

    def __getitem__(self, index):
        return self.tensor[index]

    def __len__(self):
        return self.tensor.size(0)
    

class KernelDensityDistribution(Distribution):
    def __init__(self, data, bandwidth=None, multivariate=False):
        super().__init__()
        self.multivariate = multivariate
        if self.multivariate:
            self.kde = gaussian_kde(data.numpy().T, bw_method=bandwidth)
        else:
            self.kdes = [gaussian_kde(data[:,i].numpy(), bw_method=bandwidth) for i in range(data.shape[1])]

        self.samples = self.sample((1000,))
        self.log_prob_p = self.log_prob(self.samples)
        self.dim=data.shape[1]
    def sample(self, sample_shape=torch.Size()):
        if self.multivariate:
            samples = torch.from_numpy(self.kde.resample(sample_shape))
            return samples.t()
        else:
            samples = [torch.from_numpy(kde.resample(sample_shape)) for kde in self.kdes]
            return torch.stack(samples, dim=-1).squeeze(0)

    
    def log_prob(self, value):
        if self.multivariate:
            log_prob_values = torch.from_numpy(self.kde.logpdf(value.numpy().T))
        else:
            log_prob_values = [torch.from_numpy(kde.logpdf(value[:, i].numpy())) for i, kde in enumerate(self.kdes)]
            log_prob_values = torch.stack(log_prob_values, dim=-1).sum(-1)
            
        return log_prob_values
        
    def plot_distributions(self, filename):

        assert self.multivariate==False

        num_vars = len(self.kdes)
        num_rows = int(np.ceil(num_vars / 3))  # adjust number of columns as needed
        fig, axes = plt.subplots(num_rows, 3, figsize=(15, num_rows*5))

        x = np.linspace(-5, 5, 100)  # adjust range and precision as needed
        for i, kde in enumerate(self.kdes):
            row = i // 3
            col = i % 3
            ax = axes[row, col]
            y = kde.pdf(x)
            ax.plot(x, y)
            ax.set_title(f'Variable {i+1}')
            
        # Remove empty subplots
        for j in range(i+1, num_rows*3):
            fig.delaxes(axes.flatten()[j])

        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
    def kl_kde_normal(self, q: Normal):
        """
        Calculate the Kullback-Leibler divergence between a KernelDensityDistribution and a Normal distribution.
        Monte Carlo sampling is used to approximate the KL divergence for multivariate distributions, from http://joschu.net/blog/kl-approx.html
        
        Parameters:
            p: KernelDensityDistribution object.
            q: Normal distribution object.
        """
        assert self.multivariate, "KL divergence calculation is only implemented for multivariate KernelDensityDistribution."

        log_prob_q = q.log_prob(self.samples).sum(-1)
        logr = log_prob_q - self.log_prob_p
        kl_div = (logr.exp() - 1) - logr

        return kl_div/self.dim

def plt_to_tensor():
    """Converts plot figure (plt) to tensor for tensorboard"""

    # Save plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    # Load image from memory and convert to tensor
    plt_img = Image.open(buf)
    tensor_image = ToTensor()(plt_img)

    # Close the buffer
    buf.close()

    return tensor_image


def plot_dists(real_dist, estimate_dist, iteration,tb_logger,global_budget):
    mean_MAEs=0
    std_MAEs=0

    kl_divergence = None
    # Calculate and log kL divergence
    if not isinstance(estimate_dist, torch.distributions.uniform.Uniform):
        # Skip uniform distributions for KL divergence calculation
        kl_divergence = torch.distributions.kl_divergence(real_dist, estimate_dist)
        kl_mean = kl_divergence.mean().item()
        tb_logger.add_scalar('KL_divergence', kl_mean, iteration)
        tb_logger.add_scalar('KL_divergence/budget', kl_mean, global_budget)

        # Check for multivariate normal distribution to skip detailed logging
        if not isinstance(estimate_dist, torch.distributions.multivariate_normal.MultivariateNormal):
            for i, kl_value in enumerate(kl_divergence, start=1):
                tb_logger.add_scalar(f'KL_divergence/variable_{i}', kl_value.item(), iteration)

    if(not isinstance(estimate_dist, torch.distributions.multivariate_normal.MultivariateNormal)):
        for i in range(real_dist.mean.shape[0]):
            plt.figure(dpi=75)

            # Create a univariate distribution for the current dimension
            real_uni_dist = torch.distributions.Normal(real_dist.mean[i], real_dist.stddev[i])
            if(isinstance(estimate_dist,torch.distributions.normal.Normal)):
                estimate_uni_dist = torch.distributions.Normal(estimate_dist.mean[i], estimate_dist.stddev[i])
                estimate_left = estimate_dist.mean[i] - 4 * estimate_dist.stddev[i]
                estimate_right = estimate_dist.mean[i] + 4 * estimate_dist.stddev[i]
                estimate_mean = estimate_dist.mean[i]
                estimate_std = estimate_dist.stddev[i]

            elif(isinstance(estimate_dist,torch.distributions.uniform.Uniform)):
                estimate_uni_dist = torch.distributions.Uniform(estimate_dist.low[i], estimate_dist.high[i])
                estimate_left = estimate_dist.low[i]
                estimate_right = estimate_dist.high[i]
                estimate_mean = (estimate_left+estimate_right)*0.5
                estimate_std = (estimate_right-estimate_mean)/3.0

            # get MAE and draw mean and std
            mean_MAE = abs(estimate_mean - real_dist.mean[i])
            mean_MAEs+=mean_MAE
            std_MAE = abs(estimate_std - real_dist.stddev[i])
            std_MAEs+=std_MAE


            tb_logger.add_scalar(f'dist_mae/mean_{i+1}', mean_MAE, iteration)
            tb_logger.add_scalar(f'dist_mae/std_{i+1}', std_MAE, iteration)
            tb_logger.add_scalar(f'dist/mean_{i+1}', estimate_mean, iteration)
            tb_logger.add_scalar(f'dist/std_{i+1}', estimate_std, iteration)
            # std_tb_logger.add_scalar(f'dist/stats_{i+1}', estimate_std, iteration)

            # Define range for x according to the standard deviation of the distributions
            real_std = real_uni_dist.stddev
            real_mean = real_uni_dist.mean

            # estimate_std = estimate_uni_dist.stddev
            # estimate_mean = estimate_uni_dist.mean

            # Calculate the left and right boundaries
            left = min(real_mean - 4 * real_std, estimate_left)
            right = max(real_mean + 4 * real_std, estimate_right)

            # Define x according to the range
            # x = torch.linspace(real_mean - 4 * real_std, real_mean + 4 * real_std, 100)
            x = torch.linspace(left, right, 100)

            # Calculate PDF
            real_y = torch.exp(real_uni_dist.log_prob(x))
            plt.plot(x.numpy(), real_y.numpy(), label='Real')

            if(isinstance(estimate_dist,torch.distributions.normal.Normal)):
                estimate_y = torch.exp(estimate_uni_dist.log_prob(x))
                plt.plot(x.numpy(), estimate_y.numpy(), label='Estimate')
            elif(isinstance(estimate_dist,torch.distributions.uniform.Uniform)):
                pdf_value = 1.0 / (estimate_dist.high[i] - estimate_dist.low[i])
                plt.hlines(pdf_value, xmin=estimate_dist.low[i], xmax=estimate_dist.high[i], colors="red", label='Estimate')
                plt.vlines(x=estimate_dist.low[i],ymin=0,ymax=pdf_value,colors="red")
                plt.vlines(x=estimate_dist.high[i],ymin=0,ymax=pdf_value,colors="red")

            plt.legend()
            plt.title(f"Distribution {i+1} at Iteration {iteration+1}")

            # Convert plot to tensor
            plt_img = plt_to_tensor()

            # Add image to tensorboard writer
            tb_logger.add_image(f'dist_comp/Distribution_{i+1}', plt_img, iteration)
            
            plt.close()

        tb_logger.add_scalar('dist_mae/mean_mean', mean_MAEs/real_dist.mean.shape[0], iteration)
        tb_logger.add_scalar('dist_mae/std_mean', std_MAEs/real_dist.mean.shape[0], iteration)
        tb_logger.add_scalar('dist/mean_mean', estimate_dist.mean.mean(), iteration)
        tb_logger.add_scalar('dist/std_mean', estimate_dist.stddev.mean(), iteration)
    
    return kl_divergence


def save_to_csv(x, y, filename):
    df = pd.DataFrame({'X': x, 'Y': y})
    df.to_csv(filename, index=False)

def perform_ols_analysis(x_values, y_values):
    x_with_const = sm.add_constant(x_values)
    model = sm.OLS(y_values, x_with_const)
    results = model.fit()
    return results

def plot_and_save(x, y, title, x_label, y_label, filename):
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='blue')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    plt.savefig(filename, dpi=300)

def p_test_analysis(mean_distances, val_loss_list=None, variance_of_label=None, reward_list=None, save_dir="."):
    """
    This function performs an analysis of the p-test. It checks if the save directory exists, if not it creates it.
    It then performs an OLS analysis on the mean distances and validation loss list, variance of label, and reward list if they are not None.
    The results are then saved to a log file, plotted and saved as a png, and saved as a csv file.

    Parameters:
    mean_distances (list): List of mean distances
    val_loss_list (list, optional): List of validation losses. Default is None.
    variance_of_label (list, optional): List of variances of labels. Default is None.
    reward_list (list, optional): List of rewards. Default is None.
    save_dir (str, optional): Directory to save the results. Default is current directory.
    """

    # Check if save_dir exists, if not create it
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # If val_loss_list is provided, perform OLS analysis, save the results, plot and save the plot, and save the data to a csv file
    if val_loss_list is not None:
        results_valloss = perform_ols_analysis(mean_distances, val_loss_list)
        
        with open(f"{save_dir}/p_test.log", "a") as file:
            print("Validation loss vs Distance of mean", file=file)
            print(results_valloss.summary(), file=file)
        
        plot_and_save(mean_distances, val_loss_list, 'Validation loss vs Distance of mean', 'KL divergence', 'Validation Loss', f"{save_dir}/p_test_valloss.png")
        save_to_csv(mean_distances, val_loss_list, f"{save_dir}/p_test_valloss.csv")
    
    # If variance_of_label is provided, perform OLS analysis, save the results, plot and save the plot, and save the data to a csv file
    if variance_of_label is not None:
        results_variance = perform_ols_analysis(mean_distances, variance_of_label)
        
        with open(f"{save_dir}/p_test.log", "a") as file:
            print("Variance of label vs Distance of mean", file=file)
            print(results_variance.summary(), file=file)

        plot_and_save(mean_distances, variance_of_label, 'Variance of label vs Distance of mean', 'KL divergence', 'Variance of label in validation dataset', f"{save_dir}/p_test_variance.png")
        save_to_csv(mean_distances, variance_of_label, f"{save_dir}/p_test_variance.csv")
    
    # If reward_list is provided, perform OLS analysis, save the results, plot and save the plot, and save the data to a csv file
    if reward_list is not None:
        results_reward = perform_ols_analysis(mean_distances, reward_list)
        
        with open(f"{save_dir}/p_test.log", "a") as file:
            print("Reward of label vs Distance of mean", file=file)
            print(results_reward.summary(), file=file)

        plot_and_save(mean_distances, reward_list, 'Proxy reward vs Distance of mean', 'KL divergence', 'Proxy reward in validation dataset', f"{save_dir}/p_test_reward.png")
        save_to_csv(mean_distances, reward_list, f"{save_dir}/p_test_reward.csv")



def create_reward_model_and_collector(cfg, adversary_policy, tb_logger, obs_shape, estimate_dist, args):
    reward_model = create_reward_model(cfg.reward_model, adversary_policy.collect_mode.get_attribute('device'), tb_logger)
    init_reward_state_dict = copy.deepcopy(reward_model.reward_model.state_dict())

    collector = RandomCollector(
        vic_rm=0,
        adv_model=adversary_policy._model,
        reward_model=None, 
        obs_shape=obs_shape, 
        n_sample=cfg.policy.collect.n_sample,
        bound_type=None,
        kl_coef=0,
        only_mean=args.only_mean,
        all_states=args.all_states,
        fix_states=args.fix_states,
        estimate_dist=estimate_dist,
        cfg=cfg
    )

    return reward_model, init_reward_state_dict, collector

def calculate_budgets(args, estimate_dist):
    if args.adapting_bc_budget:
        bc_budget = int(args.bc_budget * estimate_dist.stddev.mean().item())
        monitor_budget = args.bc_budget
        query_budget = max(monitor_budget, bc_budget)
    else:
        bc_budget = args.bc_budget
        monitor_budget = args.bc_budget
        query_budget = args.bc_budget

    if args.both_dynamic:
        bc_budget = int(args.bc_budget * estimate_dist.stddev.mean().item())
        monitor_budget = bc_budget
        query_budget = bc_budget

    return bc_budget, monitor_budget, query_budget
