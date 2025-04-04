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

from typing import List, Dict, Any
from collections.abc import Iterable
import random
import copy
from functools import partial

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from easydict import EasyDict

from ding.utils import REWARD_MODEL_REGISTRY
from ding.reward_model.base_reward_model import BaseRewardModel




def concat_state_action_pairs(iterator):
    """
    Overview:
        Concatenate state and action pairs from input.
    Arguments:
        - iterator (:obj:`Iterable`): Iterables with at least ``obs`` and ``action`` tensor keys.
    Returns:
        - res (:obj:`Torch.tensor`): State and action pairs.
    """
    assert isinstance(iterator, Iterable)
    res = []
    for item in iterator:
        state = item['obs'].flatten()  # to allow 3d obs and actions concatenation
        action = item['action']
        s_a = torch.cat([state, action.float()], dim=-1)
        res.append(s_a)
    return res


def concat_state_action_pairs_one_hot(iterator, action_size: int):
    """
    Overview:
        Concatenate state and action pairs from input. Action values are one-hot encoded
    Arguments:
        - iterator (:obj:`Iterable`): Iterables with at least ``obs`` and ``action`` tensor keys.
    Returns:
        - res (:obj:`Torch.tensor`): State and action pairs.
    """
    assert isinstance(iterator, Iterable)
    res = []
    for item in iterator:
        state = item['obs'].flatten()  # to allow 3d obs and actions concatenation
        action = item['action']
        action = torch.Tensor([int(i == action) for i in range(action_size)])
        s_a = torch.cat([state, action], dim=-1)
        res.append(s_a)
    return res


class RewardModelNetwork(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super(RewardModelNetwork, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, output_size)
        self.a1 = nn.Tanh()
        self.a2 = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        out = self.l1(out)
        out = self.a1(out)
        out = self.l2(out)
        out = self.a2(out)
        return out


class AtariRewardModelNetwork(nn.Module):

    def __init__(self, input_size: int, action_size: int) -> None:
        super(AtariRewardModelNetwork, self).__init__()
        self.input_size = input_size
        self.action_size = action_size
        self.conv1 = nn.Conv2d(4, 16, 7, stride=3)
        self.conv2 = nn.Conv2d(16, 16, 5, stride=2)
        self.conv3 = nn.Conv2d(16, 16, 3, stride=1)
        self.conv4 = nn.Conv2d(16, 16, 3, stride=1)
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64 + self.action_size, 1)  # here we add 1 to take consideration of the action concat
        self.a = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input: x = [B, 4 x 84 x 84 + self.action_size], last element is action
        actions = x[:, -self.action_size:]  # [B, self.action_size]
        # get observations
        x = x[:, :-self.action_size]
        x = x.reshape([-1] + self.input_size)  # [B, 4, 84, 84]
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = x.reshape(-1, 784)
        x = F.leaky_relu(self.fc1(x))
        x = torch.cat([x, actions], dim=-1)
        x = self.fc2(x)
        r = self.a(x)
        return r


@REWARD_MODEL_REGISTRY.register('custom_gail')
class CustomGailRewardModel(BaseRewardModel):
    """
    Overview:
        The Gail reward model class (https://arxiv.org/abs/1606.03476)
    Interface:
        ``estimate``, ``train``, ``load_expert_data``, ``collect_data``, ``clear_date``, \
            ``__init__``,  ``state_dict``, ``load_state_dict``, ``learn``
    Config:
           == ====================  ========   =============  ================================= =======================
           ID Symbol                Type       Default Value  Description                       Other(Shape)
           == ====================  ========   =============  ================================= =======================
           1  ``type``              str        gail           | RL policy register name, refer  | this arg is optional,
                                                              | to registry ``POLICY_REGISTRY`` | a placeholder
           2  | ``expert_data_``    str        expert_data.   | Path to the expert dataset      | Should be a '.pkl'
              | ``path``                       .pkl           |                                 | file
           3  | ``update_per_``     int        100            | Number of updates per collect   |
              | ``collect``                                   |                                 |
           4  | ``batch_size``      int        64             | Training batch size             |
           5  | ``input_size``      int                       | Size of the input:              |
              |                                               | obs_dim + act_dim               |
           6  | ``target_new_``     int        64             | Collect steps per iteration     |
              | ``data_count``                                |                                 |
           7  | ``hidden_size``     int        128            | Linear model hidden size        |
           8  | ``collect_count``   int        100000         | Expert dataset size             | One entry is a (s,a)
              |                                               |                                 | tuple
           == ====================  ========   =============  ================================= =======================

       """
    config = dict(
        type='custom_gail',
        learning_rate=1e-3,
        update_per_collect=100,
        batch_size=64,
        input_size=4,
        target_new_data_count=64,
        hidden_size=128,
        collect_count=100000,
    )

    def __init__(self, config: EasyDict, device: str, tb_logger: 'SummaryWriter') -> None:  # noqa
        """
        Overview:
            Initialize ``self.`` See ``help(type(self))`` for accurate signature.
        Arguments:
            - cfg (:obj:`EasyDict`): Training config
            - device (:obj:`str`): Device usage, i.e. "cpu" or "cuda"
            - tb_logger (:obj:`SummaryWriter`): Logger, defaultly set as 'SummaryWriter' for model summary
        """
        super().__init__()
        self.cfg = config
        assert device in ["cpu", "cuda"] or "cuda" in device
        self.device = device
        self.tb_logger = tb_logger
        obs_shape = config.input_size
        if isinstance(obs_shape, int) or len(obs_shape) == 1:
            self.reward_model = RewardModelNetwork(config.input_size, config.hidden_size, 1)
            self.concat_state_action_pairs = concat_state_action_pairs
        elif len(obs_shape) == 3:
            action_shape = self.cfg.action_size
            self.reward_model = AtariRewardModelNetwork(config.input_size, action_shape)
            self.concat_state_action_pairs = partial(concat_state_action_pairs_one_hot, action_size=action_shape)
        self.reward_model.to(self.device)
        self.expert_data = []
        self.train_data = []
        self.expert_data_loader = None
        self.opt = optim.Adam(self.reward_model.parameters(), config.learning_rate)
        self.train_iter = 0

 
    def load_expert_data_from_buf(self, current_data, bc_budget, train_val_ratio, prune=True) -> None:
        """
        Overview:
            Loads expert data from the buffer based on the provided budget and ratio for training and validation.
            Optionally prunes the data based on certain conditions.
        Effects:
            This function updates the 'self.train_expert_data' and 'self.val_expert_data' attributes with the loaded and optionally pruned data.
        """
        # Calculate the amount of data to be used for training and validation
        train_amount = int(bc_budget * train_val_ratio)
        val_amount = bc_budget - train_amount

        # Extract the training and validation data
        train_bc_query = current_data['train_query'][:train_amount]
        train_bc_label = current_data['train_label'][:train_amount]
        val_bc_query = current_data['val_query'][:val_amount]
        val_bc_label = current_data['val_label'][:val_amount]

        if prune:
            # Create a mask to filter out data where the label is not 1 or -1
            train_mask = ((train_bc_label != 1) & (train_bc_label != -1)).all(dim=1) 
            # Apply the mask to the training data
            self.train_expert_data = torch.cat([train_bc_query[train_mask], train_bc_label[train_mask]], dim=-1)

            # Create a similar mask for the validation data
            val_mask = ((val_bc_label != 1) & (val_bc_label != -1)).all(dim=1) 
            # Apply the mask to the validation data
            self.val_expert_data = torch.cat([val_bc_query[val_mask], val_bc_label[val_mask]], dim=-1)
        else:
            # If not pruning, simply concatenate the query and label data
            self.train_expert_data = torch.cat([train_bc_query, train_bc_label], dim=-1)
            self.val_expert_data = torch.cat([val_bc_query, val_bc_label], dim=-1)

        # Store the shape of the validation query data for future use
        self.s_shape = val_bc_query.shape[1]

    def state_dict(self) -> Dict[str, Any]:
        return {
            'model': self.reward_model.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.reward_model.load_state_dict(state_dict['model'])

    def learn(self, train_data: torch.Tensor, expert_data: torch.Tensor) -> float:
        """
        Overview:
            Helper function for ``train`` which calculates loss for train data and expert data.
        Arguments:
            - train_data (:obj:`torch.Tensor`): Data used for training
            - expert_data (:obj:`torch.Tensor`): Expert data
        Returns:
            - Combined loss calculated of reward model from using ``train_data`` and ``expert_data``.
        """
        # Calculate loss. The following are some hyper-parameters.
        out_1: torch.Tensor = self.reward_model(train_data) # The output for train data, ideally this should be 1
        loss_1: torch.Tensor = torch.log(out_1 + 1e-8).mean() # The output is a number between 0 and 1. The closer the output is to 1, the smaller the negative loss (the negative sign is added later)
        out_2: torch.Tensor = self.reward_model(expert_data) # The output for expert data, ideally this should be 0
        loss_2: torch.Tensor = torch.log(1 - out_2 + 1e-8).mean() # The closer the output for expert data is to 0, the better
        # log(x) with 0<x<1 is negative, so to reduce this loss we have to minimize the opposite
        loss: torch.Tensor = -(loss_1 + loss_2)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.item()

    def train(self, data) -> None:
        print("please use train_with_saved_data instead")

    
    def train_with_saved_data(self) -> None:
        """
        Overview:
            Trains the GAIL reward model. The training and expert data are randomly sampled with a batch size 
            specified in the 'self.cfg' attribute. The 'self.train_data' and 'self.expert_data' attributes 
            are used for this purpose.
        Effects:
            This function updates the GAIL reward model and increments the training iteration count.
        """
        for _ in range(self.cfg.update_per_collect):
            # Randomly sample training data from 'self.train_data'
            sample_train_data: list = random.sample(self.train_data, self.cfg.batch_size)
            # Convert the list of tensors to a single tensor and move it to the device
            sample_train_data = torch.stack(sample_train_data).to(self.device)

            # Randomly sample expert data from 'self.train_expert_data'
            indices = torch.randperm(self.train_expert_data.shape[0])[:self.cfg.batch_size]
            sample_expert_data = self.train_expert_data[indices].to(self.device)

            # Calculate the loss using the sampled training and expert data
            loss = self.learn(sample_train_data, sample_expert_data)

            # Log the loss to tensorboard
            self.tb_logger.add_scalar('reward_model/gail_loss', loss, self.train_iter)
            
            # Increment the training iteration count
            self.train_iter += 1
    
    def estimate(self, data: List) -> Any:
        print("please use estimate_dist instead")
    
    def estimate_dist(self, top_percentage: float = 1.0, multi_normal=False,args=None,new_bc_estimate_dist=None,collector=None) -> List[Dict]:
        """
        Overview:
            This function estimates the new distribution with the trained reward model.
        Arguments:
            - top_percentage (:obj:`float`): The percentage of top reward_weight to consider for distribution calculation.
            - data (:obj:`list`): the list of data used for estimation, with at least \
                ``obs`` and ``action`` keys.
        Effects:
            - This function modifies the reward values in place.
        """
        estimate_dist = args.estimate_dist

        # Move validation expert data to the device
        val_expert_data=self.val_expert_data.to(self.device) # [100,14]

        # Calculate reward weights based on the reward model
        if(args.reward_on == 'both'):
            # Prepare state and validation adversary data
            state = val_expert_data[:,0:self.s_shape]
            val_adv_data = torch.stack(self.concat_state_action_pairs(collector.collect(state=state))).to(self.device)
            with torch.no_grad():
                # Calculate reward weights for both vic and adv
                rew_weight_vic = -torch.log(self.reward_model(val_expert_data).squeeze(-1).cpu()+ 1e-8)
                rew_weight_adv = -torch.log(self.reward_model(val_adv_data).squeeze(-1).cpu()+ 1e-8)
                # Calculate final reward weight
                rew_weight = torch.clip((rew_weight_vic - rew_weight_adv) / (rew_weight_vic + rew_weight_adv) *2,min=0)
        elif(args.reward_on == 'adv'):
            # Prepare new adversary data
            new_adv_data = collector.collect(n_sample=20000)
            new_adv_data = self.concat_state_action_pairs(new_adv_data)
            val_expert_data=torch.stack(new_adv_data).to(self.device)
            with torch.no_grad():
                # Calculate reward weight for adv
                rew_weight = -torch.log(self.reward_model(val_expert_data).squeeze(-1).cpu()+ 1e-8)
        elif(args.reward_on == 'vic'):  
            with torch.no_grad():
                # Calculate reward weight for vic
                rew_weight = -torch.log(self.reward_model(val_expert_data).squeeze(-1).cpu()+ 1e-8)

        # If estimate_dist is "p_test", append the mean reward weight to reward_list
        if(args.estimate_dist == "p_test"):
            args.reward_list.append(rew_weight.mean())

        # Add histogram of reward weights to tensorboard logger
        self.tb_logger.add_histogram('reward_model/reward_weight', rew_weight, self.train_iter/self.cfg.update_per_collect)

        # Calculate threshold and mask for top percentage of reward weights
        threshold = torch.quantile(rew_weight, 1-top_percentage)
        mask = rew_weight >= threshold

        # Apply the mask to keep the original order
        val_expert_data = val_expert_data[:,:self.s_shape].cpu()
        rew_weight_mask = rew_weight[mask]
        val_expert_data_mask = val_expert_data[mask, :]

        # Normalize the weights
        normalized_weights = rew_weight / torch.sum(rew_weight)
        normalized_weights_mask = rew_weight_mask / torch.sum(rew_weight_mask)

        # Calculate the weighted mean using top percentage of data
        weighted_mean = torch.sum(val_expert_data_mask * normalized_weights_mask.unsqueeze(1), dim=0)
        if(new_bc_estimate_dist is not None):
            weighted_mean = copy.deepcopy(new_bc_estimate_dist.mean.data)
            
        # If estimate_dist is "uniform_dist", return a uniform distribution
        if(estimate_dist=="uniform_dist"):
            estimate_range = 3
            estimate_dist=torch.distributions.Uniform(weighted_mean-estimate_range, weighted_mean+estimate_range)
            return estimate_dist
        
        # If multi_normal is True, calculate the weighted covariance and return a multivariate normal distribution
        if multi_normal:
            centered_data = val_expert_data - weighted_mean
            outer_product = torch.einsum('ij,ik->ijk', centered_data, centered_data)
            weighted_covariance = torch.sum(normalized_weights.view(-1, 1, 1) * outer_product, dim=0)
            self.tb_logger.add_scalar('reward_model/weighted_cov', weighted_covariance.mean().item(), self.train_iter/self.cfg.update_per_collect)
            estimate_dist = torch.distributions.MultivariateNormal(weighted_mean,weighted_covariance)
        else:
            # Calculate the weighted variance and standard deviation, and return a normal distribution
            weighted_variance = torch.sum(normalized_weights.unsqueeze(1) * (val_expert_data - weighted_mean) ** 2, dim=0)
            weighted_std = torch.sqrt(weighted_variance)
            self.tb_logger.add_scalar('reward_model/weighted_std', weighted_std.mean().item(), self.train_iter/self.cfg.update_per_collect)
            estimate_dist = torch.distributions.Normal(weighted_mean,weighted_std)

        return estimate_dist

    def collect_data(self, data: list) -> None:
        """
        Overview:
            Collecting training data formatted by  ``fn:concat_state_action_pairs``.
        Arguments:
            - data (:obj:`Any`): Raw training data (e.g. some form of states, actions, obs, etc)
        Effects:
            - This is a side effect function which updates the data attribute in ``self``
        """
        self.train_data.extend(self.concat_state_action_pairs(data))
        # self.train_data = torch.cat([data['obs'],data['action']],dim=-1)


    def clear_data(self) -> None:
        """
        Overview:
            Clearing training data. \
            This is a side effect function which clears the data attribute in ``self``
        """
        self.train_data.clear()
