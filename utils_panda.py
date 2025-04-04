
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

import os
import torch
import numpy as np
from rl_zoo3.utils import get_model_path
from rl_zoo3 import ALGOS, create_test_env, get_saved_hyperparams

class panda_evaluator:
    def __init__(self,env) -> None:
        self.env = env

    def eval(self,all_timestep=1000,model=None,type='sb3',tb_logger=None,name=None,global_epoch=0,global_budget=0, args=None, record_video=False):
        env = self.env
        lstm_states = None
        episode_start = np.ones((env.num_envs,), dtype=bool)
        successes = []
        episode_reward = 0.0
        episode_rewards, episode_lengths = [], []
        ep_len = 0
        obs = env.reset()

        if(record_video==True):
            images = [env.render()]
        else:
            images= None
        time_step=0
        while(time_step<all_timestep):
            time_step += 1
            if(type=='sb3'):
                action, lstm_states = model.predict(
                    obs,
                    state=lstm_states,
                    episode_start=episode_start,
                    deterministic=True,
                )
            elif(type=='ding'):
                with torch.no_grad():
                    stack_obs = np.concatenate([obs['achieved_goal'].flatten(),
                                                obs['desired_goal'].flatten(),
                                                obs['observation'].flatten()], axis=0).reshape(1,-1)
                    
                    stack_obs = torch.from_numpy(stack_obs)
                    if(args!=None):
                        if(args.normalize_input==True): 
                            stack_obs = (stack_obs - args.means) / args.stds
                    stack_obs= stack_obs.cuda()
                    logit = model(stack_obs, 'compute_actor')['logit']
                    mu,sigma = logit
                    action = torch.tanh(mu).cpu().numpy()
            obs, reward, done, infos = env.step(action)



            episode_start = done

            episode_reward += reward[0]
            ep_len += 1

            if done:
                episode_rewards.append(episode_reward)
                episode_lengths.append(ep_len)
                episode_reward = 0.0
                ep_len = 0
                if infos[0].get("is_success") is not None:
                    successes.append(infos[0].get("is_success", False))

            if(record_video==True):
                images.append(env.render())
                if(len(episode_rewards)==8):
                    break
        success_rate = 100 * np.mean(successes)
        num_episodes = len(episode_rewards)
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        mean_episode_length = np.mean(episode_lengths)
        std_episode_length = np.std(episode_lengths)

        if(tb_logger!=None):
            tb_logger.add_scalar(f"{name}_epoch/success_rate", success_rate,global_epoch)
            tb_logger.add_scalar(f"{name}_epoch/number_of_episodes", num_episodes,global_epoch)
            tb_logger.add_scalar(f"{name}_epoch/return_mean", mean_reward,global_epoch)
            tb_logger.add_scalar(f"{name}_epoch/return_std", std_reward,global_epoch)
            tb_logger.add_scalar(f"{name}_epoch/episode_length_mean", mean_episode_length,global_epoch)
            tb_logger.add_scalar(f"{name}_epoch/episode_length_std", std_episode_length,global_epoch)

            tb_logger.add_scalar(f"{name}_budget/success_rate", success_rate,global_budget)
            tb_logger.add_scalar(f"{name}_budget/number_of_episodes", num_episodes,global_budget)
            tb_logger.add_scalar(f"{name}_budget/return_mean", mean_reward,global_budget)
            tb_logger.add_scalar(f"{name}_budget/return_std", std_reward,global_budget)
            tb_logger.add_scalar(f"{name}_budget/episode_length_mean", mean_episode_length,global_budget)
            tb_logger.add_scalar(f"{name}_budget/episode_length_std", std_episode_length,global_budget)
        print(
            f"{name}_budget/success_rate", success_rate,global_budget,'\n',
            f"{name}_budget/number_of_episodes", num_episodes,global_budget,'\n',
            f"{name}_budget/return_mean", mean_reward,global_budget,'\n',
            f"{name}_budget/return_std", std_reward,global_budget,'\n',
            f"{name}_budget/episode_length_mean", mean_episode_length,global_budget,'\n',
            f"{name}_budget/episode_length_std", std_episode_length,global_budget
        )
        return successes, episode_rewards,episode_lengths, images

def load_victim_policy_eval_panda(env_name,algo,folder,seed=0, load_checkpoint=None):
    # Load victim model, evaluate victim model and return the victim model

    # Load the config for victim model
    _, model_path, log_path = get_model_path(
        exp_id=0, # 0 indicate loading latest experiment
        folder=folder,
        algo=algo,
        env_name=env_name,
        load_checkpoint=load_checkpoint
    )

    stats_path = os.path.join(log_path, env_name)
    hyperparams, stats_path = get_saved_hyperparams(stats_path, norm_reward=False, test_mode=True)

    env = create_test_env( # Create test env
        env_name.gym_id,
        n_envs=1,
        stats_path=stats_path,
        seed=seed,
        log_dir=None,
        should_render=False,
        hyperparams=hyperparams,
        env_kwargs={},
    )

    kwargs = dict(seed=seed)
    kwargs.update(dict(buffer_size=1))
    if "optimize_memory_usage" in hyperparams:
        kwargs.update(optimize_memory_usage=False)

    custom_objects = {
        "learning_rate": 0.0,
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.0,
    }
    model = ALGOS[algo].load(model_path, env=env, custom_objects=custom_objects, device="cuda", **kwargs)
    evaluator = panda_evaluator(env)
    successes, episode_rewards,episode_lengths,_ = evaluator.eval(all_timestep=1000,model=model,name="victim")
    success_rate = 100 * np.mean(successes)
    return_mean = np.mean(episode_rewards)
    return_std = np.std(episode_rewards)

    return model, success_rate,return_mean,return_std, env,log_path


def construct_transfer_dataset_panda(global_budget, vic_model,query_budget, estimate_dist,args, train_val_ratio=0.8,sampler=None,cfg=None):

    global_budget += query_budget

    if(sampler==None):
        query = estimate_dist.sample((query_budget,))
    else:
        sampler.update_distribution(estimate_dist)
        query = sampler.get_samples(query_budget)

    chunks = torch.chunk(query, 20) # Split bc_query into 20 chunks

    label_list = []  # Initialize a empty list to save bc_label for each chunks
    
    with torch.no_grad():
        for chunk in chunks:
            chunk = chunk.cuda()
            # (mu,sigma) = vic_model(chunk,'compute_actor')['logit']
            # label_chunk = torch.tanh(mu)

            achieved_goal = chunk[:, :3]
            desired_goal = chunk[:, 3:6]
            observation = chunk[:, 6:]

            data_dict = {
                'achieved_goal': achieved_goal,
                'desired_goal': desired_goal,
                'observation': observation
            }

            label_chunk = vic_model.actor(data_dict, deterministic=True)

            label_list.append(label_chunk.cpu())

    # 将所有的bc_label合并起来
    label = torch.cat(label_list, dim=0) 
    train_size = int(train_val_ratio * query_budget)
    current_data = {"train_query": query[:train_size],
                    "train_label": label[:train_size],
                    "val_query": query[train_size:],
                    "val_label": label[train_size:]}


    return current_data,global_budget

def add_noise_to_zero_std_dims(obs_data):
    """
    Adds noise to the dimensions of obs_data with zero standard deviation.
    
    Parameters:
    - obs_data (torch.Tensor): The input data tensor.
    
    Returns:
    - obs_data (torch.Tensor): The modified data tensor with noise added to dimensions with zero standard deviation.
    - zero_std_dims (list): A list of dimensions that had zero standard deviation.
    """

    std = obs_data['obs_data'].std(dim=0)
    zero_std_dims = (std == 0).nonzero(as_tuple=True)[0].tolist()

    zero_std_dims_values = {dim: obs_data['obs_data'][0, dim].item() for dim in zero_std_dims}
    obs_data['obs_data'] = obs_data['obs_data'] + np.random.normal(0, 1e-7, obs_data['obs_data'].shape)

    obs_data['mean'] = obs_data['obs_data'].mean(dim=0)
    obs_data['std'] = obs_data['obs_data'].std(dim=0)

    obs_data_mean_subtracted = obs_data['obs_data'] - obs_data['mean']
    obs_data['covariance'] = obs_data_mean_subtracted.t().mm(obs_data_mean_subtracted) / (obs_data['obs_data'].shape[0] - 1)
        
    obs_data['zero_std_dims_values'] = zero_std_dims_values 
    
    return obs_data
