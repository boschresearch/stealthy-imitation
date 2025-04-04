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
import logging

import torch
import numpy as np
from numpngw import write_apng
from ding.policy import create_policy
from ding.config import read_config, compile_config

import panda_gym
from huggingface_sb3 import EnvironmentName
from utils_panda import panda_evaluator, load_victim_policy_eval_panda

logger = logging.getLogger(__name__)

def evaluate_panda(all_timestep=1000,model=None,env=None,record_video=False):
    lstm_states = None
    episode_start = np.ones((env.num_envs,), dtype=bool)
    successes = []
    episode_reward = 0.0
    episode_rewards, episode_lengths = [], []
    ep_len = 0
    generator = range(all_timestep)
    obs = env.reset()
    if(record_video==True):
        images = [env.render()]
    else:
        images= None
    for _ in generator:
        action, lstm_states = model.predict(
            obs,
            state=lstm_states,
            episode_start=episode_start,
            deterministic=True,
        )

        obs, reward, done, infos = env.step(action)
        if(record_video==True):
            images.append(env.render())
        episode_start = done

        episode_reward += reward[0]
        ep_len += 1

        if done:
            # NOTE: for env using VecNormalize, the mean reward
            # is a normalized reward when `--norm_reward` flag is passed
            print(f"Episode Reward: {episode_reward:.2f}")
            print("Episode Length", ep_len)
            episode_rewards.append(episode_reward)
            episode_lengths.append(ep_len)
            episode_reward = 0.0
            ep_len = 0
            if infos[0].get("is_success") is not None:
                successes.append(infos[0].get("is_success", False))
        if(len(episode_rewards)==8):
            break


    return successes, episode_rewards,episode_lengths, images

def collect_panda(all_timestep=1000,model=None,env=None):
    data={'obs':[],'action':[]}
    lstm_states = None
    episode_start = np.ones((env.num_envs,), dtype=bool)

    generator = range(all_timestep) #
    obs = env.reset()
    for _ in generator:
        stack_obs = np.concatenate([obs['achieved_goal'].flatten(),
                                    obs['desired_goal'].flatten(),
                                    obs['observation'].flatten()], axis=0).reshape(1,-1)
        data['obs'].append(torch.from_numpy(stack_obs))
        action, lstm_states = model.predict(
            obs,
            state=lstm_states,
            episode_start=episode_start,
            deterministic=True,
        )
        data['action'].append(torch.from_numpy(action))
        obs, reward, done, infos = env.step(action)

        episode_start = done

    return data

def main(args):
    
    # load the victim policy
    vic_model, success_rate,return_mean,return_std,env,log_path = load_victim_policy_eval_panda(args.env,args.algo,args.folder,seed=0)

    # Configure logging to output DEBUG and higher level messages to both a file and the console
    logging.basicConfig(filename=f'{log_path}/logger.log', filemode='a', level=logging.DEBUG, force=True)
    console_handler = logging.StreamHandler(sys.stdout)
    logging.getLogger().addHandler(console_handler)

    if(args.model_type=="sb3"):
        model = vic_model
    else:
        # load the trained attacker policy to record video
        log_path = args.model_cfg
        model_cfg = f"{args.model_cfg}/formatted_total_config.py"
        cfg, create_cfg = read_config(model_cfg)
        create_cfg.policy.type = create_cfg.policy.type + '_command'
        cfg = compile_config(cfg, seed=args.seed, env=None, auto=True, create_cfg=create_cfg, save_cfg=False)
        policy = create_policy(cfg.policy, model=None, enable_field=['eval', 'learn','collect','command'])
        model_path = f"{args.model_cfg}/ckpt/best_val_loss.pth.tar"
        policy._model.load_state_dict(torch.load(model_path, map_location='cpu')['model'],strict=True)
        model = policy._model

    if(args.record_video):
        evaluator = panda_evaluator(env)
        success_rate, episode_rewards,episode_lengths,images = evaluator.eval(model=model,type=args.model_type,record_video=args.record_video, name="victim" if args.model_type=="sb3" else "attacker")
        write_apng(log_path+"/anim.png", images, delay=40)
        return

    data = collect_panda(all_timestep=args.n_sample,model=model,env=env)
    obs_data = torch.cat(data['obs'],dim=0)
    action_data = torch.cat(data['action'],dim=0)

    # Covariance
    obs_data_mean_subtracted = obs_data - obs_data.mean(dim=0)
    covariance = obs_data_mean_subtracted.t().mm(obs_data_mean_subtracted) / (obs_data.shape[0] - 1)

    # Save mean, covariance, PCA model and transformed data
    torch.save({
        'obs_data':obs_data,
        'mean': obs_data.mean(dim=0),
        'std':obs_data.std(dim=0),
        'covariance': covariance
    }, log_path+f'/obs_data_{args.n_sample}.pt')


    def tensor_to_string(tensor):
        return ', '.join(f"{value:.2f}" for value in tensor)
    
    logger.info(f"Success rate: {success_rate:.2f}%")
    logger.info(f"Mean reward: {return_mean:.2f} +/- {return_std:.2f}")
    logger.info("covariance")
    logger.info(covariance)

    logger.info(f"policy_action_mean: [{tensor_to_string(action_data.mean(dim=0))}]")
    logger.info(f"policy_action_std: [{tensor_to_string(action_data.std(dim=0))}]")
    logger.info(f"policy_action_max: [{tensor_to_string(action_data.max(dim=0)[0])}]")
    logger.info(f"policy_action_min: [{tensor_to_string(action_data.min(dim=0)[0])}]")

    logger.info(f"obs_state_std: [{tensor_to_string(obs_data.std(dim=0))}]")
    logger.info(f"obs_state_mean: [{tensor_to_string(obs_data.mean(dim=0))}]")
    logger.info(f"obs_state_min: [{tensor_to_string(obs_data.min(dim=0)[0])}]")
    logger.info(f"obs_state_max: [{tensor_to_string(obs_data.max(dim=0)[0])}]")

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This script is used to analyze a policy in a reinforcement learning environment.")

    parser.add_argument("--folder", type=str, default="exp/panda",
                        help="Path to the directory where the policy and its configuration are stored.")

    parser.add_argument("--algo", type=str, default="tqc",
                        help="Algorithm used for training the policy.")

    parser.add_argument("--env", type=EnvironmentName, default="PandaReach-v3",
                        help="ID of the environment to be used.")

    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for random number generation to ensure reproducibility.")

    parser.add_argument("--n_sample", type=int, default=100,
                        help="Number of samples to collect for analysis.")

    parser.add_argument("--visualize", type=int, default=0,
                        help="Set to 1 to visualize the obs_data in kde, 0 otherwise.")

    parser.add_argument("--record_video", type=int, default=0,
                        help="Set to 1 to record a video of the policy's behavior, 0 otherwise.")

    parser.add_argument("--model_type", type=str, choices=["sb3","ding"], default="sb3",
                        help="Type of the model to be used. sb3 is the victim policy, ding is the attacker policy.")

    parser.add_argument("--model_cfg", type=str, default=None,
                        help="Configuration of the model.")

    args = parser.parse_args()

    main(args)

