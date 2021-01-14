import collections
import queue
import numpy as np
import cv2
import carla
import argparse
import logging
import time
import math
import random
import sys
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import MultivariateNormal
import wandb
import copy
import carla

from carla_env import CarlaEnv
sys.path.append("model_scripts/")
from ppo import PPO_Agent

global device

class Memory():
    rewards = []
    eps_frames = []
    eps_frames_raw = []
    eps_mes = []
    eps_mes_raw = []
    actions = []
    actions_log_probs = []
    states_p = []
    terminals = []
    batch_ep_returns = []

    def add(self,s,a,a_log_prob,reward,s_prime,done):
        eps_frames.append(format_frame(s[0]).detach().clone())
        eps_frames_raw.append(copy.deepcopy(s[0]))
        eps_mes.append(format_mes(s[1:]).detach().clone())
        eps_mes_raw.append(copy.deepcopy(s[1:]))
        actions.append(a.detach().clone())
        actions_log_probs.append(a_log_prob.detach().clone())
        rewards.append(copy.deepcopy(reward))
        states_p.append(copy.deepcopy(s_prime))
        terminals.append(copy.deepcopy(done))

    def clear (self):
        rewards = list(rewards.numpy())
        actions_log_probs = list(actions_log_probs.numpy())

        rewards.clear()
        eps_frames.clear()
        eps_frames_raw.clear()
        eps_mes.clear()
        eps_mes_raw.clear()
        actions.clear()
        actions_log_probs.clear()
        states_p.clear()
        terminals.clear()
        batch_ep_returns.clear()

def train_model(args):
    wandb.init(project='PPO_Carla_Navigation')
    n_iters = 10000
    n_epochs = 50
    max_steps = 2000
    gamma = 0.99
    lr = 0.0001
    clip_val = 0.2
    avg_t = 0
    moving_avg = 0
    n_states = 11
    #currently the action array will be [throttle, steer]
    n_actions = 2
    action_std = 0.5

    config = wandb.config
    config.learning_rate = lr

    #init models
    policy = PPO_Agent(n_states, n_actions, action_std,lr, gamma, n_epochs,clip_val).to(device)
    prev_policy = PPO_Agent(n_states, n_actions, action_std,lr, gamma, n_epochs,clip_val).to(device)
    prev_policy.load_state_dict(policy.state_dict())
    memory = Memory()

    wandb.watch(prev_policy)

    for iters in range(n_iters):
        with CarlaEnv(args,save_video=False) as env:
            s, _, _, _ = env.reset(False, iters)
            t = 0
            episode_return = 0
            done = False

            while not done:
                a, a_log_prob = prev_policy.choose_action(policy.format_state(s))
                s_prime, reward, done, info = env.step(action=a.detach().tolist(), timeout=2)

                memory.add(s,a,a_log_prob,reward,s_prime,done)

                s = copy.deepcopy(s_prime)
                t += 1
                episode_return += reward

            # TODO change this hack to calculate when PPO training is triggered, look at PPO batch
            batch_ep_returns.append(episode_return)
            prev_timestep_mod = timestep_mod
            timestep_mod = total_timesteps // update_timestep

            if timestep_mod > prev_timestep_mod:
                prev_policy = policy.train(memory, prev_policy)
                memory.clear()

                avg_batch_ep_returns = sum(batch_ep_returns)/len(batch_ep_returns)
                moving_avg = (avg_batch_ep_returns - moving_avg) * (2 / (train_iters + 2)) + avg_batch_ep_returns
                train_iters += 1

                wandb.log({
                    "episode_return (suggested reward w/ ri)": avg_batch_ep_returns,
                    "average_return (suggested reward w/ ri)": moving_avg,
                    "percent_completed": info[0],
                    "number_of_collisions": info[1],
                    "number_of_trafficlight_violations": info[2],
                    "number_of_stopsign_violations": info[3],
                    "number_of_route_violations": info[4],
                    "number_of_times_vehicle_blocked": info[5],
                    "timesteps before termination": t,
                    'Sample image': [
                        wandb.Image(eps_frames_raw[img][0], caption=f'Img#: {len(eps_frames_raw) * (-img)}') for img in
                        [0, -1]],
                })

def launch_client(args):
    client = carla.Client(args.host, args.world_port)
    client.set_timeout(args.client_timeout)
    return client

def main(args):
    global device
    # set GPU to be used for policy gradient backprop
    device = torch.device(f"cuda:{args.client_gpu}" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    # Create client outside of Carla environment to avoid creating zombie clients
    args.client = launch_client(args)
    train_model(args)
    # random_baseline(host,world_port)
    # run_model(host,world_port)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--client-gpu', type=int, default=0)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--world-port', type=int, required=True)
    parser.add_argument('--tm-port', type=int, required=True)
    parser.add_argument('--n_vehicles', type=int, default=1)
    parser.add_argument('--client_timeout', type=int, default=10)

    main(parser.parse_args())
