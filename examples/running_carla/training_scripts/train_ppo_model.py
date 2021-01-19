import numpy as np
import carla
import argparse
import logging
import time
import random
import sys
import torch
import wandb
import copy
import carla

sys.path.append("/scratch/cluster/stephane/Carla_RL/examples/running_carla/Carla/")
from carla_env import CarlaEnv
sys.path.append("/scratch/cluster/stephane/Carla_RL/examples/running_carla/model_scripts/")
from ppo import PPO_Agent

global device

class Memory():
    def __init__(self):
        #data for entire rollout
        self.rewards = []
        self.eps_frames = []
        self.eps_frames_raw = []
        self.eps_mes = []
        self.eps_mes_raw = []
        self.actions = []
        self.actions_log_probs = []
        self.states_p = []
        self.terminals = []
        self.advantages = []

        #single episdoe data
        self.se_rewards = []
        self.se_eps_frames = []
        self.se_eps_frames_raw = []
        self.se_eps_mes = []
        self.se_eps_mes_raw = []
        self.se_actions = []
        self.se_actions_log_probs = []
        self.se_states_p = []
        self.se_terminals = []

    def add(self,frame,mes,raw_frame,raw_mes,a,a_log_prob,reward,s_prime,done):
        self.eps_frames.append(frame.detach().clone())
        self.eps_frames_raw.append(copy.deepcopy(raw_frame))
        self.eps_mes.append(mes.detach().clone())
        self.eps_mes_raw.append(copy.deepcopy(raw_mes))
        self.actions.append(a.detach().clone())
        self.actions_log_probs.append(a_log_prob.detach().clone())
        self.rewards.append(copy.deepcopy(reward))
        self.states_p.append(copy.deepcopy(s_prime))
        self.terminals.append(copy.deepcopy(done))

    def se_add(self,frame,mes,raw_frame,raw_mes,a,a_log_prob,reward,s_prime,done):
        self.se_eps_frames.append(frame.detach().clone())
        self.se_eps_frames_raw.append(copy.deepcopy(raw_frame))
        self.se_eps_mes.append(mes.detach().clone())
        self.se_eps_mes_raw.append(copy.deepcopy(raw_mes))
        self.se_actions.append(a.detach().clone())
        self.se_actions_log_probs.append(a_log_prob.detach().clone())
        self.se_rewards.append(copy.deepcopy(reward))
        self.se_states_p.append(copy.deepcopy(s_prime))
        self.se_terminals.append(copy.deepcopy(done))

    def add_advantages(self,advantages):
        self.advantages.extend(advantages)

    def clear (self):
        #self.rewards = list(self.rewards.numpy())
        #self.actions_log_probs = list(self.actions_log_probs.numpy())

        self.rewards.clear()
        self.eps_frames.clear()
        self.eps_frames_raw.clear()
        self.eps_mes.clear()
        self.eps_mes_raw.clear()
        self.actions.clear()
        self.actions_log_probs.clear()
        self.states_p.clear()
        self.terminals.clear()
        self.advantages.clear()

    def se_clear(self):
        #self.se_rewards = list(self.se_rewards.numpy())
        #self.se_actions_log_probs = list(self.se_actions_log_probs.numpy())

        self.se_rewards.clear()
        self.se_eps_frames.clear()
        self.se_eps_frames_raw.clear()
        self.se_eps_mes.clear()
        self.se_eps_mes_raw.clear()
        self.se_actions.clear()
        self.se_actions_log_probs.clear()
        self.se_states_p.clear()
        self.se_terminals.clear()


    def reservoir_sample(self,k):
        eps_frames_reservoir = []
        eps_mes_reservoir = []
        actions_reservoir = []
        actions_log_probs_reservoir = []
        rewards_reservoir = []
        terminals_reservoir = []
        advantages_reservoir = []

        for i in range (len(self.eps_frames)):
            if len(eps_frames_reservoir) < k:
                eps_frames_reservoir.append(self.eps_frames[i])
                eps_mes_reservoir.append(self.eps_mes[i])
                actions_reservoir.append(self.actions[i])
                actions_log_probs_reservoir.append(self.actions_log_probs[i])
                rewards_reservoir.append(self.rewards[i])
                terminals_reservoir.append(self.terminals[i])
                advantages_reservoir.append(self.advantages[i])
            else:
                j = int(random.uniform(0,i))
                if j < k:
                    eps_frames_reservoir[j] = self.eps_frames[i]
                    eps_mes_reservoir[j] = self.eps_mes[i]
                    actions_reservoir[j] = self.actions[i]
                    actions_log_probs_reservoir[j] = self.actions_log_probs[i]
                    rewards_reservoir[j] = self.rewards[i]
                    terminals_reservoir[j] = self.terminals[i]
                    advantages_reservoir[j] = self.advantages[i]
        return eps_frames_reservoir,eps_mes_reservoir,actions_reservoir,actions_log_probs_reservoir,rewards_reservoir,terminals_reservoir,advantages_reservoir

def train_model(args):
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

    #init models
    policy = PPO_Agent(n_states, n_actions, action_std,lr, gamma, n_epochs,clip_val,device).to(device)
    prev_policy = PPO_Agent(n_states, n_actions, action_std,lr, gamma, n_epochs,clip_val,device).to(device)
    prev_policy.load_state_dict(policy.state_dict())
    memory = Memory()

    #start WANDB logging
    wandb.init(project='PPO_Carla_Navigation')
    config = wandb.config
    config.learning_rate = lr
    wandb.watch(prev_policy)

    batch_ep_returns = []
    timestep_mod = 0
    train_iters = 0
    total_timesteps = 0
    update_timestep = 2000
    prev_batch_return = 0
    save_video = False

    for iters in range(n_iters):
        with CarlaEnv(args,save_video=save_video) as env:
            s, _, _, _ = env.reset(False, iters)
            t = 0
            episode_return = 0
            done = False

            while not done:
                frame,mes = prev_policy.format_state(s)
                a, a_log_prob = prev_policy.choose_action(frame,mes)
                s_prime, reward, done, info = env.step(action=a.detach().tolist(), timeout=2)

                memory.add(frame, mes,s[0],s[1:],a,a_log_prob,reward,s_prime,done)
                memory.se_add(frame, mes,s[0],s[1:],a,a_log_prob,reward,s_prime,done)

                s = copy.deepcopy(s_prime)
                t += 1
                total_timesteps +=1
                episode_return += reward

            #compute episode advtanages using the single episodes collected data
            memory.add_advantages(policy.compute_advantages(memory))
            #clear single episodes collected data
            memory.se_clear()

            # TODO change this hack to calculate when PPO training is triggered, look at PPO batch
            batch_ep_returns.append(episode_return)
            prev_timestep_mod = timestep_mod
            timestep_mod = total_timesteps // update_timestep

            if timestep_mod > prev_timestep_mod:
                prev_policy = policy.train(memory, prev_policy,iters)

                avg_batch_ep_returns = sum(batch_ep_returns)/len(batch_ep_returns)
                moving_avg = (avg_batch_ep_returns - moving_avg) * (2 / (train_iters + 2)) + avg_batch_ep_returns
                train_iters += 1
                batch_ep_returns.clear()

                if avg_batch_ep_returns - prev_batch_return > 0.3:
                    save_video = True
                else:
                    save_video = False

                prev_batch_return = avg_batch_ep_returns

                wandb.log({
                    "avg batch return": avg_batch_ep_returns,
                    "exponential avg return": moving_avg,
                    "percent_completed": info[0],
                    "number_of_collisions": info[1],
                    "number_of_trafficlight_violations": info[2],
                    "number_of_stopsign_violations": info[3],
                    "number_of_route_violations": info[4],
                    "number_of_times_vehicle_blocked": info[5],
                    "timesteps before termination": t,
                    'Sample image': [
                        wandb.Image(memory.eps_frames_raw[img][0], caption=f'Img#: {len(memory.eps_frames_raw) * (-img)}') for img in
                        [0, -1]],
                })
                memory.clear()

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
    parser.add_argument('--n-vehicles', type=int, default=1)
    parser.add_argument('--client-timeout', type=int, default=10)

    main(parser.parse_args())
