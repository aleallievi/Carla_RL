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
from torch.distributions import kl

class PPO_Agent(nn.Module):
    def __init__(self, linear_state_dim, action_dim, action_std,lr, gamma, n_epochs,clip_val,device):
        """
        Initializes PPO actor critic models
        """

        super(PPO_Agent, self).__init__()
        # action mean range -1 to 1
        self.actorConv = nn.Sequential(
                nn.Conv2d(3, 6, 5),
                nn.Tanh(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(6, 12, 5),
                nn.Tanh(),
                nn.MaxPool2d(2, 2),
                nn.Flatten()
                )
        self.actorLin = nn.Sequential(
                nn.Linear(12*17*17 + linear_state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 32),
                nn.Tanh(),
                nn.Linear(32, action_dim),
                nn.Tanh()
                )

        self.criticConv = nn.Sequential(
                nn.Conv2d(3, 6, 5),
                nn.Tanh(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(6, 12, 5),
                nn.Tanh(),
                nn.MaxPool2d(2, 2),
                nn.Flatten()
                )
        self.criticLin = nn.Sequential(
                nn.Linear(12*17*17 + linear_state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 32),
                nn.Tanh(),
                nn.Linear(32, 1)
                )
        self.device = device
        self.action_var = torch.full((action_dim,), action_std*action_std).to(self.device)

        self.optimizer = Adam(self.parameters(), lr=lr)
        self.mse = nn.MSELoss()
        self.lr = lr
        self.gamma = gamma
        self.n_epochs = n_epochs
        self.clip_val = clip_val

    def actor(self, frame, mes):
        """
        Input: formatted frame and measurements
        Output: mean of action probability distribution (as a tensor)
        """

        frame = frame.to(self.device)
        mes = mes.to(self.device)
        if len(list(mes.size())) == 1:
            mes = mes.unsqueeze(0)
        vec = self.actorConv(frame)
        X = torch.cat((vec, mes), 1)
        return self.actorLin(X)

    def critic(self, frame, mes):
        """
        Input: formatted frame and measurements
        Output: value of current state (as a tensor)
        """

        frame = frame.to(self.device)
        mes = mes.to(self.device)
        if len(list(mes.size())) == 1:
            mes = mes.unsqueeze(0)
        vec = self.criticConv(frame)
        X = torch.cat((vec, mes), 1)
        return self.criticLin(X)

    def choose_action(self, frame, mes):
        """
        Input: formatted frame and measurements
        Output: formatted action and action log probability (as tensors)
        """

        with torch.no_grad():
            mean = self.actor(frame, mes)
            cov_matrix = torch.diag(self.action_var).to(self.device)
            gauss_dist = MultivariateNormal(mean, cov_matrix)
            action = gauss_dist.sample()
            action_log_prob = gauss_dist.log_prob(action)
        return action, action_log_prob

    def get_training_params(self, frame, mes, action):
        """
        Input: formatted frame and measurements, and action
        Output: action log probabilities, state values, and distribution entropy (as tensors)
        """

        frame = torch.squeeze(torch.stack(frame))
        mes = torch.squeeze(torch.stack(mes))
        action = torch.stack(action)

        mean = self.actor(frame, mes)
        action_expanded = self.action_var.expand_as(mean)
        cov_matrix = torch.diag_embed(action_expanded).to(self.device)

        gauss_dist = MultivariateNormal(mean, cov_matrix)
        action_log_prob = gauss_dist.log_prob(action).to(self.device)
        entropy = gauss_dist.entropy().to(self.device)
        state_value = torch.squeeze(self.critic(frame, mes)).to(self.device)
        return action_log_prob, state_value, entropy

    def format_frame(self,frame):
        """
        Input: raw RGB image (formatted as a nparray)
        Output: formatted frame as a tensor, shape  = (1,c,h,w)
        """
        frame = torch.FloatTensor(frame.copy())
        _, h, w, c = frame.shape
        frame = frame.unsqueeze(0).view(1, c, h, w)
        return frame

    def format_mes(self,mes):
        """
        Input: list of measurements
        Output: measurements (as tensor)
        """
        mes = torch.FloatTensor(mes)
        return mes

    def format_state (self,s):
        """
        Input: raw state (nparray image and list of measurements)
        Output: frame and measurements (as tensors)
        """

        return self.format_frame(s[0]), self.format_mes(s[1:])

    def discount_rewards(self,r, gamma, terminals):
        """ take 1D float array of rewards and compute discounted reward """
        # from https://github.com/GoogleCloudPlatform/tensorflow-without-a-phd/blob/master/tensorflow-rl-pong/trainer/helpers.py
        r = np.array(r)
        rev_terminals = terminals[::-1]
        discounted_r = np.zeros_like(r)
        running_add = 0
        for t in reversed(range(0, r.size)):
            if rev_terminals[t]:
                running_add = 0
            running_add = running_add * self.gamma + r[t]
            discounted_r[t] = running_add
        return discounted_r.tolist()

    def compute_advantages(self, memory):
        """
        Input: memory object
        Output: normalized advantage (as list)
        """
        current_action_log_probs, state_values, entropies = self.get_training_params(memory.se_eps_frames, memory.se_eps_mes, memory.se_actions)
        returns = self.discount_rewards(memory.se_rewards, self.gamma, memory.se_terminals)
        returns = torch.tensor(returns).to(self.device)
        advantage = returns - state_values.detach()
        advantage = (advantage - advantage.mean()) / advantage.std()
        return advantage

    def train(self, memory, prev_policy, iters):
        """
        Input: memory object, previous policy, current iteration
        Output: updated previous policy (ie: current policy)
        """
        mean_entropies = []
        for n in range (10):
            eps_frames, eps_mes,actions,actions_log_probs,rewards,terminals,advantage= memory.reservoir_sample(256)

            returns = self.discount_rewards(rewards, self.gamma,terminals)
            returns = torch.tensor(returns).to(self.device)
            actions_log_probs = torch.FloatTensor(actions_log_probs).to(self.device)
            advantage = torch.FloatTensor(advantage).to(self.device)

            #train PPO
            for i in range(self.n_epochs):
                current_action_log_probs, state_values, entropies = self.get_training_params(eps_frames, eps_mes, actions)
                policy_ratio = torch.exp(current_action_log_probs - actions_log_probs.detach())

                adv_l_update1 = policy_ratio*advantage
                adv_l_update2 = (torch.clamp(policy_ratio, 1-self.clip_val, 1+self.clip_val) * advantage).float()
                adv_l = torch.min(adv_l_update1, adv_l_update2)
                loss_v = self.mse(state_values.float(), returns.float())

                loss = \
                    - adv_l \
                    + (0.5 * loss_v) \
                    - (0.01 * entropies)

                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()

                mean_entropies.append(entropies.detach().numpy().mean())

            print("    on epoch " + str(n))

        if iters % 50 == 0:
            torch.save(self.state_dict(), "vanilla_policy_state_dictionary.pt")
        prev_policy.load_state_dict(self.state_dict())
        return prev_policy, np.array(mean_entropies).mean()
