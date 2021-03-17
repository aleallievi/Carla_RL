"""
lib
"""
import argparse
import gym
from gym.spaces import Discrete, Box, MultiBinary, Dict
import numpy as np
import os
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import scipy.signal
import wandb

from ray.tune.integration.wandb import WandbLogger
from ray.tune.integration.wandb import wandb_mixin
from ray.tune.integration.wandb import WandbLoggerCallback

import ray
from ray import tune
from ray.tune import grid_search
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
import ray.rllib.agents.ppo as ppo
from ray.rllib.agents import impala
from ray.rllib.models.modelv2 import ModelV2, restore_original_dimensions
from ray.rllib.policy.torch_policy_template import build_torch_policy
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy

from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.policy.policy_template import build_policy_class
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation.postprocessing import compute_gae_for_sample_batch, \
    Postprocessing
from ray.rllib.utils.torch_ops import apply_grad_clipping, \
    convert_to_torch_tensor, explained_variance, sequence_mask
from ray.rllib.agents.ppo.ppo_torch_policy import vf_preds_fetches, \
    ValueNetworkMixin, KLCoeffMixin, EntropyCoeffSchedule, LearningRateSchedule

"""
loss
"""
def custom_ppo_loss(policy, model, dist_class, train_batch):
    """
    Args:
        policy (Policy): The Policy to calculate the loss for.
        model (ModelV2): The Model to calculate the loss for.
        dist_class (Type[ActionDistribution]: The action distr. class.
        train_batch (SampleBatch): The training data.
    Returns:
        Union[TensorType, List[TensorType]]: A single loss tensor or a list
            of loss tensors.
    """

    """
    references:
    https://docs.ray.io/en/master/rllib-concepts.html
    https://github.com/ray-project/ray/blob/master/rllib/agents/ppo/ppo_torch_policy.py
    https://github.com/ray-project/ray/blob/master/rllib/examples/custom_torch_policy.py
    https://github.com/ray-project/ray/issues/8507
    """

    # We imperatively execute the forward pass by calling model() on the observations followed by dist_class() on the output logits. 
    logits, state = model.from_batch(train_batch, is_training=True)
    curr_action_dist = dist_class(logits, model)
    prev_action_dist = dist_class(train_batch[SampleBatch.ACTION_DIST_INPUTS],model)
    mask = None
    reduce_mean_valid = torch.mean

    #policy_ratio = torch.exp(current_action_log_probs - actions_log_probs.detach())
    policy_ratio = torch.exp(curr_action_dist.logp(train_batch[SampleBatch.ACTIONS]) - train_batch[SampleBatch.ACTION_LOGP])

    # used as a regulizer
    action_kl = prev_action_dist.kl(curr_action_dist)
    mean_kl = reduce_mean_valid(action_kl)
    
    # for possible regularization parameter
    curr_entropy = curr_action_dist.entropy()
    mean_entropy = reduce_mean_valid(curr_entropy)

    update = torch.min(
        train_batch[Postprocessing.ADVANTAGES] * policy_ratio,
        train_batch[Postprocessing.ADVANTAGES] * torch.clamp(policy_ratio, 1 - policy.config["clip_param"],1 + policy.config["clip_param"]))
    mean_policy_loss = reduce_mean_valid(-update)

    # GAE
    prev_values = train_batch[SampleBatch.VF_PREDS]
    current_values = model.value_function()

    #l1 = (V - R)^2
    #l2 = ((Prev_V + clip(V - Prev_V)) - R)^2
    #vf_loss = max(L1, L2)
    vf_loss1 = torch.pow(current_values - train_batch[Postprocessing.VALUE_TARGETS], 2.0)
    vf_clipped = prev_values + torch.clamp(current_values - prev_values, -policy.config["vf_clip_param"],policy.config["vf_clip_param"])
    vf_loss2 = torch.pow(vf_clipped - train_batch[Postprocessing.VALUE_TARGETS], 2.0)
    vf_loss = torch.max(vf_loss1, vf_loss2)
    mean_vf_loss = reduce_mean_valid(vf_loss)
    total_loss = reduce_mean_valid(
            -update + policy.kl_coeff * action_kl +
            policy.config["vf_loss_coeff"] * vf_loss -
            policy.entropy_coeff * curr_entropy)


    total_loss_wo_kl = reduce_mean_valid(
            -update + policy.kl_coeff * action_kl +
            policy.config["vf_loss_coeff"] * vf_loss -
            policy.entropy_coeff * curr_entropy)

    # mean_vf_loss = torch.Tensor([0.0]).item()
    # total_loss = reduce_mean_valid(-update + policy.kl_coeff * action_kl - policy.entropy_coeff * curr_entropy)

   
    # print (total_loss)
    # print ("----------------------")
    # print (total_loss_wo_kl)

    # Store stats in policy for stats_fn.
    policy._total_loss = total_loss 
    policy._mean_policy_loss = mean_policy_loss
    policy._mean_vf_loss = mean_vf_loss
    policy._vf_explained_var = explained_variance(
        train_batch[Postprocessing.VALUE_TARGETS],
        policy.model.value_function())
    policy._mean_entropy = mean_entropy 
    policy._mean_kl = mean_kl 
    
    return total_loss

def setup_mixins(policy, obs_space,action_space,config):
    ValueNetworkMixin.__init__(policy, obs_space, action_space, config)
    KLCoeffMixin.__init__(policy, config)
    EntropyCoeffSchedule.__init__(policy, config["entropy_coeff"],
                                  config["entropy_coeff_schedule"])
    LearningRateSchedule.__init__(policy, config["lr"], config["lr_schedule"])


"""
discounted cumulative sum of rewards
"""

def discount_cumsum(delta_t,gamma):
    discounted = 0
    for i in range (0,len(delta_t)):
        delta_t[len(delta_t)-1-i] = delta_t[len(delta_t)-1-i] + (gamma*discounted)
        discounted = delta_t[len(delta_t)-1-i]
    delta_t = np.array(delta_t,dtype=np.float64)
    return delta_t

def original_discount_cumsum(x,gamma):
    """Calculates the discounted cumulative sum over a reward sequence `x`.
    y[t] - discount*y[t+1] = x[t]
    reversed(y)[t] - discount*reversed(y)[t-1] = reversed(x)[t]
    Args:
        gamma (float): The discount factor gamma.
    Returns:
        float: The discounted cumulative sum over the reward sequence `x`.
    """
    return scipy.signal.lfilter([1], [1, float(-gamma)], x[::-1], axis=0)[::-1]

"""
advantages
"""
def compute_advantages(rollout,last_r,gamma = 0.9,lambda_= 1.0,use_gae=True,use_critic = True):
    """
    Given a rollout, compute its value targets and the advantages.
    Args:
        rollout (SampleBatch): SampleBatch of a single trajectory.
        last_r (float): Value estimation for last observation.
        gamma (float): Discount factor.
        lambda_ (float): Parameter for GAE.
        use_gae (bool): Using Generalized Advantage Estimation.
        use_critic (bool): Whether to use critic (value estimates). Setting
            this to False will use 0 as baseline.
    Returns:
        SampleBatch (SampleBatch): Object with experience from rollout and
            processed rewards.
    """
    #Finish here:
    #https://github.com/ray-project/ray/blob/488f63efe395a5dc146ac7cd14cabd1acce1e5f6/rllib/evaluation/postprocessing.py#L12
    #-------------------RLLIB IMPLEMENTATION JUST FOR TESTING-----------------
    vpred_t = np.concatenate([rollout[SampleBatch.VF_PREDS],np.array([last_r])])
    

    """
    #todo: there is a slight sigfig difference between this and the RLLIB delta calculation
    delta_t = [0 for i in range(len(rollout[SampleBatch.REWARDS]))]
    for i in reversed(range(len(rollout[SampleBatch.REWARDS]))):
      delta_t[i] = rollout[SampleBatch.REWARDS][i] + gamma * vpred_t[i+1] - vpred_t[i]
    """

    #only to achieve identical results
    # delta_t = np.round(delta_t,8)
    delta_t = (rollout[SampleBatch.REWARDS] + gamma * vpred_t[1:] - vpred_t[:-1])

    # rollout[Postprocessing.ADVANTAGES] = delta_t
    rollout[Postprocessing.ADVANTAGES] = original_discount_cumsum(delta_t,lambda_*gamma)
    # print (delta_t)
    # print ("\n")
    
    rollout[Postprocessing.VALUE_TARGETS] = (rollout[Postprocessing.ADVANTAGES] + rollout[SampleBatch.VF_PREDS]).astype(np.float32)
    #-------------------RLLIB IMPLEMENTATION JUST FOR TESTING-----------------
    rollout[Postprocessing.ADVANTAGES] = rollout[Postprocessing.ADVANTAGES].astype(np.float32)
    return rollout

def compute_gae_for_sample_batch(policy,sample_batch,other_agent_batches=None,episode=None):
    completed = sample_batch[SampleBatch.DONES][-1]
    if completed:
        last_r = 0.0
    else:
        input_dict = policy.model.get_input_dict(sample_batch, index="last")
        last_r = policy._value(**input_dict)
    return compute_advantages(sample_batch, last_r, policy.config["gamma"],policy.config["lambda"])

def policy_gradient_loss(policy, model, dist_class, train_batch):
    logits, _ = model({SampleBatch.CUR_OBS: train_batch[SampleBatch.CUR_OBS]})
    action_dist = dist_class(logits, model)
    log_probs = action_dist.logp(train_batch[SampleBatch.ACTIONS])
    return -train_batch[SampleBatch.REWARDS].dot(log_probs)


"""
logging
"""
def on_episode_step(info):
    pass
    # episode = info["episode"]
    # f = open("Ported_RLLIB_PPO_actions_seed=1_pl_pa.txt", "a")
    # f.write(np.array2string(episode.last_action_for("agent0"), separator=',') + "\n")
    # f.close()

def on_episode_end(info):
    pass
    # episode = info["episode"]
    # f = open("Ported_RLLIB_PPO_actions_seed=1_pl_pa.txt", "a")
    # f.write("\n-------------------END OF EPISODE-------------------\n")
    # f.close()

def on_train_result(info):
    print ("ended training...")
    pass
