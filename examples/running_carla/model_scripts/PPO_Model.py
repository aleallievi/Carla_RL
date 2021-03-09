import numpy as np
import os
import torch
import torch.nn as nn
import gym

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
from ray.rllib.models.utils import get_activation_fn, get_filter_config
from ray.rllib.models.modelv2 import ModelV2, restore_original_dimensions
from ray.rllib.utils.torch_ops import one_hot 

class ComplexInputNetwork(TorchModelV2, nn.Module):
    """TorchModelV2 concat'ing CNN outputs to flat input(s), followed by FC(s).

    Note: This model should be used for complex (Dict or Tuple) observation
    spaces that have one or more image components.

    The data flow is as follows:

    `obs` (e.g. Tuple[img0, Box0..., ONE-HOT]) -> `CNN + FCNET + ONE-HOT`
    `CNN + CNN + ONE-HOT` -> concat all flat outputs -> `out`
    `out` -> FC-stack -> `out2`
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):

        #TODO: Maybe look at this?
        #https://github.com/ray-project/ray/issues/7583

    
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,model_config, name)
        nn.Module.__init__(self)

        # TODO: (sven) Support Dicts as well.
        self.original_space = obs_space.original_space if \
            hasattr(obs_space, "original_space") else obs_space
        # assert isinstance(self.original_space, (Tuple)), \
        #     "`obs_space.original_space` must be Tuple!"

        # super().__init__(self.original_space, action_space, num_outputs,model_config, name)
        # Build NN
        self.torch_sub_model = TorchFC(obs_space, action_space, num_outputs,model_config, name)

        config = {
                    "conv_filters": model_config.get("conv_filters", get_filter_config((84, 84, 3))),
                    "conv_activation": model_config.get("conv_activation"),
                    "post_fcnet_hiddens": [],
                }
        component_ = gym.spaces.Box(low=0, high=255,shape=(84, 84, 3)) #TODO THIS IS A VERY HADRCODED WAY OF DOING THINGS!!
        #cnn = ModelCatalog.get_model_v2(component,action_space,num_outputs=None,model_config=config,framework="torch",name="cnn_{}".format(i))
        self.cnn = ModelCatalog.get_model_v2(component_,action_space,num_outputs=None,model_config=config,framework="torch",name="cnn")


        concat_size = 0
        #for img
        concat_size += self.cnn.num_outputs
        #for one hot 
        concat_size += 6
        #for linea network
        concat_size+=5

        # Optional post-concat FC-stack.
        post_fc_stack_config = {
            "fcnet_hiddens": model_config.get("post_fcnet_hiddens", []),
            "fcnet_activation": model_config.get("post_fcnet_activation",
                                                 "relu")
        }
        self.post_fc_stack = ModelCatalog.get_model_v2(
            gym.spaces.Box(float("-inf"),
                float("inf"),
                shape=(concat_size, ),
                dtype=np.float32),
            self.action_space,
            None,
            post_fc_stack_config,
            framework="torch",
            name="post_fc_stack")

    def forward(self, input_dict, state, seq_lens):
        outs = []
        cnn_input = input_dict["obs"]["img"].float()
        one_hot_input = input_dict["obs"]["command"]
        linear_input = torch.cat((input_dict["obs"]["d2target"],input_dict["obs"]["pitch"],input_dict["obs"]["roll"],input_dict["obs"]["velocity_mag"],input_dict["obs"]["yaw"]), 1)
        linear_input = torch.transpose(linear_input,0,1)
        linear_input = linear_input.float()

        #run all inputs through their respective model (CNN + ONEHOT + NN)
        cnn_out, _ = self.cnn(cnn_input)

        outs.append(cnn_out)
        outs.append(one_hot(one_hot_input, gym.spaces.MultiDiscrete([2,2,2,2,2,2])))

        fc_out, _ = self.torch_sub_model(linear_input)
        outs.append(fc_out)

        # Concat all outputs and the non-image inputs.
        out = torch.cat(outs, axis=1)
        # Push through (optional) FC-stack (this may be an empty stack).
        out, _ = self.post_fc_stack(out, [], None)
        return out, []

    def value_function(self):
        return 0
        # return torch.reshape(self.torch_sub_model.value_function(), [-1])



class Vanilla_PPO(TorchModelV2, nn.Module):
    #TODO: For CNN + NN check out ComplexInputNetwork: https://docs.ray.io/en/master/rllib-models.html

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,model_config, name)
        nn.Module.__init__(self)

        self.torch_sub_model = TorchFC(obs_space, action_space, num_outputs,model_config, name)

    #same forward functiom
    def forward(self, input_dict, state, seq_lens):
        # print (input_dict["obs"])
        #format obs
        input_dict = torch.cat((input_dict["obs"]["command"], input_dict["obs"]["d2target"],input_dict["obs"]["pitch"],input_dict["obs"]["roll"],input_dict["obs"]["velocity_mag"],input_dict["obs"]["yaw"]), 1)
       
        input_dict = torch.transpose(input_dict,0,1)
        input_dict = input_dict.float()
        # print ("==================================")
        # print (input_dict.shape)
        # print ("==================================")
        fc_out, _ = self.torch_sub_model(input_dict, state, seq_lens)
        return fc_out, []
    
    #same value function
    def value_function(self):
        return torch.reshape(self.torch_sub_model.value_function(), [-1])
