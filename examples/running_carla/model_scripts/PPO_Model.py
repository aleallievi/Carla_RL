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
from ray.rllib.models.torch.misc import normc_initializer as \
    torch_normc_initializer, SlimFC
from ray.rllib.utils.annotations import override


class ComplexInputNetwork(TorchModelV2, nn.Module):
    """TorchModelV2 concat'ing CNN outputs to flat input(s), followed by FC(s).
    Note: This model should be used for complex (Dict or Tuple) observation
    spaces that have one or more image components.
    The data flow is as follows:
    `obs` (e.g. Tuple[img0, img1, discrete0]) -> `CNN0 + CNN1 + ONE-HOT`
    `CNN0 + CNN1 + ONE-HOT` -> concat all flat outputs -> `out`
    `out` -> (optional) FC-stack -> `out2`
    `out2` -> action (logits) and vaulue heads.
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):

        # # TODO: (sven) Support Dicts as well.
        self.original_space = obs_space.original_space if \
            hasattr(obs_space, "original_space") else obs_space
        
        # print (self.original_space)
        tuple_values = []
        #convert obs_space from gym.spaces.Dict to gym.spaces.Tuple
        #Note: this is my patch because RLLIB seems to lack support of this
        for k, space in  self.original_space.spaces.items():
            tuple_values.append(space)
        self.original_space = gym.spaces.Tuple(tuple_values)
        assert isinstance(self.original_space, (gym.spaces.Tuple)), \
            "`obs_space.original_space` must be Tuple!"

        nn.Module.__init__(self)
        TorchModelV2.__init__(self, obs_space, action_space,num_outputs, model_config, name)

        # Build the CNN(s) given obs_space's image components.
        self.cnns = {}
        self.one_hot = {}
        self.flatten = {}
        self.meas = {}
        concat_size = 0
        for i, component in enumerate(self.original_space):
            # Image space.
            if len(component.shape) == 3:
                config = {
                    "conv_filters": model_config.get(
                        "conv_filters", get_filter_config(component.shape)),
                    "conv_activation": model_config.get("conv_activation"),
                    "post_fcnet_hiddens": [],
                }
                # if self.cnn_type == "atari":
                cnn = ModelCatalog.get_model_v2(
                    component,
                    action_space,
                    num_outputs=None,
                    model_config=config,
                    framework="torch",
                    name="cnn_{}".format(i))
              
                concat_size += cnn.num_outputs
                self.cnns[i] = cnn
                self.add_module("cnn_{}".format(i), cnn)
            # Discrete inputs -> One-hot encode.
            elif isinstance(component, gym.spaces.MultiDiscrete):
                self.one_hot[i] = True
                concat_size += component.shape[0]
            else:
                self.meas[i] = True
                concat_size+=1
    
        # Optional post-concat FC-stack.
        post_fc_stack_config = {
            "fcnet_hiddens": model_config.get("post_fcnet_hiddens", []),
            "fcnet_activation": model_config.get("post_fcnet_activation",
                                                 "relu")
        }
        self.post_fc_stack = ModelCatalog.get_model_v2(
            gym.spaces.Box(float("-inf"),float("inf"),shape=(concat_size, ),dtype=np.float32),
            self.action_space,
            None,
            post_fc_stack_config,
            framework="torch",
            name="post_fc_stack")

        # Actions and value heads.
        self.logits_layer = None
        self.value_layer = None
        self._value_out = None

              
        if num_outputs:
            # Action-distribution head.
            self.logits_layer = SlimFC(
                in_size=self.post_fc_stack.num_outputs+1,#TODO: very suspicious bug. I am not sure why num_outputs is initially off by 1  
                out_size=num_outputs,
                activation_fn=None,
            )
            # Create the value branch model.
            self.value_layer = SlimFC(
                in_size=self.post_fc_stack.num_outputs+1,#TODO: very suspicious bug. I am not sure why num_outputs is initially off by 1  
                out_size=1,
                activation_fn=None,
                initializer=torch_normc_initializer(0.01))
        else:
            self.num_outputs = concat_size

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        # Push image observations through our CNNs.
        outs = []
        collected_nn_input = []
        for i, component in enumerate(input_dict["obs"]):
            if i in self.cnns:
                cnn_out, _ = self.cnns[i]({"obs": input_dict["obs"].get(component)})
                outs.append(cnn_out)
            elif i in self.one_hot:
                #TODO: need to double check this but I am pretty sure we do not need to reencode our one-hot vector
                #  encoded = one_hot(input_dict["obs"].get(component), self.original_space.spaces[i])
                 outs.append(input_dict["obs"].get(component))
            else:
                collected_nn_input.append(input_dict["obs"].get(component))
        
        # Concat all outputs and the non-image inputs.
        out = torch.cat(outs, dim=1)
        # Push through (optional) FC-stack (this may be an empty stack).
        out, _ = self.post_fc_stack({"obs": out}, [], None)
        
        # No logits/value branches.
        if self.logits_layer is None:
            return out, []

        # Logits- and value branches. 
        logits = self.logits_layer(out) 
        values = self.value_layer(out)
        self._value_out = torch.reshape(values, [-1])
        return logits, []

    @override(ModelV2)
    def value_function(self):
        return self._value_out


class BADComplexInputNetwork(TorchModelV2, nn.Module):
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

        #Based of this:
        #https://github.com/ray-project/ray/blob/ad8e35b9195a84722ea0accf2a63785bd39911c4/rllib/models/torch/complex_input_net.py#L19
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
        print (input_dict)
        print ("==============================")
        print ("\n")

        outs = []
        cnn_input = input_dict["obs"]["img"].float()
        one_hot_input = input_dict["obs"]["command"]
        linear_input = torch.cat((input_dict["obs"]["d2target"],input_dict["obs"]["pitch"],input_dict["obs"]["roll"],input_dict["obs"]["velocity_mag"],input_dict["obs"]["yaw"]), 1)
        linear_input = torch.transpose(linear_input,0,1)
        linear_input = linear_input.float()

        #run all inputs through their respective model (CNN + ONEHOT + NN)
        # if not isinstance(cnn_input, dict):
        #     cnn_input = {"obs":cnn_input}

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
