
import ray
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.torch_policy_template import build_torch_policy
import ray.rllib.agents.ppo as ppo
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.tune.integration.wandb import WandbLogger
import os

# sys.path.append("../Carla/")
from Carla_eval_tools.carla_env import CarlaEnv
# sys.path.append("../model_scripts/")
from model_scripts.PPO_Model import Vanilla_PPO,ComplexInputNetwork
from Raytune_Carla_eval_tools.carla_env import CarlaEnv
from model_scripts.Ported_RLLIB_PPO import custom_ppo_loss, compute_gae_for_sample_batch, setup_mixins, LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin,ValueNetworkMixin, on_episode_step, on_episode_end,on_train_result


def main (args):
    # Create client outside of Carla environment to avoid creating zombie clients
    
    as_test = False
    ray.init()
    stop_iters = 10000
    stop_timesteps = 10000000
    stop_reward = 100

    # Can also register the env creator function explicitly with:
    # register_env("corridor", lambda config: SimpleCorridor(config))
    ModelCatalog.register_custom_model("my_model", ComplexInputNetwork)

    config = ppo.DEFAULT_CONFIG.copy()

    config["seed"] = 1
    config["logger_config"] = {"wandb": {
              "project": "Carla_PPO_Single_Route",
              "api_key": "64820e399b4da86d50ba9f5931d435ceb2cc8846",
          }}
    config["num_gpus"] = int(os.environ.get("RLLIB_NUM_GPUS", "0"))
    config["framework"] = "torch"
    config["num_workers"] = 1
    config["model"] = {
            "custom_model": "my_model",
            # "vf_share_layers": True,
            # "fcnet_hiddens": [64, 32],
            # "fcnet_activation": "tanh"
        }
    config["callbacks"] = { "on_episode_step": on_episode_step,"on_episode_end": on_episode_end,"on_train_result": on_train_result}
    config["env"] = CarlaEnv
    config["env_config"] = {"args": args,}
    # config["env_config"]["experiment"]["type"] = EXPERIMENT_CLASS

    stop = {
        "training_iteration": stop_iters,
        "timesteps_total": stop_timesteps,
        "episode_reward_mean": stop_reward,
    }

    #https://github.com/ray-project/ray/issues/7855 solves correct loss not being used
    CustomPolicy = PPOTorchPolicy.with_updates(name="Ported_PPO",loss_fn=custom_ppo_loss,postprocess_fn=compute_gae_for_sample_batch,before_loss_init=setup_mixins,mixins=[LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin,ValueNetworkMixin])
    CustomTrainer = PPOTrainer.with_updates(name = "Custom_PPO_trainer",default_policy=CustomPolicy,get_policy_class=None)
    # results = tune.run(ppo.PPOTrainer, config=config, stop=stop,loggers=[WandbLogger])
    results = tune.run(CustomTrainer, config=config, stop=stop,checkpoint_freq=1,checkpoint_at_end=True)

    ray.shutdown()
    

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
