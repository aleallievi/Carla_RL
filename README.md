
### Starting Carla on Bosch Lambda

## SSH

`ssh name@boschaustin.csres.utexas.edu`

## Storage

All files (inlcuding this directory) are kept in `/scratch/cluster/stephane`. 

## Running CARLA 

The path to Carla is set in `examples/running_carla/local/run_aggent_ppo.sh`. This script to run (ie: train_rllib_ppo.py) can be set here as well. 

To start the server and run the client, run the following:

```bash
./examples/running_carla/local/run_aggent_ppo.sh
```

then you can look at the running jobs on the specified WANDB dashboard, or using Teamviewer`.  

## User Specific Configs

Script to run, Carla Directory, World Port, and TM Port are all set in `examples/running_carla/local/run_aggent_ppo.sh`
The Carla directory is also set in `/scripts/launch_carla.py`
WANDB API Key is set in `examples/running_carla/training_scripts/...

### Starting Carla on hypnotoad

## SSH

`ssh -X -v stephane@hypnotoad.cs.utexas.edu`

## Storage

All files (inlcuding this directory) are kept in `/scratch/cluster/stephane`. 

## Running CARLA 

The path to Carla is set in `examples/running_carla/cluster/params.py`. This script to run (ie: carla_ppo.py) can be set here as well. 

```bash
python3 generate_slurm.py examples/running_carla/cluster/params.py

cd examples/running_carla/cluster/slurm_scripts

sbatch model=ppo.submit
```

then you can look at the running jobs using `squeue -u stephane`.  

When your job is running/done you'll be able to check out the `examples/running_carla/cluster/logs` and see the individual STDERR, STDOUT.

