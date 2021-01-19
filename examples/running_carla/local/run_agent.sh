#!/bin/bash
# change these.
CARLA_DIR=/home/boschaustin/projects/CL_AD/Carla_RL/CARLA_9_10_1
AGENTS_DIR=/home/boschaustin/projects/CL_AD/Carla_RL/examples/running_carla
SCRIPTS_DIR=/home/boschaustin/projects/CL_AD/Carla_RL/scripts

export WORLD_PORT=9000
export TM_PORT=$((${WORLD_PORT}+50))
export SERVER_GPU=2
export CLIENT_GPU=3

export CARLA_EGG=$(ls ${CARLA_DIR}/PythonAPI/carla/dist/*py3*)

export PYTHONPATH=$PYTHONPATH:$AGENTS_DIR
export PYTHONPATH=$PYTHONPATH:$SCRIPTS_DIR
export PYTHONPATH=$PYTHONPATH:$CARLA_EGG
export PYTHONPATH=$PYTHONPATH:$CARLA_DIR/PythonAPI/carla


#Launch CARLA server
python3 ${SCRIPTS_DIR}/launch_carla.py --world-port $WORLD_PORT --gpu $SERVER_GPU

#Launch CARLA client
python3 ${AGENTS_DIR}/training_scripts/train_ppo_model.py --world-port $WORLD_PORT --tm-port $TM_PORT --client-gpu $CLIENT_GPU
