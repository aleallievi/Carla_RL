#!/bin/bash

export REPO=Carla_RL
export SRC=/bosch/src/
export CARLA=$SRC/9_10_1
export CARLA_DIR=$HOME$CARLA
export CARLA_RL=$HOME/$SRC/$REPO

export AGENTS_DIR=$CARLA_RL/examples/running_carla/
export SCRIPTS_DIR=$CARLA_RL/scripts 

export WORLD_PORT=2000 # change per user
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
python3 ${AGENTS_DIR}/training_scripts/train_rllib_ppo.py --client-timeout 999 --world-port $WORLD_PORT --tm-port $TM_PORT --client-gpu $CLIENT_GPU
