import pathlib

# change these.
CLUSTER_QUICKSTART_DIR = '/scratch/cluster/stephane/Carla_RL'
CARLA_DIR = '/scratch/cluster/stephane/Carla_0.9.10'

HEADER = f"""#!/bin/bash

trap "exit" INT TERM ERR
trap "kill 0" EXIT

export WORLD_PORT=$(python3 {CLUSTER_QUICKSTART_DIR}/scripts/find_open_ports.py)
export TM_PORT=$(python3 {CLUSTER_QUICKSTART_DIR}/scripts/find_open_ports.py)

export CARLA_EGG=$(ls {CARLA_DIR}/PythonAPI/carla/dist/*py3*)
export PYTHONPATH=$PYTHONPATH:$CARLA_EGG

echo $WORLD_PORT
echo $TM_PORT

sh {CARLA_DIR}/CarlaUE4.sh -world-port=$WORLD_PORT -opengl -quality-level=Epic &

sleep 100
"""

BODY = """
cd {target_dir}
python3 train_model.py --client-timeout 999 --world-port $WORLD_PORT --tm-port $TM_PORT
"""

FOOTER = """
ps
kill 0
"""

PARAMS = {
        'model': ["ppo"],
        }

def get_job():
    body = BODY.format(target_dir=pathlib.Path(__file__).parent.parent)
    return HEADER + body + FOOTER
