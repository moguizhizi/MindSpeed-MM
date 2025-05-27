export CUDA_DEVICE_MAX_CONNECTIONS=1
export HYDRA_FULL_ERROR=1

source /home/c30061641/RL/cann/b070/ascend-toolkit/set_env.sh
source /home/c30061641/RL/cann/b070/nnal/atb/set_env.sh

export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=2


SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
export PYTHONPATH=$SCRIPT_DIR/../..:$PYTHONPATH
PROJECT_PATH=$SCRIPT_DIR/../..

python "$PROJECT_PATH"/cli/train_grpo.py --config-dir="$PROJECT_PATH"/configs --config-name=grpo_trainer_qwen25vl_3b_integrated 2>&1 | tee logs/train_grpo.log 