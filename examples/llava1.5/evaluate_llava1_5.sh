
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 通过此配置选择使用的NPU卡
# export ASCEND_RT_VISIBLE_DEVICES="0,1,2,3"
# 该变量只用于规避megatron对其校验，对npu无效
export CUDA_DEVICE_MAX_CONNECTIONS=1
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export TASK_QUEUE_ENABLE=1
export COMBINED_ENABLE=1
export CPU_AFFINITY_CONF=1
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_EXEC_TIMEOUT=7200

NPUS_PER_NODE=4
MASTER_ADDR=localhost
MASTER_PORT=29501
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

TP=1
PP=1
CP=1
MBS=1
GBS=$(($WORLD_SIZE*$MBS/$CP))

export TOKENIZERS_PARALLELISM=false

MM_MODEL=examples/llava1.5/evaluate_llava1_5.json

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --context-parallel-size ${CP} \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --seq-length 2048 \
    --tokenizer-type NullTokenizer \
    --vocab-size 32000 \
    --position-embedding-type rope \
    --no-masked-softmax-fusion \
    --fp16 \
    --distributed-timeout-minutes 1000 \
    --normalization RMSNorm
"
MM_ARGS="
    --mm-model ${MM_MODEL} \
"


IMG_ARGS="
    --img-h 336 \
    --img-w 336 \
    --patch-dim 14
"

torchrun $DISTRIBUTED_ARGS \
    evaluate_vlm.py \
    $GPT_ARGS \
    $MM_ARGS \
    $IMG_ARGS \
    --distributed-backend nccl