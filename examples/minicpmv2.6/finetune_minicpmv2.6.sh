#!/bin/bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export TASK_QUEUE_ENABLE=2
export COMBINED_ENABLE=1
export CPU_AFFINITY_CONF=1
export HCCL_CONNECT_TIMEOUT=1200
# 该变量只用于规避megatron对其校验，对npu无效
export CUDA_DEVICE_MAX_CONNECTIONS=1
export ACLNN_CACHE_LIMIT=100000

NPUS_PER_NODE=1
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

MM_DATA="./examples/minicpmv2.6/data.json"
MM_MODEL="./examples/minicpmv2.6/model.json"
MM_TOOL="./mindspeed_mm/tools/tools.json"
LOAD_PATH="pretrained/MiniCPM-V-2_6"
SAVE_PATH="save_dir"

MBS=1
GRAD_ACC_STEP=8
TP=1
PP=1
CP=1
DP=$(($WORLD_SIZE/$TP/$PP/$CP))
GBS=$(($MBS*$GRAD_ACC_STEP*$DP))

MM_ARGS="
    --mm-data ${MM_DATA} \
    --mm-model ${MM_MODEL} \
    --mm-tool ${MM_TOOL}
"

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
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --tokenizer-type NullTokenizer \
    --vocab-size 151666 \
    --position-embedding-type rope \
    --rotary-base 100000 \
    --swiglu \
    --no-masked-softmax-fusion \
    --lr 4e-5 \
    --min-lr 0.0 \
    --train-iters 5000 \
    --lr-decay-style cosine \
    --weight-decay 0.05 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --no-gradient-accumulation-fusion \
    --no-load-optim \
    --no-load-rng \
    --no-save-optim \
    --no-save-rng \
    --use-distributed-optimizer \
    --bf16 \
    --load $LOAD_PATH \
    --use-flash-attn \
    --use-fused-rotary-pos-emb \
    --variable-seq-lengths \
    --normalization RMSNorm \
    --use-fused-rmsnorm \
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 5000 \
    --eval-interval 5000 \
    --eval-iters 5000 \
    --save $SAVE_PATH \
"

logfile=$(date +%Y%m%d)_$(date +%H%M%S)
mkdir -p logs
torchrun $DISTRIBUTED_ARGS \
    pretrain_vlm.py \
    $GPT_ARGS \
    $MM_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    | tee logs/train_${logfile}.log 2>&1
chmod 440 logs/train_${logfile}.log
set +x