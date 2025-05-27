#!/bin/bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# 该变量只用于规避megatron对其校验，对npu无效
export CUDA_DEVICE_MAX_CONNECTIONS=1
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export TASK_QUEUE_ENABLE=1
export COMBINED_ENABLE=1
export CPU_AFFINITY_CONF=1
export HCCL_CONNECT_TIMEOUT=1200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=29505
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

TP=1
PP=1
CP=1
MBS=1
GBS=$(($WORLD_SIZE*$MBS/$CP/$TP))

BASEPATH=$(cd `dirname $0`; cd ../../../; pwd)

MM_DATA="$BASEPATH/tests/st/run_configs/pretrain_wan2.1_t2v/data.json"
MM_MODEL="$BASEPATH/tests/st/run_configs/pretrain_wan2.1_t2v/model.json"
MM_TOOL="$BASEPATH/mindspeed_mm/tools/tools.json"
LOAD_PATH="/home/ci_resource/models/wan2.1/wan2.1-t2v-14B-transformer"
layerzero_config="$BASEPATH/examples/wan2.1/zero_config.yaml"

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
    --num-layers 1 \
    --hidden-size 3072 \
    --num-attention-heads 48 \
    --seq-length 24 \
    --max-position-embeddings 24 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --tokenizer-type NullTokenizer \
    --vocab-size 0 \
    --position-embedding-type rope \
    --rotary-base 500000 \
    --swiglu \
    --no-masked-softmax-fusion \
    --lr 5e-6 \
    --min-lr 5e-6 \
    
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --adam-eps 1e-8 \
    --lr-decay-style constant \
    --weight-decay 1e-2 \
    --lr-warmup-init 0 \
    --lr-warmup-iters 0 \
    --clip-grad 1.0 \
    --train-iters 3 \
    --no-gradient-accumulation-fusion \
    --load $LOAD_PATH \
    --no-load-optim \
    --no-load-rng \
    --no-save-optim \
    --no-save-rng \
    --bf16 \
    --recompute-granularity full \
    --recompute-method block \
    --recompute-num-layers 40 \
    --use-distributed-optimizer \
    --overlap-grad-reduce \
    --overlap-param-gather \
    --normalization RMSNorm \
    --use-fused-rmsnorm \
    --layerzero \
    --layerzero-config ${layerzero_config} \
    --num-workers 4 \
"

MM_ARGS="
    --mm-data $MM_DATA \
    --mm-model $MM_MODEL \
    --mm-tool $MM_TOOL
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 10000 \
    --eval-interval 10000 \
    --eval-iters 10 \
"

torchrun $DISTRIBUTED_ARGS $BASEPATH/pretrain_sora.py \
    $GPT_ARGS \
    $MM_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl
