source /usr/local/Ascend/ascend-toolkit/set_env.sh
# 该变量只用于规避megatron对其校验，对npu无效
export CUDA_DEVICE_MAX_CONNECTIONS=1
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
NPUS_PER_NODE=1
WORLD_SIZE=$(($NPUS_PER_NODE * $NNODES))

TP=1
PP=1
CP=1
MBS=1
GBS=$(($WORLD_SIZE*$MBS/$CP/$TP))

MM_MODEL="examples/hunyuanvideo/i2v/inference_model.json"
LOAD_PATH="./ckpt/hunyuanvideo"

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"
MM_ARGS="
 --mm-model $MM_MODEL
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
    --train-iters 5010 \
    --weight-decay 0 \
    --clip-grad 0.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --no-gradient-accumulation-fusion \
    --no-load-optim \
    --no-load-rng \
    --no-save-optim \
    --no-save-rng \
    --bf16 \
    --normalization RMSNorm \
    --use-fused-rmsnorm \
    --load $LOAD_PATH \
"

torchrun $DISTRIBUTED_ARGS inference_sora.py $MM_ARGS $GPT_ARGS