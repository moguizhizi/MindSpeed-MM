
# 网络名称,同目录名称,需要模型审视修改
Network="sdxl"

scripts_path="./sdxl"

# 预训练模型
model_name="stabilityai/stable-diffusion-xl-base-1.0"
vae_name="madebyollin/sdxl-vae-fp16-fix"
dataset_name="laion5b"
batch_size=4
num_processors=8
max_train_steps=2000
mixed_precision="fp16"
resolution=1024
config_file="${scripts_path}/pretrain_${mixed_precision}_accelerate_config.yaml"

for para in $*; do
  if [[ $para == --model_name* ]]; then
    model_name=$(echo ${para#*=})
  elif [[ $para == --vae_name* ]]; then
    vae_name=$(echo ${para#*=})
  elif [[ $para == --batch_size* ]]; then
    batch_size=$(echo ${para#*=})
  elif [[ $para == --max_train_steps* ]]; then
    max_train_steps=$(echo ${para#*=})
  elif [[ $para == --mixed_precision* ]]; then
    mixed_precision=$(echo ${para#*=})
  elif [[ $para == --resolution* ]]; then
    resolution=$(echo ${para#*=})
  elif [[ $para == --dataset_name* ]]; then
    dataset_name=$(echo ${para#*=})
  elif [[ $para == --config_file* ]]; then
    config_file=$(echo ${para#*=})
  fi
done

export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_GLOBAL_EVENT_ENABLE=0
export TASK_QUEUE_ENABLE=2
export COMBINED_ENABLE=1
export HCCL_WHITELIST_DISABLE=1
export HCCL_CONNECT_TIMEOUT=1200
export ACLNN_CACHE_LIMIT=100000
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export CPU_AFFINITY_CONF=1

# cd到与test文件夹同层级目录下执行脚本，提高兼容性；test_path_dir为包含test文件夹的路径
cur_path=$(pwd)
cur_path_last_dirname=${cur_path##*/}
if [ x"${cur_path_last_dirname}" == x"test" ]; then
  test_path_dir=${cur_path}
  cd ..
  cur_path=$(pwd)
else
  test_path_dir=${cur_path}/test
fi

echo ${test_path_dir}

#创建DeviceID输出目录，不需要修改
output_path=${cur_path}/logs

mkdir -p ${output_path}

#训练开始时间，不需要修改
start_time=$(date +%s)
echo "start_time: ${start_time}"

accelerate launch --config_file ${config_file} \
  ./examples/text_to_image/train_text_to_image_sdxl.py \
  --pretrained_model_name_or_path=$model_name \
  --pretrained_vae_model_name_or_path=$vae_name \
  --dataset_name=$dataset_name --caption_column="text" \
  --train_batch_size=$batch_size \
  --resolution=$resolution --random_flip \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --max_train_steps=$max_train_steps \
  --learning_rate=1e-05 --lr_scheduler="constant_with_warmup" --lr_warmup_steps=0 \
  --max_grad_norm=1 \
  --dataloader_num_workers=8 \
  --enable_npu_flash_attention \
  --mixed_precision=$mixed_precision \
  --checkpointing_steps=500 \
  --output_dir=${output_path} \
  2>&1 | tee ${output_path}/train_${mixed_precision}.log
wait
chmod 440 ${output_path}/train_${mixed_precision}.log

#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(($end_time - $start_time))

#结果打印，不需要修改
echo "------------------ Final result ------------------"

#输出性能FPS，需要模型审视修改
FPS=$(grep "FPS: " ${output_path}/train_${mixed_precision}.log | awk '{print $NF}' | sed -n '100,199p' | awk '{a+=$1}END{print a/NR}')

#获取性能数据，不需要修改
# - 吞吐量
ActualFPS=$(awk 'BEGIN{printf "%.2f\n", '${FPS}'}')

# - 打印，不需要修改
echo "Final Performance images/sec : $ActualFPS"

# - loss值，不需要修改
ActualLoss=$(grep -o "step_loss=[0-9.]*" ${output_path}/train_${mixed_precision}.log | awk 'END {print $NF}')

# - 打印，不需要修改
echo "Final Train Loss : ${ActualLoss}"
echo "E2E Training Duration sec : $e2e_time"

#性能看护结果汇总
# - 训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=$(uname -m)
CaseName=${Network}_bs${BatchSize}_'8p'_'acc'

#单迭代训练时长
TrainingTime=$(awk 'BEGIN{printf "%.2f\n", '${batch_size}'*'${num_processors}'/'${FPS}'}')

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" >${output_path}/${CaseName}.log
echo "BatchSize = ${BatchSize}" >>${output_path}/${CaseName}.log
echo "DeviceType = ${DeviceType}" >>${output_path}/${CaseName}.log
echo "CaseName = ${CaseName}" >>${output_path}/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >>${output_path}/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >>${output_path}/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >>${output_path}/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >>${output_path}/${CaseName}.log
