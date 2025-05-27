Network="StableDiffusionXLControlNetDeepspeed"

model_name="stabilityai/stable-diffusion-xl-base-1.0"
vae_name="madebyollin/sdxl-vae-fp16-fix"
dataset_name="fusing/fill50k"
batch_size=5
num_processors=8
max_train_steps=2000
checkpointing_steps=2000
validation_steps=2000
mixed_precision="fp16"
resolution=1024

for para in $*; do
  if [[ $para == --batch_size* ]]; then
    batch_size=$(echo ${para#*=})
  elif [[ $para == --max_train_steps* ]]; then
    max_train_steps=$(echo ${para#*=})
  elif [[ $para == --mixed_precision* ]]; then
    mixed_precision=$(echo ${para#*=})
  elif [[ $para == --resolution* ]]; then
    resolution=$(echo ${para#*=})
  elif [[ $para == --vae_name* ]]; then
    vae_name=$(echo ${para#*=})
  elif [[ $para == --checkpointing_steps* ]]; then
    checkpointing_steps=$(echo ${para#*=})
  elif [[ $para == --validation_steps* ]]; then
    validation_steps=$(echo ${para#*=})
  fi
done

export TASK_QUEUE_ENABLE=2
export HCCL_CONNECT_TIMEOUT=1200
export ACLNN_CACHE_LIMIT=100000
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export CPU_AFFINITY_CONF=1

cur_path=$(pwd)
cur_path_last_dirname=${cur_path##*/}
if [ x"${cur_path_last_dirname}" == x"test" ]; then
  test_path_dir=${cur_path}
  cd ..
  cur_path=$(pwd)
else
  test_path_dir=${cur_path}
fi

output_path=${cur_path}/logs


mkdir -p ${output_path}


start_time=$(date +%s)
echo "start_time: ${start_time}"



accelerate launch --config_file ./sdxl/accelerate_deepspeed_config.yaml \
 ./examples/controlnet/train_controlnet_sdxl.py \
 --pretrained_model_name_or_path=$model_name \
 --dataset_name=$dataset_name \
 --mixed_precision=$mixed_precision \
 --resolution=$resolution \
 --learning_rate=1e-5 \
 --max_train_steps=$max_train_steps \
 --checkpointing_steps=$checkpointing_steps \
 --validation_image "conditioning_image_1.png" "conditioning_image_2.png" \
 --validation_prompt "red circle with blue background" "cyan circle with brown floral background" \
 --validation_steps=$validation_steps \
 --train_batch_size=$batch_size \
 --gradient_accumulation_steps=4 \
 --dataloader_num_workers=8 \
 --pretrained_vae_model_name_or_path=$vae_name \
 --seed=1234 \
 --enable_npu_flash_attention \
 --output_dir=${output_path} \
 2>&1 | tee ${output_path}/train_${mixed_precision}_sdxl_controlnet_deepspeed.log
wait
chmod 440 ${output_path}/train_${mixed_precision}_sdxl_controlnet_deepspeed.log

#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(($end_time - $start_time))

#结果打印，不需要修改
echo "------------------ Final result ------------------"

#输出性能FPS，需要模型审视修改
FPS=$(grep "FPS: " ${output_path}/train_${mixed_precision}_sdxl_controlnet_deepspeed.log | awk '{print $NF}' | sed -n '100,199p' | awk '{a+=$1}END{print a/NR}')

#获取性能数据，不需要修改
#吞吐量
ActualFPS=$(awk 'BEGIN{printf "%.2f\n", '${FPS}'}')

#打印，不需要修改
echo "Final Performance images/sec : $ActualFPS"

#loss值，不需要修改
ActualLoss=$(grep -o "step_loss=[0-9.]*" ${output_path}/train_${mixed_precision}_sdxl_controlnet_deepspeed.log | awk 'END {print $NF}')

#打印，不需要修改
echo "Final Train Loss : ${ActualLoss}"
echo "E2E Training Duration sec : $e2e_time"

#性能看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=$(uname -m)
CaseName=${Network}_bs${BatchSize}_'8p'_'acc'

#单迭代训练时长
TrainingTime=$(awk 'BEGIN{printf "%.2f\n", '${batch_size}'* '${num_processors}'/'${FPS}'}')

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" >${output_path}/${CaseName}.log
echo "BatchSize = ${BatchSize}" >>${output_path}/${CaseName}.log
echo "DeviceType = ${DeviceType}" >>${output_path}/${CaseName}.log
echo "CaseName = ${CaseName}" >>${output_path}/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >>${output_path}/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >>${output_path}/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >>${output_path}/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >>${output_path}/${CaseName}.log
