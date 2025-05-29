# HunyuanVideo使用指南
- [HunyuanVideo使用指南](#hunyuanvideo使用指南)
  - [环境安装](#环境安装)
    - [仓库拉取](#仓库拉取)
    - [环境搭建](#环境搭建)
    - [Decord搭建](#decord搭建)
  - [权重下载及转换](#权重下载及转换)
    - [TextEncoder下载](#textencoder下载)
    - [HunyuanVideoDiT与VAE下载](#hunyuanvideodit与vae下载)
    - [权重转换](#权重转换)
  - [预训练](#预训练)
    - [数据预处理](#数据预处理)
    - [特征提取](#特征提取)
      - [准备工作](#准备工作)
      - [参数配置](#参数配置)
      - [启动特征提取](#启动特征提取)
    - [训练](#训练)
      - [准备工作](#准备工作-1)
      - [参数配置](#参数配置-1)
      - [启动训练](#启动训练)
      - [权重后处理](#权重后处理)
  - [I2V lora微调](#i2v-lora微调)
    - [准备工作](#准备工作-2)
      - [权重转换](#权重转换-1)
      - [特征提取](#特征提取-1)
      - [配置参数](#配置参数)
    - [启动lora微调](#启动lora微调)
  - [推理](#推理)
    - [准备工作](#准备工作-3)
    - [参数配置](#参数配置-2)
    - [启动推理](#启动推理)
  - [环境变量声明](#环境变量声明)

## 环境安装

【模型开发时推荐使用配套的环境版本】

请参考[安装指南](https://gitee.com/ascend/MindSpeed-MM/blob/master/docs/features/install_guide.md)

### 仓库拉取

```shell
git clone https://gitee.com/ascend/MindSpeed-MM.git 
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout core_r0.8.0
cp -r megatron ../MindSpeed-MM/
cd ..
cd MindSpeed-MM
```

### 环境搭建

```bash
# python3.10
conda create -n test python=3.10
conda activate test

# 安装 torch 和 torch_npu，注意要选择对应python版本、x86或arm的torch、torch_npu及apex包
pip install torch-2.1.0-cp310-cp310m-manylinux2014_aarch64.whl 
pip install torch_npu-2.1.0*-cp310-cp310m-linux_aarch64.whl

# apex for Ascend 参考 https://gitee.com/ascend/apex
# 建议从原仓编译安装 

# 将shell脚本中的环境变量路径修改为真实路径，下面为参考路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh 

# 安装加速库
git clone https://gitee.com/ascend/MindSpeed.git
cd MindSpeed
# checkout commit from MindSpeed core_r0.8.0
git checkout 9bd51f777820aff70ab7507c8b4da7dde566b37b
pip install -r requirements.txt 
pip install -e .
cd ..

# 安装其余依赖库
pip install -e .
```

### Decord搭建

【X86版安装】

```bash
pip install decord==0.6.0
```

【ARM版安装】

`apt`方式安装请[参考链接](https://github.com/dmlc/decord)

`yum`方式安装请[参考脚本](https://github.com/dmlc/decord/blob/master/tools/build_manylinux2010.sh)

---

## 权重下载及转换

### TextEncoder下载
+ [llava-llama-3-8b](https://huggingface.co/xtuner/llava-llama-3-8b-v1_1-transformers)
+ [clip-vit-large](https://huggingface.co/openai/clip-vit-large-patch14)

### HunyuanVideoDiT与VAE下载
+ [tencent/HunyuanVideo](https://huggingface.co/tencent/HunyuanVideo)
+ [tencent/HunyuanVideo-I2V](https://huggingface.co/tencent/HunyuanVideo-I2V)
下载后的权重结构分别如下
```shell
HunyuanVideo
  ├──README.md
  ├──hunyuan-video-t2v-720p
  │  ├──transformers
  │  │  ├──mp_rank_00_model_states.pt
  │  ├──vae
  │  │  ├──config.json
  │  │  ├──pytorch_model.pt
```
```shell
  HunyuanVideo-I2V
    ├──README.md
    ├──hunyuan-video-i2v-720p
    │  ├──transformers
    │  │  ├──mp_rank_00_model_states.pt
    │  ├──vae
    │  ├──lora
    │  │  ├──embrace_kohaya_weights.safetensors
    │  │  ├──hair_growth_kohaya_weights.safetensors
  ```
其中`HunyuanVideo/hunyuan-video-t2v-720p/transformers`和`HunyuanVideo-I2V/hunyuan-video-i2v-720p/transformers`是transformer部分的权重，`HunyuanVideo/hunyuan-video-t2v-720p/vae`和`HunyuanVideo-I2V/hunyuan-video-i2v-720p/vae`是VAE部分的权重，`HunyuanVideo-I2V/hunyuan-video-i2v-720p/lora`是lora权重

### 权重转换
需要对`llava-llama3-8b`模型进行权重转换，运行权重转换脚本：
```shell
python examples/convert_ckpt_to_mm.py --module text_encoder --source_path <llava-llama-3-8b> --target_path <llava-llama-3-8b-text-encoder-tokenizer>
```

需要分别对hunyuanvideo-t2v和i2v的transformer部分进行权重转换，运行权重转换脚本：
```shell
python examples/hunyuanvideo/convert_ckpt_to_mm.py --source_path <hunyuan-video-t2v-720p/transformers/mp_rank_00/model_states.pt> --target_path <./ckpt/hunyuanvideo> --tp_size <tp_size>
```
```bash
python examples/hunyuanvideo/convert_ckpt_to_mm.py --source_path <hunyuan-video-i2v-720p/transformers/mp_rank_00_model_states.pt> --target_path <./ckpt/hunyuanvideo-i2v> --tp_size <tp_size>
```

需要对hunyuanvideo-i2v的lora权重转换，运行权重转换脚本：
```bash
python examples/hunyuanvideo/convert_ckpt_to_mm.py --module lora --source_path <hunyuan-video-i2v-720p/lora/embrace_kohaya_weights.safetensors> --target_path <./ckpt/hunyuanvideo-i2v-lora>
```

权重转换脚本的参数说明如下：
|参数| 含义 | 默认值 |
|:------------|:----|:----|
| --module |  转换text encoder部分或transformer部分 | "dit" （转换transformer权重） |
| --source_path | 原始权重路径 | ./transformers/mp_rank_00/model_states.pt |
| --target_path | 转换后的权重保存路径 | ./ckpt/hunyuanvideo |
| --tp_size | tp size | 2 |
| --mode | split表示按tp size对权重进行切分， merge表示按tp size对权重进行合并 | split |

---

## 预训练
### 数据预处理

将数据处理成如下格式

```bash
</data/hunyuanvideo/dataset>
  ├──data.json
  ├──videos
  │  ├──video0001.mp4
  │  ├──video0002.mp4
```

其中，`videos/`下存放视频，data.json中包含该数据集中所有的视频-文本对信息，具体示例如下：

```json
[
    {
        "path": "videos/video0001.mp4",
        "cap": "Video discrimination1.",
        "num_frames": 93,
        "fps": 24,
        "resolution": {
            "height": 480,
            "width": 848
        }
    },
    {
        "path": "videos/video0002.mp4",
        "cap": "Video discrimination2.",
        "num_frames": 93,
        "fps": 24,
        "resolution": {
            "height": 480,
            "width": 848
        }
    },
    ......
]
```

修改`examples/hunyuanvideo/feature_extract/data.txt`文件，其中每一行表示个数据集，第一个参数表示数据文件夹的路径，第二个参数表示`data.json`文件的路径，用`,`分隔

### 特征提取

#### 准备工作

在开始之前，请确认环境准备、模型权重和数据集预处理已经完成

#### 参数配置

检查模型权重路径、数据集路径、提取后的特征保存路径等配置是否完成

| 配置文件                                                     |       修改字段        | 修改说明                                            |
| ------------------------------------------------------------ | :-------------------: | :-------------------------------------------------- |
| examples/hunyuanvideo/feature_extract/data.json              |      num_frames       | 最大的帧数，超过则随机选取其中的num_frames帧        |
| examples/hunyuanvideo/feature_extract/data.json              | max_height, max_width | 最大的长宽，超过则centercrop到最大分辨率            |
| examples/hunyuanvideo/feature_extract/data.json              |    from_pretrained    | 修改为下载的权重所对应路径（包括Tokenizer） |
| examples/hunyuanvideo/feature_extract/feature_extraction.sh  |     NPUS_PER_NODE     | 卡数                                                |
| examples/hunyuanvideo/feature_extract/model_hunyuanvideo.json |    from_pretrained    | 修改为下载的权重所对应路径（包括VAE、Text Encoder） |
| examples/hunyuanvideo/feature_extract/tools.json             |       save_path       | 提取后的特征保存路径                                |

#### 启动特征提取

```bash
bash examples/hunyuanvideo/feature_extract/feature_extraction.sh
```

### 训练

#### 准备工作

在开始之前，请确认环境准备、模型权重下载、特征提取已完成。

#### 参数配置

检查模型权重路径、并行参数配置等是否完成

| 配置文件                                                   |      修改字段       | 修改说明                                            |
| ---------------------------------------------------------- | :-----------------: | :-------------------------------------------------- |
| examples/hunyuanvideo/{task_name}/data.txt                 |      文件内容       | 提取后的特征保存路径                                |
| examples/hunyuanvideo/{task_name}/feature_data.json        |   from_pretrained   | 修改为下载的权重所对应路径（包括VAE、Text Encoder） |
| examples/hunyuanvideo/{task_name}/pretrain_hunyuanvideo.sh |    NPUS_PER_NODE    | 每个节点的卡数                                      |
| examples/hunyuanvideo/{task_name}/pretrain_hunyuanvideo.sh |       NNODES        | 节点数量                                            |
| examples/hunyuanvideo/{task_name}/pretrain_hunyuanvideo.sh |      LOAD_PATH      | 权重转换后的预训练权重路径                          |
| examples/hunyuanvideo/{task_name}/pretrain_hunyuanvideo.sh |      SAVE_PATH      | 训练过程中保存的权重路径                            |
| examples/hunyuanvideo/{task_name}/pretrain_hunyuanvideo.sh |         TP          | 训练时的TP size（建议根据训练时设定的分辨率调整）   |
| examples/hunyuanvideo/{task_name}/pretrain_hunyuanvideo.sh |         CP          | 训练时的CP size（建议根据训练时设定的分辨率调整）   |
| examples/hunyuanvideo/{task_name}/pretrain_hunyuanvideo.sh | --sequence-parallel | 使能TP-SP，默认开启                                 |

【并行化配置参数说明】：

当调整模型参数或者视频序列长度时，需要根据实际情况启用以下并行策略，并通过调试确定最优并行策略。

+ CP: 序列并行，当前支持Ulysess序列并行。

  - 使用场景：在视频序列（分辨率X帧数）较大时，可以开启来降低内存占用。
  
  - 使能方式：在启动脚本中设置 CP > 1，如：CP=2；
  
  - 限制条件：head 数量需要能够被TP*CP整除（在`examples/hunyuanvideo/{task_name}/model_hunyuanvideo.json`中配置，默认为24）


+ TP: 张量模型并行

  - 使用场景：模型参数规模较大时，单卡上无法承载完整的模型，通过开启TP可以降低静态内存和运行时内存。

  - 使能方式：在启动脚本中设置 TP > 1，如：TP=8

  - 限制条件：head 数量需要能够被TP*CP整除（在`examples/hunyuanvideo/{task_name}/model_hunyuanvideo.json`中配置，默认为24）


+ TP-SP
  
  - 使用场景：在张量模型并行的基础上，进一步对 LayerNorm 和 Dropout 模块的序列维度进行切分，以降低动态内存。 

  - 使能方式：在 GPT_ARGS 设置 --sequence-parallel
  
  - 使用建议：建议在开启TP时同步开启该设置

+ layer_zero

  - 使用场景：在模型参数规模较大时，单卡上无法承载完整的模型，可以通过开启layerzero降低静态内存。
  
  - 使能方式：`examples/hunyuanvideo/{task_name}/pretrain_hunyuanvideo.sh`的`GPT_ARGS`中加入`--layerzero`和`--layerzero-config ${layerzero_config}`

  - 使用建议: 该特性和TP只能二选一，使能该特性时，TP必须设置为1，配置文件`examples/hunyuanvideo/zero_config.yaml`中的`zero3_size`推荐设置为单机的卡数
  
  - 训练权重后处理：使用该特性训练时，保存的权重需要使用下面的转换脚本进行后处理才能用于推理：
    ```bash
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    # your_mindspeed_path和your_megatron_path分别替换为之前下载的mindspeed和megatron的路径
    export PYTHONPATH=$PYTHONPATH:<your_mindspeed_path>
    export PYTHONPATH=$PYTHONPATH:<your_megatron_path>
    # input_folder为layerzero训练保存权重的路径，output_folder为输出的megatron格式权重的路径
    python <your_mindspeed_path>/mindspeed/core/distributed/layerzero/state/scripts/convert_to_megatron.py --input_folder ./save_ckpt/hunyuanvideo/iter_000xxxx/ --output_folder ./save_ckpt/hunyuanvideo_megatron_ckpt/iter_000xxxx/ --prefix predictor
    ```

+ 选择性重计算 + FA激活值offload
  
  - 如果显存比较充裕，可以开启选择性重计算（FA不进行重计算）以提高吞吐，建议同步开启FA激活值offload，将FA的激活值异步卸载至CPU
  
  - 在`examples/hunyuanvideo/{task_name}/model_hunyuanvideo.json`中，`attention_async_offload`表示是否开启FA激活值offload，默认开启

  - 在`examples/hunyuanvideo/{task_name}/model_hunyuanvideo.json`中，`double_stream_full_recompute_layers`和`single_stream_full_recompute_layers`表示该模型的double_stream_block和single_stream_block进行全重计算的层数，可以逐步减小这两个参数，直至显存打满

> ⚠️**hunyuanvideo i2v目前未适配CP与TPSP**

#### 启动训练

```bash
bash examples/hunyuanvideo/{task_name}/pretrain_hunyuanvideo.sh
```

#### 权重后处理

如果训练时`TP>1`，需要对训练得到的权重进行合并，合并后的权重才能用于推理，运行命令

```bash
python examples/convert_ckpt_to_mm.py --source_path <./save_ckpt/hunyuanvideo> --target_path <./save_ckpt_merged/hunyuanvideo> --tp_size <tp_size>
```

## I2V lora微调

### 准备工作
配置脚本前请确认环境准备已完成。

#### 权重转换
 需要对hunyuanvideo-i2v的transformer部分进行权重转换，运行权重转换脚本：
```bash
python examples/hunyuanvideo/convert_ckpt_to_mm.py --module dit --source_path <hunyuan-video-i2v-720p/transformers/mp_rank_00_model_states.pt> --tp_size 1 --target_path <./ckpt/hunyuanvideo-i2v>
```

#### 特征提取
请参考上述[特征提取](#特征提取)章节内容，并修改VAE权重为`hunyuan-video-i2v-720p`目录下的VAE权重路径


#### 配置参数

默认的配置已经经过测试，用户可按照自身环境修改如下内容：

1. 权重配置

  权重转换完成后根据实际任务情况在启动脚本文件（`examples/hunyuanvideo/i2v/pretrain_hunyuanvideo_lora.sh`）中的`LOAD_PATH="your_converted_dit_ckpt_dir"`变量中添加转换后的权重的实际路径，如`LOAD_PATH="./ckpt/hunyuanvideo-i2v"`,其中`./ckpt/hunyuanvideo-i2v`为转换后的权重的实际路径。`LOAD_PATH`变量中填写的完整路径一定要正确，填写错误的话会导致权重无法加载但运行并不会提示报错。
  根据需要填写`SAVE_PATH`变量中的路径，用以保存训练后的lora权重。


### 启动lora微调

```shell
bash examples/hunyuanvideo/i2v/pretrain_hunyuanvideo_lora.sh
```

训练完成后保存的权重仅为lora微调部分，如果需要合并到原始权重中，可以执行以下脚本完成合并（配置仅供参考）：

```bash
python  checkpoint/merge_base_lora_weight.py --base_save_dir './converted_transformer' --lora_save_dir './my_ckpt' --merge_save_dir './merge_base_lora_target' --lora-target-modules linear fc1 fc2 img_attn_qkv img_attn_proj txt_attn_qkv txt_attn_proj linear1_qkv linear1_mlp linear2 proj_out --lora-alpha 64 --lora-r 64 --pp_size 1 --tp_size 1
```

## 推理

### 准备工作

在开始之前，请确认环境准备、模型权重下载已完成

### 参数配置

检查模型权重路径、并行参数等配置是否完成

| 配置文件                                           |               修改字段               |                修改说明                 |
|---------------------------------------------------|:--------------------------------:|:-----------------------------------|
| examples/hunyuanvideo/{task_name}/inference_model.json |         from_pretrained          |            修改为下载的权重所对应路径（包括VAE、Text Encoder）            |
| examples/hunyuanvideo/{task_name}/samples_prompts.txt |               文件内容               |      可自定义自己的prompt，一行为一个prompt      |
| examples/hunyuanvideo/{task_name}/inference_model.json |  input_size |  生成视频的分辨率，格式为 [t, h, w] |
| examples/hunyuanvideo/{task_name}/inference_model.json |  save_path |  生成视频的保存路径 |
| examples/hunyuanvideo/{task_name}/inference_hunyuanvideo.sh |   LOAD_PATH | 转换之后的transform部分权重路径 |

### 启动推理

```shell
bash examples/hunyuanvideo/{task_name}/inference_hunyuanvideo.sh
```

## 环境变量声明
ASCEND_SLOG_PRINT_TO_STDOUT： 是否开启日志打印， 0：关闭日志打屏，1：开启日志打屏  
ASCEND_GLOBAL_LOG_LEVEL： 设置应用类日志的日志级别及各模块日志级别，仅支持调试日志。0：对应DEBUG级别，1：对应INFO级别，2：对应WARNING级别，3：对应ERROR级别，4：对应NULL级别，不输出日志  
TASK_QUEUE_ENABLE： 用于控制开启task_queue算子下发队列优化的等级，0：关闭，1：开启Level 1优化，2：开启Level 2优化  
COMBINED_ENABLE： 设置combined标志。设置为0表示关闭此功能；设置为1表示开启，用于优化非连续两个算子组合类场景  
CPU_AFFINITY_CONF： 控制CPU端算子任务的处理器亲和性，即设定任务绑核，设置0或未设置：表示不启用绑核功能， 1：表示开启粗粒度绑核， 2：表示开启细粒度绑核  
HCCL_CONNECT_TIMEOUT:  用于限制不同设备之间socket建链过程的超时等待时间，需要配置为整数，取值范围[120,7200]，默认值为120，单位s  
PYTORCH_NPU_ALLOC_CONF： 控制缓存分配器行为  
NPUS_PER_NODE： 配置一个计算节点上使用的NPU数量