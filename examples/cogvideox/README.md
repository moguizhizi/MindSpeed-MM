# CogVideoX 使用指南

<p align="left">
</p>

## 目录
- [CogVideoX 使用指南](#cogvideox-使用指南)
  - [目录](#目录)
  - [支持任务列表](#支持任务列表)
  - [环境安装](#环境安装)
      - [仓库拉取](#仓库拉取)
      - [环境搭建](#环境搭建)
      - [Decord搭建](#decord搭建)
  - [权重下载及转换](#权重下载及转换)
      - [VAE下载](#vae下载)
      - [transformer文件下载](#transformer文件下载)
      - [T5模型下载](#t5模型下载)
      - [权重转换](#权重转换)
  - [数据集准备及处理](#数据集准备及处理)
  - [预训练](#预训练)
      - [准备工作](#准备工作)
      - [配置参数](#配置参数)
      - [启动预训练](#启动预训练)
  - [推理](#推理)
      - [准备工作](#准备工作-1)
      - [配置参数](#配置参数-1)
      - [启动推理](#启动推理)
  - [lora微调](#lora微调)
      - [准备工作](#准备工作-2)
      - [配置参数](#配置参数-2)
      - [启动lora微调](#启动lora微调)
  - [预训练模型扩参示例(15B)](#预训练模型扩参示例15b)
      - [模型参数修改](#模型参数修改)
      - [启动脚本修改](#启动脚本修改)
  - [环境变量声明](#环境变量声明)
---
<a id="jump1"></a>
## 支持任务列表
支持以下模型任务类型

|      模型      | 任务类型 | 任务列表 | 是否支持 |
|:------------:|:----:|:----:|:-----:|
| CogVideoX-5B | t2v  |预训练  | ✔ |
| CogVideoX-5B | t2v  |在线推理 | ✔ |
| CogVideoX-5B | i2v  |预训练  | ✔ |
| CogVideoX-5B | i2v  |在线推理 | ✔ |

<a id="jump2"></a>
## 环境安装

【模型开发时推荐使用配套的环境版本】

请参考[安装指南](https://gitee.com/ascend/MindSpeed-MM/blob/master/docs/features/install_guide.md)

<a id="jump2.1"></a>
#### 仓库拉取

```shell
git clone https://gitee.com/ascend/MindSpeed-MM.git 
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout core_r0.8.0
cp -r megatron ../MindSpeed-MM/
cd ..
cd MindSpeed-MM
```
<a id="jump2.2"></a>
#### 环境搭建

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
git checkout 6f11a6c9edd409f32a805a71e710b01f9191438f
pip install -r requirements.txt 
pip install -e .
cd ..

# 安装其余依赖库
pip install -e .
```
<a id="jump2.3"></a>
#### Decord搭建

【X86版安装】

```bash
pip install decord==0.6.0
```

【ARM版安装】

`apt`方式安装请[参考链接](https://github.com/dmlc/decord)

`yum`方式安装请[参考脚本](https://github.com/dmlc/decord/blob/master/tools/build_manylinux2010.sh)

---
<a id="jump3"></a>
## 权重下载及转换

<a id="jump3.1"></a>
#### VAE下载

+ [VAE下载链接](https://cloud.tsinghua.edu.cn/f/fdba7608a49c463ba754/?dl=1)

<a id="jump3.2"></a>
#### transformer文件下载
+ [CogVideoX1.0-5B-t2v](https://cloud.tsinghua.edu.cn/d/fcef5b3904294a6885e5/?p=%2F&mode=list)
+ [CogVideoX1.0-5B-i2v](https://cloud.tsinghua.edu.cn/d/5cc62a2d6e7d45c0a2f6/?p=%2F1&mode=list)
+ [CogVideoX1.5-5B-t2v](https://huggingface.co/THUDM/CogVideoX1.5-5B-SAT/tree/main/transformer_t2v)
+ [CogVideoX1.5-5B-i2v](https://huggingface.co/THUDM/CogVideoX1.5-5B-SAT/tree/main/transformer_i2v)

<a id="jump3.3"></a>
#### T5模型下载
仅需下载tokenizer和text_encoder目录的内容：[下载链接](https://huggingface.co/THUDM/CogVideoX-5b/tree/main)

预训练权重结构如下：
   ```
   CogVideoX-5B
   ├── text_encoder
   │   ├── config.json
   │   ├── model-00001-of-00002.safetensors
   │   ├── model-00002-of-00002.safetensors
   │   └── model.safetensors.index.json   
   ├── tokenizer
   │   ├── added_tokens.json
   │   ├── special_tokens_map.json
   │   ├── spiece.model
   │   └── tokenizer_config.json   
   ├── transformer
   │   ├── 1000 (or 1)
   │   │   └── mp_rank_00_model_states.pt
   │   └── latest
   └── vae
       └── 3d-vae.pt
   ```

<a id="jump3.4"></a>
#### 权重转换
权重转换source_path参数请配置transformer权重文件的路径：
```bash
python examples/cogvideox/cogvideox_sat_convert_to_mm_ckpt.py --source_path <your source path> --target_path <target path> --task t2v --tp_size 1 --pp_size 10 11 11 10 --num_layers 42 --mode split
```
其中--tp_size 后为实际的tp切分策略， --task 的值为t2v或i2v，
当开启PP时，--pp_size 后参数值个数与PP的数值相等，并且参数之和与--num_layers 参数相等，举例：当PP=4, --num_layers 4, --pp_size 1 1 1 1; 当PP=4, --num_layers 42, --pp_size 10 11 11 10 

转换后的权重结构如下：

TP=1,PP=1时：
```
CogVideoX-5B-Converted
├── release
│   └──mp_rank_00
│      └──model_optim_rng.pt
└──latest_checkpointed_iterations.txt
```
TP=2,PP=1, TP>2的情况依此类推：
```
CogVideoX-5B-Converted
├── release
│   ├──mp_rank_00
│   │    └──model_optim_rng.pt
│   └──mp_rank_01
│      └──model_optim_rng.pt
└──latest_checkpointed_iterations.txt
```
TP=1,PP=4, PP>1及TP>1的情况依此类推：
```
CogVideoX-5B-Converted
├── release
│   ├──mp_rank_00_000
│   │   └──model_optim_rng.pt
│   ├──mp_rank_00_001
│   │   └──model_optim_rng.pt
│   ├──mp_rank_00_002
│   │   └──model_optim_rng.pt
│   └──mp_rank_00_003
│       └──model_optim_rng.pt
└──latest_checkpointed_iterations.txt
```

---
<a id="jump4"></a>
## 数据集准备及处理

数据集格式应该如下：
```
.
├── data.jsonl
├── labels
│   ├── 1.txt
│   ├── 2.txt
│   ├── ...
└── videos
    ├── 1.mp4
    ├── 2.mp4
    ├── ...
```
每个 txt 与视频同名，为视频的标签。视频与标签应该一一对应。

data.jsonl文件内容如下示例：
```
{"file": "dataPath/1.mp4", "captions": "Content from 1.txt"}
{...}
...
```

---
<a id="jump5"></a>
## 预训练

<a id="jump5.1"></a>
#### 准备工作
配置脚本前需要完成前置准备工作，包括：**环境安装**、**权重下载及转换**、**数据集准备及处理**，详情可查看对应章节。

<a id="jump5.2"></a>
#### 配置参数
<a id="jump5.2"></a>
CogvideoX训练阶段的启动文件为shell脚本，主要分为如下4个：
|            | I2V | T2V |
|:------------:|:----:|:----:|
| 1.0 |  pretrain_cogvideox_i2v.sh |pretrain_cogvideox_t2v.sh  |
| 1.5 | pretrain_cogvideox_i2v_1.5.sh |pretrain_cogvideox_t2v_1.5.sh |

模型参数的配置文件如下：
|            | I2V | T2V |
|:------------:|:----:|:----:|
| 1.0 |  model_cogvideox_i2v.json |model_cogvideox_t2v.json  |
| 1.5 | model_cogvideox_i2v_1.5.json |model_cogvideox_t2v_1.5.json |

以及涉及训练数据集的`data.json`文件

默认的配置已经经过测试，用户可按照自身环境修改如下内容：

1. 权重配置

  需根据实际任务情况在启动脚本文件（如`pretrain_cogvideox_i2v.sh`）中的`LOAD_PATH="your_converted_dit_ckpt_dir"`变量中添加转换后的权重的实际路径，如`LOAD_PATH="./CogVideoX-5B-Converted"`,其中`./CogVideoX-5B-Converted`为转换后的权重的实际路径，其文件夹内容结构如权重转换一节所示。`LOAD_PATH`变量中填写的完整路径一定要正确，填写错误的话会导致权重无法加载但运行并不会提示报错。

根据需要填写`SAVE_PATH`变量中的路径，用以保存训练后的权重。

2. 数据集路径配置

  根据实际情况修改`data.json`中的数据集路径，分别为`"data_path":"/data_path/data.jsonl"`、`"data_folder":"/data_path/"`，替换`"/data_path/"`为实际的数据集路径。

3. VAE及T5模型路径配置

  根据实际情况修改模型参数配置文件（如`model_cogvideox_i2v.json`）以及`data.json`文件中VAE及T5模型文件的实际路径。其中，T5文件的路径字段为`"from_pretrained": "5b-cogvideo/tokenizer"`及`"from_pretrained": "5b-cogvideo"`，替换`5b-cogvideo`为实际的路径；VAE模型文件的路径字段为`"from_pretrained": "3d-vae.pt"`，替换`3d-vae.pt`为实际的路径。

  当需要卸载VAE跟T5时，将模型参数配置文件中的`"load_video_features": false`及`"load_text_features": false`字段中的值分别改为`true`。将`data.json`中的`"use_feature_data"`字段的值改为`true`。

4. 切分策略配置

* 当PP开启时，在启动脚本文件中添加`--optimization-level 2 --use-multiparameter-pipeline-model-parallel`参数，并且在模型参数配置文件中的将`pipeline_num_layers`参数的值由`null`改为实际切分情况，例如PP=4，num_layers=42时，`"pipeline_num_layers":[11, 10, 11, 10]`,具体值根据实际的PP切分策略确定。

* 当开启VAE CP时，修改模型参数配置文件中的`ae`字典内的关键字`cp_size`的值为所需要的值，不兼容Encoder-DP、未验证与分层Zero效果。

* 当开启SP时，在启动脚本文件中添加`--sequence-parallel`参数。

* 当开启Encoder-DP时，需要将[model_cogvideox_i2v_1.5.json](i2v_1.5/model_cogvideox_i2v_1.5.json) 或者[model_cogvideox_t2v_1.5.json](t2v_1.5/model_cogvideox_t2v_1.5.json)中的`enable_encoder_dp`选项改为`true`。注意:需要在开启CP/TP，并且`load_video_features`为`false`及`load_text_features`为`false`才能启用，不兼容PP场景、VAE-CP、分层Zero。

* 当开启分层Zero时，需要在[pretrain_cogvideox_t2v_1.5.sh](t2v_1.5/pretrain_cogvideox_t2v_1.5.sh)或者[pretrain_cogvideox_i2v_1.5.sh](i2v_1.5/pretrain_cogvideox_i2v_1.5.sh)里面添加下面的参数。
  注意：不兼容Encoder-DP特性、TP场景、PP场景，与VAE-CP效果未验证。
  ```shell
  --layerzero \
  --layerzero-config ./zero_config.yaml \
  ```
  参数里面的yaml文件如下面所示:
  ```yaml
  zero3_size: 8  
  transformer_layers:
    - mindspeed_mm.models.predictor.dits.sat_dit.VideoDiTBlock
  backward_prefetch: 'BACKWARD_PRE'
  param_dtype: "bf16"
  reduce_dtype: "fp32"
  forward_prefetch: True
  limit_all_gathers: True
  ignored_modules:
    - ae
    - text_encoder
  ```
  
  该特性和TP不能兼容，开启时TP必须设置为1，使用该特性训练时，保存的权重需要使用下面的转换脚本进行后处理才能用于推理：
  
      ```bash
      source /usr/local/Ascend/ascend-toolkit/set_env.sh
      # your_mindspeed_path和your_megatron_path分别替换为之前下载的mindspeed和megatron的路径
      export PYTHONPATH=$PYTHONPATH:<your_mindspeed_path>
      export PYTHONPATH=$PYTHONPATH:<your_megatron_path>
      # input_folder为layerzero训练保存权重的路径，output_folder为输出的megatron格式权重的路径
      python <your_mindspeed_path>/mindspeed/core/distributed/layerzero/state/scripts/convert_to_megatron.py --input_folder ./save_ckpt/hunyuanvideo/iter_000xxxx/ --output_folder ./save_ckpt/hunyuanvideo_megatron_ckpt/iter_000xxxx/ --prefix predictor
      ```

模型参数配置文件中的`head_dim`字段原模型默认配置为64。此字段调整为128会更加亲和昇腾。

在sh启动脚本中可以修改运行卡数(NNODES为节点数，GPUS_PER_NODE为每个节点的卡数，相乘即为总运行卡数)：
```shell
GPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=29501
NNODES=1  
NODE_RANK=0  
WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))
```
<a id="jump5.3"></a>
#### 启动预训练

t2v 1.0版本任务启动预训练
```shell
bash examples/cogvideox/t2v_1.0/pretrain_cogvideox_t2v.sh
```
t2v 1.5版本任务启动预训练
```shell
bash examples/cogvideox/t2v_1.5/pretrain_cogvideox_t2v_1.5.sh
```
i2v 1.0版本任务启动预训练
```shell
bash examples/cogvideox/i2v_1.0/pretrain_cogvideox_i2v.sh
```
i2v 1.5版本任务启动预训练
```shell
bash examples/cogvideox/i2v_1.5/pretrain_cogvideox_i2v_1.5.sh
```
---

<a id="jump6"></a>
## 推理

<a id="jump6.1"></a>
#### 准备工作

在开始之前，请确认环境准备、模型权重下载已完成

<a id="jump6.2"></a>
#### 配置参数

CogvideoX推理启动文件为shell脚本，主要分为如下4个：
|            | I2V | T2V |
|:------------:|:----:|:----:|
| 1.0 |  inference_cogvideox_i2v.sh |inference_cogvideox_t2v.sh  |
| 1.5 | inference_cogvideox_i2v_1.5.sh |inference_cogvideox_t2v_1.5.sh |

模型参数的配置文件如下：
|            | I2V | T2V |
|:------------:|:----:|:----:|
| 1.0 |  inference_model_i2v.json |inference_model_t2v.json  |
| 1.5 | inference_model_i2v_1.5.json |inference_model_t2v_1.5.json |

1. 权重配置

  需根据实际任务情况在启动脚本文件（如`inference_cogvideox_i2v.sh`）中的`LOAD_PATH="your_converted_dit_ckpt_dir"`变量中添加转换后的权重的实际路径，如`LOAD_PATH="./CogVideoX-5B-Converted"`,其中`./CogVideoX-5B-Converted`为转换后的权重的实际路径，其文件夹内容结构如权重转换一节所示。`LOAD_PATH`变量中填写的完整路径一定要正确，填写错误的话会导致权重无法加载但运行并不会提示报错。

2. VAE及T5模型路径配置

  根据实际情况修改模型参数配置文件（如`inference_model_i2v.json`）中VAE及T5模型文件的实际路径。其中，T5文件的路径字段为`"from_pretrained": "5b-cogvideo"`，替换`5b-cogvideo`为实际的路径；VAE模型文件的路径字段为`"from_pretrained": "3d-vae.pt"`，替换`3d-vae.pt`为实际的路径。

3. prompts配置

| t2v prompts配置文件                               |               修改字段               |                修改说明                 |
|----------------------------------------|:--------------------------------:|:-----------------------------------:|
| examples/cogvideox/samples_prompts.txt |               文件内容               |      自定义prompt      |


| i2v prompts配置文件                                   |               修改字段               |       修改说明       |
|--------------------------------------------|:--------------------------------:|:----------------:|
| examples/cogvideox/samples_i2v_images.txt  |               文件内容               |       图片路径       |
| examples/cogvideox/samples_i2v_prompts.txt |               文件内容               |    自定义prompt     |


如果使用训练后保存的权重进行推理，需要使用脚本进行转换，权重转换source_path参数请配置训练时的保存路径
```bash
python examples/cogvideox/cogvideox_sat_convert_to_mm_ckpt.py --source_path <your source path> --target_path <target path> --task t2v --tp_size 1 --pp_size 42  --num_layers 42 --mode merge
```

<a id="jump6.3"></a>
#### 启动推理
t2v 1.0版本启动推理脚本

```bash
bash examples/cogvideox/t2v_1.0/inference_cogvideox_t2v.sh
```
t2v 1.5版本启动推理脚本

```bash
bash examples/cogvideox/t2v_1.5/inference_cogvideox_t2v_1.5.sh
```
i2v 1.0版本启动推理脚本

```bash
bash examples/cogvideox/i2v_1.0/inference_cogvideox_i2v.sh
```
i2v 1.5版本启动推理脚本

```bash
bash examples/cogvideox/i2v_1.5/inference_cogvideox_i2v_1.5.sh
```
---

<a id="jump7"></a>
## lora微调

<a id="jump7.1"></a>
#### 准备工作
配置脚本前请确认环境准备已完成。

1. 权重下载及转换

 模型权重下载链接(链接下包含模型权重以及tokenizer和text_encoder):

 + [t2v下载链接](https://huggingface.co/THUDM/CogVideoX1.5-5B/tree/main)
 + [i2v下载链接](https://huggingface.co/THUDM/CogVideoX1.5-5B-I2V/tree/main)
  
  lora微调功能的权重转换使用`cogvideox_hf_convert_to_mm_ckpt.py`脚本，注意`source_path`为权重所在目录的目录名，非权重本身。
```bash
python examples/cogvideox/cogvideox_hf_convert_to_mm_ckpt.py --source_path <your source path> --target_path <target path> --task t2v --tp_size 1 --pp_size 42 --num_layers 42 --mode split
```

 VAE权重下载

+ [VAE下载链接](https://cloud.tsinghua.edu.cn/f/fdba7608a49c463ba754/?dl=1)

2. 数据集准备及处理

[lora数据集下载链接](https://huggingface.co/datasets/Wild-Heart/Disney-VideoGeneration-Dataset)
  
原始数据集不包含MM套件所需的data.jsonl文件形式，需要将原始数据集中prompt.txt和videos.txt合并生成data.jsonl文件。
  
推荐使用提供的`cogvideox_lora_dataset_convert.py`脚本完成转换:
```bash
python examples/cogvideox/cogvideox_lora_dataset_convert.py --video_path '/data_path/videos.txt' --prompt_path '/data_path/prompt.txt' --output_path '/data_path/data.jsonl'
```

<a id="jump7.2"></a>
#### 配置参数
CogvideoX lora微调阶段的启动文件为shell脚本，主要分为如下2个：
|            | I2V | T2V |
|:------------:|:----:|:----:|
| 1.5 | finetune_cogvideox_lora_i2v_1.5.sh |finetune_cogvideox_lora_t2v_1.5.sh |

模型参数的配置文件如下：
|            | I2V | T2V |
|:------------:|:----:|:----:|
| 1.5 | model_cogvideox_i2v_1.5.json |model_cogvideox_t2v_1.5.json |

以及涉及训练数据集的`data.json`文件

默认的配置已经经过测试，用户可按照自身环境修改如下内容：

1. 权重配置

  权重转换完成后根据实际任务情况在启动脚本文件（如`finetune_cogvideox_lora_i2v_1.5.sh`）中的`LOAD_PATH="your_converted_dit_ckpt_dir"`变量中添加转换后的权重的实际路径，如`LOAD_PATH="./CogVideoX-5B-Converted"`,其中`./CogVideoX-5B-Converted`为转换后的权重的实际路径，其文件夹内容结构如权重转换一节所示。`LOAD_PATH`变量中填写的完整路径一定要正确，填写错误的话会导致权重无法加载但运行并不会提示报错。
  根据需要填写`SAVE_PATH`变量中的路径，用以保存训练后的lora权重。

2. 数据集路径配置
  
  准备好数据集后，根据实际情况修改`data.json`中的数据集路径，分别为`"data_path":"/data_path/data.jsonl"`、`"data_folder":"/data_path/"`，替换`"/data_path/"`为实际的数据集路径。

3. VAE及T5模型路径配置

  请参考预训练相同章节
  
4. 切分策略配置

  请参考预训练相同章节

<a id="jump7.3"></a>
#### 启动lora微调


t2v 1.5版本任务启动微调
```shell
bash examples/cogvideox/t2v_1.5/finetune_cogvideox_lora_t2v_1.5.sh
```

i2v 1.5版本任务启动微调
```shell
bash examples/cogvideox/i2v_1.5/finetune_cogvideox_lora_i2v_1.5.sh
```

训练完成后保存的权重仅为lora微调部分，如果需要合并到原始权重中，可以执行以下脚本完成合并（配置仅供参考）：

```bash
python  checkpoint/merge_base_lora_weight.py --base_save_dir './converted_transformer' --lora_save_dir './my_ckpt' --merge_save_dir './merge_base_lora_target' --lora_target_modules proj_qkv proj_out --lora_alpha 64 --lora_r 128 --pp_size 1 --tp_size 1
```

---

<a id="jump8"></a>
## 预训练模型扩参示例(15B)

<a id="jump8.1"></a>
#### 模型参数修改
通过增加扩散模型层数等配置可以模拟15B参数量，如下所示，修改模型参数配置文件（`model_cogvideox_i2v.json`）中`"predictor"`下的`"num_layers"`、`"num_heads"`和`"head_dim"`的值

```
"predictor": {
    "num_layers": 64,
    "num_heads": 32,
    "head_dim": 128,
    ...
}
```

<a id="jump8.2"></a>
#### 启动脚本修改

修改GPT_ARGS参数如下，根据实际分辨率、帧数调整启动脚本中的分布式配置（单机16卡CP4效果较佳）：

```shell
GPUS_PER_NODE=16
MASTER_ADDR={主节点IP}
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))

TP=4
PP=1
CP=4
MBS=1
GBS=$(($WORLD_SIZE*$MBS/$CP/$TP))

GPT_ARGS="
    --num-attention-heads 64 \
    --seq-length 32 \
    --max-position-embeddings 32 \
    --recompute-num-layers 64 \
    ...
"
```
<a id="jump9"></a>
## 环境变量声明
ASCEND_SLOG_PRINT_TO_STDOUT： 是否开启日志打印， 0：关闭日志打屏，1：开启日志打屏  
ASCEND_GLOBAL_LOG_LEVEL： 设置应用类日志的日志级别及各模块日志级别，仅支持调试日志。0：对应DEBUG级别，1：对应INFO级别，2：对应WARNING级别，3：对应ERROR级别，4：对应NULL级别，不输出日志  
TASK_QUEUE_ENABLE： 用于控制开启task_queue算子下发队列优化的等级，0：关闭，1：开启Level 1优化，2：开启Level 2优化  
COMBINED_ENABLE： 设置combined标志。设置为0表示关闭此功能；设置为1表示开启，用于优化非连续两个算子组合类场景  
CPU_AFFINITY_CONF： 控制CPU端算子任务的处理器亲和性，即设定任务绑核，设置0或未设置：表示不启用绑核功能， 1：表示开启粗粒度绑核， 2：表示开启细粒度绑核  
HCCL_CONNECT_TIMEOUT:  用于限制不同设备之间socket建链过程的超时等待时间，需要配置为整数，取值范围[120,7200]，默认值为120，单位s  
PYTORCH_NPU_ALLOC_CONF： 控制缓存分配器行为  
NPUS_PER_NODE： 配置一个计算节点上使用的NPU数量  
GPUS_PER_NODE： 配置一个计算节点上使用的GPU数量