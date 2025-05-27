# OpenSoraPlan1.2 使用指南

<p align="left">
</p>

## 目录

- [环境安装](#jump1)
  - [仓库拉取](#jump1.1)
  - [环境搭建](#jump1.2)
- [权重下载及转换](#jump2)
  - [权重下载](#jump2.1)
  - [权重转换](#jump2.2)
- [数据集准备及处理](#jump3)
  - [数据集下载](#jump3.1)
- [预训练](#jump4)
  - [准备工作](#jump4.1)
  - [配置参数](#jump4.2)
  - [启动预训练](#jump4.3)
- [推理](#jump5)
  - [准备工作](#jump5.1)
  - [配置参数](#jump5.2)
  - [启动推理](#jump5.3)
- [环境变量声明](#jump6)

---
<a id="jump1"></a>

## 环境安装

【模型开发时推荐使用配套的环境版本】

请参考[安装指南](https://gitee.com/ascend/MindSpeed-MM/blob/master/docs/features/install_guide.md)

<a id="jump1.1"></a>

#### 1. 仓库拉取

```shell
    git clone https://gitee.com/ascend/MindSpeed-MM.git 
    git clone https://github.com/NVIDIA/Megatron-LM.git
    cd Megatron-LM
    git checkout core_r0.8.0
    cp -r megatron ../MindSpeed-MM/
    cd ..
    cd MindSpeed-MM
```

<a id="jump1.2"></a>

#### 2. 环境搭建

```bash
    # python3.8
    conda create -n test python=3.8
    conda activate test

    # 安装 torch 和 torch_npu，注意要选择对应python版本、x86或arm的torch、torch_npu及apex包
    pip install torch-2.1.0-cp38-cp38m-manylinux2014_aarch64.whl 
    pip install torch_npu-2.1.0*-cp38-cp38m-linux_aarch64.whl
    
    # apex for Ascend 参考 https://gitee.com/ascend/apex
    # 建议从原仓编译安装

    # 将shell脚本中的环境变量路径修改为真实路径，下面为参考路径
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 

    # 安装加速库
    git clone https://gitee.com/ascend/MindSpeed.git
    cd MindSpeed
    git checkout 3f09d6736571cf1e30f8ac97de77982d0ab32cc5
    pip install -r requirements.txt 
    pip3 install -e .
    cd ..

    # 安装其余依赖库
    pip install -e .
```

#### 3. Decord搭建

【X86版安装】

```bash
pip install decord==0.6.0
```

【ARM版安装】

`apt`方式安装请[参考链接](https://github.com/dmlc/decord)

`yum`方式安装请[参考脚本](https://github.com/dmlc/decord/blob/master/tools/build_manylinux2010.sh)

---

<a id="jump2"></a>

## 权重下载及转换

<a id="jump2.1"></a>

#### 1. 权重下载

从Huggingface等网站下载开源模型权重

- [LanguageBind/Open-Sora-Plan-v1.2.0](https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.2.0/tree/main)：CasualVAE模型和VideoDiT模型；

- [DeepFloyd/mt5-xxl](https://huggingface.co/google/mt5-xxl/)： MT5模型；

<a id="jump2.2"></a>

#### 2. 权重转换

MindSpeed-MM修改了部分原始网络的结构名称，因此需要使用`convert_ckpt_to_mm.py`脚本进行转换，该脚本实现了从hugging face下载的预训练权重到MindSpeed-MM权重的转换以及TP（Tensor Parallel）权重的切分与合并。

首先修改 examples/opensoraplan1.2/convert_ckpt_to_mm.py 传入参数。
权重转换脚本的参数说明与默认值如下：
|参数| 含义 | 默认值 |
|:------------|:----|:----|
| --tp_size | tp size | 1 |
| --dit_hg_weight_path | dit部分原始权重路径 | "./raw_ckpt/open-sora-plan/93x480p/diffusion_pytorch_model.safetensors"
| --dit_mm_save_path | dit部分转换或切分后权重保存路径 | "./ckpt/open-sora-plan-12/93x480p" |
| --vae_convert | 是否转换vae权重 | True |
| --vae_hg_weight_path | vae部分原始权重路径 | "./raw_ckpt/open-sora-plan/vae/checkpoint.ckpt" |
| --vae_mm_save_path | vae部分转换后权重保存路径 | "./ckpt/vae" |
| --dit_mm_weight_path | 用于合并操作的，TP切分的dit权重路径 | "./ckpt/open-sora-plan-12/93x480p" |
| --dit_merge_save_path | 合并后dit权重保存路径 | "./ckpt/open-sora-plan-12/merge" |
| --mode | split表示按tp size对权重进行切分， merge表示按tp size对权重进行合并 | "split" |


启动脚本

    # 根据实际情况修改 ascend-toolkit 路径

    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    python examples/opensoraplan1.2/convert_ckpt_to_mm.py   --mode="split" --tp-size=2
    


同步修改examples/opensoraplan1.2/pretrain_opensoraplan1_2.sh 中的--load参数，该路径为转换后或者切分后的权重，注意--load配置的是转换到MindSpeed-MM后的dit权重路径，vae权重路径在model_opensoraplan1_2.json中配置

    LOAD_PATH="./ckpt/open-sora-plan-12/93x480p"

---
<a id="jump3"></a>

## 数据集准备及处理

<a id="jump3.1"></a>

#### 1. 数据集下载

用户需自行获取并解压[pixabay_v2](https://huggingface.co/datasets/LanguageBind/Open-Sora-Plan-v1.1.0/tree/main/pixabay_v2_tar)数据集，获取数据结构如下：

   ```
   $pixabay_v2
   ├── annotation.json
   ├── folder_01
   ├── ├── video0.mp4
   ├── ├── video1.mp4
   ├── ├── ...
   ├── ├── annotation.json
   ├── folder_02
   ├── folder_03
   └── ...
   ```

---

<a id="jump4"></a>

## 预训练

<a id="jump4.1"></a>

#### 1. 准备工作

配置脚本前需要完成前置准备工作，包括：**环境安装**、**权重下载及转换**、**数据集准备及处理**，详情可查看对应章节。

<a id="jump4.2"></a>

#### 2. 配置参数

需根据实际情况修改`model_opensoraplan1_2.json`和`data.json`中的权重和数据集路径，包括`from_pretrained`、`data_path`、`data_folder`字段。

【interleaved特性】

在`model_opensoraplan1_2.json`文件中，启用`interleaved`参数可以在DIT进行forward前，批量完成`interleaved_steps`个batch的数据encode。从而在DIT的forward过程中，卸载`encoder`到cpu上，降低显存峰值。
该特性默认关闭，适用于模型规模较大或视频序列较长的自定义训练场景，可以有效降低内存占用。
特性启用后，由于`encoder`会根据`interleaved_steps`数值大小，随着数据的批量处理反复从CPU加载到NPU上，再从NPU上卸载，所以`interleaved_steps`的数值应足够大以避免因为模型的反复加载影响训练性能。经测试以OpenSoraPlan1.2为例，`interleaved_steps`大于10后对整体性能影响较小。

【单机运行】

```shell
    GPUS_PER_NODE=8
    MASTER_ADDR=localhost
    MASTER_PORT=29501
    NNODES=1  
    NODE_RANK=0  
    WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))
```

【多机运行】

```shell
    # 根据分布式集群实际情况配置分布式参数
    GPUS_PER_NODE=8  #每个节点的卡数
    MASTER_ADDR="your master node IP"  #都需要修改为主节点的IP地址（不能为localhost）
    MASTER_PORT=29501
    NNODES=2  #集群里的节点数，以实际情况填写,
    NODE_RANK="current node id"  #当前节点的RANK，多个节点不能重复，主节点为0, 其他节点可以是1,2..
    WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))
```

<a id="jump4.3"></a>

#### 3. 启动预训练

```shell
    bash examples/opensoraplan1.2/pretrain_opensoraplan1_2.sh
```

**注意**：

- 多机训练需在多个终端同时启动预训练脚本(每个终端的预训练脚本只有NODE_RANK参数不同，其他参数均相同)
- 如果使用多机训练，需要在每个节点准备训练数据和模型权重

---

<a id="jump5"></a>

## 推理

<a id="jump5.1"></a>

#### 1. 准备工作

参考上述的权重下载及转换章节，需求的权重需要到huggingface中下载，以及参考上面的权重转换代码进行转换。
链接参考: [predict_model](https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.2.0/tree/main/29x480p) [VAE](https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.2.0/tree/main/vae) [tokenizer/text_encoder](https://huggingface.co/google/mt5-xxl/tree/main)

<a id="jump5.2"></a>

#### 2. 配置参数

将准备好的权重传入到inference_model_29x480x640.json中，更改其中的路径，包括from_pretrained，自定义的prompt可以传入到prompt字段中

<a id="jump5.3"></a>

#### 3. 启动推理

启动推理脚本

```shell
bash examples/opensoraplan1.2/inference_opensoraplan1_2.sh
```

---
<a id="jump6"></a>
## 环境变量声明
ASCEND_RT_VISIBLE_DEVICES： 指定NPU设备的索引值  
ASCEND_SLOG_PRINT_TO_STDOUT： 是否开启日志打印， 0：关闭日志打屏，1：开启日志打屏  
ASCEND_GLOBAL_LOG_LEVEL： 设置应用类日志的日志级别及各模块日志级别，仅支持调试日志。0：对应DEBUG级别，1：对应INFO级别，2：对应WARNING级别，3：对应ERROR级别，4：对应NULL级别，不输出日志  
TASK_QUEUE_ENABLE： 用于控制开启task_queue算子下发队列优化的等级，0：关闭，1：开启Level 1优化，2：开启Level 2优化  
COMBINED_ENABLE： 设置combined标志。设置为0表示关闭此功能；设置为1表示开启，用于优化非连续两个算子组合类场景  
CPU_AFFINITY_CONF： 控制CPU端算子任务的处理器亲和性，即设定任务绑核，设置0或未设置：表示不启用绑核功能， 1：表示开启粗粒度绑核， 2：表示开启细粒度绑核  
HCCL_CONNECT_TIMEOUT:  用于限制不同设备之间socket建链过程的超时等待时间，需要配置为整数，取值范围[120,7200]，默认值为120，单位s  
NPUS_PER_NODE： 配置一个计算节点上使用的NPU数量  
GPUS_PER_NODE： 配置一个计算节点上使用的GPU数量