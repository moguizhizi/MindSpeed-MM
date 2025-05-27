# OpenSora1.0 使用指南

<p align="left">
</p>

## 目录

- [环境安装](#jump1)
  - [仓库拉取](#jump1.1)
  - [环境搭建](#jump1.2)
- [权重下载及转换](#jump2)
  - [权重下载](#jump2.1)
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
    git checkout 6f11a6c9edd409f32a805a71e710b01f9191438f
    pip install -r requirements.txt 
    pip3 install -e .
    cd ..

    # 安装其余依赖库
    pip install -e .
```

---

<a id="jump2"></a>

## 权重下载及转换

<a id="jump2.1"></a>

#### 1. 权重下载

从Huggingface等网站下载开源模型权重

- [PixArt-alpha/PixArt-alpha](https://huggingface.co/PixArt-alpha/PixArt-alpha/resolve/main/PixArt-XL-2-512x512.pth)   - PixArt-XL-2-512x512模型(训练用)
- [stabilityai/sd-vae-ft-ema](https://huggingface.co/stabilityai/sd-vae-ft-ema)   - vae模型
- [DeepFloyd/t5-v1_1-xxl](https://huggingface.co/DeepFloyd/t5-v1_1-xxl)       -  t5模型
- [hpcai-tech/Open-Sora](https://huggingface.co/hpcai-tech/Open-Sora/resolve/main/OpenSora-v1-HQ-16x512x512.pth)        -  预训练权重(推理用)

---

<a id="jump3"></a>

## 数据集准备及处理

<a id="jump3.1"></a>

#### 1. 数据集下载

用户需自行获取并解压MSRVTT数据集，
参考[OpenSora官方](https://github.com/hpcaitech/Open-Sora/tree/v1.0.0/tools/datasets)进行数据集预处理，
获取数据结构如下：

   ```
   $MSRVTT
   ├── train
   ├── ├── videos
   ├── ├── ├── video0.mp4
   ├── ├── ├── video1.mp4
   ├── ├── ├── ...
   ├── ├── annotation.json
   ├── val
   ├── test
   └── ...
   ```

---

<a id="jump4"></a>

## 预训练

<a id="jump4.1"></a>

#### 1. 准备工作

配置脚本前需要完成前置准备工作，包括：**环境安装**、**权重下载及转换**、**数据集准备及处理**，详情可查看对应章节

 <a id="jump4.2"></a>

#### 2. 配置参数

需根据实际情况修改`model.json`和`data.json`中的权重和数据集路径，包括`from_pretrained`、`data_path`、`data_folder`字段

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
    bash examples/opensora1.0/pretrain_opensora1_0.sh
```

**注意**：

- 多机训练需在多个终端同时启动预训练脚本(每个终端的预训练脚本只有NODE_RANK参数不同，其他参数均相同)
- 如果使用多机训练，需要在每个节点准备训练数据和模型权重

---

<a id="jump5"></a>

## 推理

<a id="jump5.1"></a>

#### 1. 准备工作

从Huggingface等网站下载开源模型权重

- [stabilityai/sd-vae-ft-ema](https://huggingface.co/stabilityai/sd-vae-ft-ema)   - vae模型
- [DeepFloyd/t5-v1_1-xxl](https://huggingface.co/DeepFloyd/t5-v1_1-xxl)       -  t5模型
- [hpcai-tech/Open-Sora](https://huggingface.co/hpcai-tech/Open-Sora/resolve/main/OpenSora-v1-HQ-16x512x512.pth)        -  预训练权重

<a id="jump5.2"></a>

#### 2. 配置参数

将上述权重传到json文件中 from_pretrained字段中，prompt字段可以自定义成自己的prompt,目前有两种视频格式选择。

```
examples/opensora1.0/inference_model_16x512x512.json
examples/opensora1.0/inference_model_120x256x256.json
```

<a id="jump5.3"></a>

#### 3. 启动推理

将上面的json，传入下面shell中MM_MODEL字段中，执行下面的推理脚本即可进行推理

```shell
examples/opensora1.0/inference_opensora1_0.sh
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