## 安全声明

### 系统安全加固

1. 建议用户在系统中配置开启ASLR（级别2），又称**全随机地址空间布局随机化**，可参考以下方式进行配置：

    ```
    echo 2 > /proc/sys/kernel/randomize_va_space
    ```

### 运行用户建议

出于安全性及权限最小化角度考虑，不建议使用root等管理员类型账户使用MindSpeed-MM。

### 文件权限控制

1. 建议用户在主机（包括宿主机）及容器中设置运行系统umask值为0027及以上，保障新增文件夹默认最高权限为750，新增文件默认最高权限为640。
2. 建议用户对个人数据、商业资产、源文件、训练过程中保存的各类文件等敏感内容做好权限管控。涉及场景如MindSpeed-MM安装目录权限管控、多用户使用共享数据集权限管控，管控权限可参考表1进行设置。
3. MindSpeed-MM在数据预处理中会生成训练数据，在训练过程会生成权重文件，文件权限默认640，用户可根据实际需求对生成文件权限进行进阶管控。

**表1 文件（夹）各场景权限管控推荐最大值**

| 类型          | linux权限参考最大值 |
| --------------- | --------------------|
| 用户主目录                          |    750（rwxr-x---）                |
| 程序文件（含脚本文件、库文件等）      |    550（r-xr-x---）                |
| 程序文件目录                        |    550（r-xr-x---）                |
| 配置文件                            |    640（rw-r-----）                |
| 配置文件目录                        |    750（rwxr-x---）                |
| 日志文件（记录完毕或者已经归档）      |    440（r--r-----）                |
| 日志文件（正在记录）                 |    640（rw-r-----）                |
| 日志文件记录                        |    750（rwxr-x---）                |
| Debug文件                          |    640（rw-r-----）                |
| Debug文件目录                      |    750 (rwxr-x---)                 |
| 临时文件目录                       |     750（rwxr-x---）                |
| 维护升级文件目录                    |    770（rwxrwx---）                |
| 业务数据文件                       |     640（rw-r-----）                |
| 业务数据文件目录                   |     750（rwxr-x---）                |
| 密钥组件、私钥、证书、密文文件目录   |     700（rwx------）                |
| 密钥组件、私钥、证书、加密密文      |     600（rw-------）                |
| 加解密接口、加解密脚本             |     500（r-x------）                |

### 数据安全声明

1. MindSpeed-MM会加载和保存模型文件，其中部分模型文件使用了风险模块pickle，可能存在数据风险。

### 运行安全声明

1. 建议用户结合运行资源状况编写对应训练脚本。若训练脚本与资源状况不匹配，如数据集加载内存大小超出内存容量限制、训练脚本在本地生成数据超过磁盘空间大小等情况，可能引发错误并导致进程意外退出。
2. MindSpeed-MM内部用到了pytorch和torch_npu,可能会因为版本不匹配导致运行错误，具体可参考pytorch及torch_npu[安全声明](https://gitee.com/ascend/pytorch#%E5%AE%89%E5%85%A8%E5%A3%B0%E6%98%8E)。

## 公开接口声明

MindSpeed-MM 暂时未发布wheel包，无正式对外公开接口，所有功能均通过shell脚本调用。6个入口脚本分别为:

- [pretrain_sora](https://gitee.com/ascend/MindSpeed-MM/blob/master/pretrain_sora.py)
- [pretrain_llava](https://gitee.com/ascend/MindSpeed-MM/blob/master/pretrain_llava.py)
- [pretrain_internvl](https://gitee.com/ascend/MindSpeed-MM/blob/master/pretrain_internvl.py)
- [pretrain_whisper](https://gitee.com/ascend/MindSpeed-MM/blob/master/pretrain_whisper.py)
- [inference_sora](https://gitee.com/ascend/MindSpeed-MM/blob/master/inference_sora.py)
- [inference_vlm](https://gitee.com/ascend/MindSpeed-MM/blob/master/inference_vlm.py)

## 通信安全加固

[通信安全加固说明](https://gitee.com/ascend/pytorch/blob/master/SECURITYNOTE.md#%E9%80%9A%E4%BF%A1%E5%AE%89%E5%85%A8%E5%8A%A0%E5%9B%BA
)

## 通信矩阵

[通信矩阵说明](https://gitee.com/ascend/pytorch/blob/master/SECURITYNOTE.md#%E9%80%9A%E4%BF%A1%E7%9F%A9%E9%98%B5%E4%BF%A1%E6%81%AF)

## 特殊场景

| 场景                                                  | 使用方法                                                     | 端口           | 可能的风险                                                   |
| ----------------------------------------------------- | ------------------------------------------------------------ | -------------- | ------------------------------------------------------------ |
| 用户下载并使用HuggingFace的开源数据集                 | 调用`load_dataset`函数，并填写目标开源数据集路径             | 随机端口       | 数据集可能包含敏感或不合法内容，导致合规问题。数据集中可能存在质量问题，如标签错误或数据偏差，影响数据预处理。 |
| 使用`from_pretrained`信任特定代码，使用相关模型的实现 | 调用`from_pretrained`函数，设置`trust_remote_code=True`      | 随机端口       | 如果 trust_remote_code=True，下载的代码可能包含恶意逻辑或后门，威胁系统安全。但同时已设置`local_files_only=True`，程序仅会运行本地的文件来规避风险 |
| 使用MindSpeed-MM进行训练任务时，新增端口32个          | MindSpeed-MM 调用 Megatron 原生函数 `mpu.initialize_model_parallel` 来初始化模型并行组，并通过使用 PyTorch 分布式训练相关的 API 来启动任意任务。 | [1024,65520]内 | 网络配置错误可能引发端口冲突或连接问题，影响训练效率。       |

### 公网地址声明

代码涉及公网地址参考 [public_address_statement.md](https://gitee.com/ascend/MindSpeed-MM/blob/master/docs/public_address_statement.md)
