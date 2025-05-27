# 安装指导

## 模型开发时推荐使用配套的环境版本

<table border="0">
  <tr>
    <th>软件</th>
    <th>版本</th>
    <th>安装指南</th>
  </tr>
  <tr>
    <td> Python </td>
    <td> 3.8 / 3.10 </td>
  </tr>
  <tr>
    <td> Driver </td>
    <td> AscendHDK 25.0.RC1 </td>
    <td rowspan="2">《<a href="https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/softwareinst/instg/instg_0003.html?Mode=PmIns&OS=Ubuntu&Software=cannToolKit">驱动固件安装指南</a> 》</td>
  </tr>
  <tr>
    <td> Firmware </td>
    <td> AscendHDK 25.0.RC1 </td>
  </tr>
  <tr>
    <td> CANN </td>
    <td> CANN 8.1.RC1 </td>
    <td>《<a href="https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/softwareinst/instg/instg_0003.html">CANN 软件安装指南</a> 》</td>
  </tr>
  <tr>
    <td> Torch </td>
    <td> 2.1.0 </td>
    <td rowspan="2">《<a href="https://www.hiascend.com/document/detail/zh/Pytorch/700/configandinstg/instg/insg_0004.html">Ascend Extension for PyTorch 配置与安装</a> 》</td>
  </tr>
  <tr>
    <td> Torch_npu </td>
    <td> release v7.0.0 </td>
  </tr>
</table>

## 驱动固件安装

下载[驱动固件](https://www.hiascend.com/hardware/firmware-drivers/community?product=4&model=26&cann=8.0.RC3.beta1&driver=1.0.27.alpha)，请根据系统和硬件产品型号选择对应版本的`driver`和`firmware`。参考[安装NPU驱动固件](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/softwareinst/instg/instg_0003.html?Mode=PmIns&OS=Ubuntu&Software=cannToolKit)或执行以下命令安装：

```shell
bash Ascend-hdk-*-npu-firmware_*.run --full
bash Ascend-hdk-*-npu-driver_*.run --full --force
```

## CANN安装

下载[CANN](https://www.hiascend.com/developer/download/community/result?module=cann)，请根据根据系统选择`aarch64`或`x86_64`对应版本的`cann-toolkit`、`cann-kernel`和`cann-nnal`。参考[CANN安装](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/softwareinst/instg/instg_0003.html)或执行以下命令安装：

```shell
# 因为版本迭代，包名存在出入，根据实际修改
bash Ascend-cann-toolkit_8.1.RC1_linux-aarch64.run --install
bash Ascend-cann-kernels-*_8.1.RC1_linux-aarch64.run --install
# 设置环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

## PTA安装

准备[torch_npu](https://www.hiascend.com/developer/download/community/result?module=pt)和[apex](https://gitee.com/ascend/apex)，参考[Ascend Extension for PyTorch 配置与安装](https://www.hiascend.com/document/detail/zh/Pytorch/700/configandinstg/instg/insg_0004.html)或执行以下命令安装：

安装torch和torch_npu,以下为python 3.8版本

```shell
conda create -n test python=3.8
conda activate test
pip install torch-2.1.0-cp38-cp38m-manylinux2014_aarch64.whl 
pip install torch_npu-2.1.0*-cp38-cp38m-linux_aarch64.whl
```

安装torch和torch_npu,以下为python 3.10版本

```shell
conda create -n test python=3.10
conda activate test
pip install torch-2.1.0-cp310-cp310m-manylinux2014_aarch64.whl 
pip install torch_npu-2.1.0*-cp310-cp310m-linux_aarch64.whl
```

安装apex

```shell
# apex for Ascend 构建参考 https://gitee.com/ascend/apex
pip3 install --upgrade apex-0.1+ascend-{version}.whl # version为python版本和cpu架构
```
