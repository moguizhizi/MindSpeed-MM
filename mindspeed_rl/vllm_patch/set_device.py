import torch
import os

import vllm
import vllm_ascend
from vllm_ascend.platform import NPUPlatform
from vllm.model_executor import set_random_seed
import vllm_ascend.worker.worker

def get_adjusted_device(device: any) -> torch.device:
    """
    接收一个设备标识（字符串、整数或 torch.device），应用 DEVICES_OFFSET 偏移后，
    返回调整后的 torch.device 对象。

    支持格式：
        - torch.device("npu:0")
        - "npu:0"
        - 0
        - "0"
    """
    # 类型检查和转换
    if isinstance(device, torch.device):
        dev_type = device.type
        if device.index is not None:
            index = int(device.index)
        else:
            index = 0
    elif isinstance(device, (str, int)):
        if isinstance(device, str):
            if ':' in str(device):
                dev_type, dev_index = device.split(':')
                index = int(dev_index)
            else:
                dev_type = device
                index = 0
        else:  # int
            dev_type = 'npu'  # 默认为 npu
            index = device
    else:
        raise TypeError(f"Unsupported device type: {type(device)}")

    # 应用偏移
    offset = int(os.environ.get("DEVICES_OFFSET", 0))
    index -= offset

    # 构造新的 device
    return torch.device(f"{dev_type}:{index}")

def init_device(self) -> None:
    if self.device_config.device.type == "npu":
        self.device = torch.device(f"npu:{self.local_rank}")
        self.device = get_adjusted_device(self.device)
        NPUPlatform.set_device(self.device)
        NPUPlatform.empty_cache()
        self.init_npu_memory = NPUPlatform.mem_get_info()[0]
    else:
        raise RuntimeError(
            f"Not support device type: {self.device_config.device}")
    # Initialize the distributed environment.
    self._init_worker_distributed_environment(self.vllm_config, self.rank,
                                                self.distributed_init_method,
                                                self.local_rank)
    # Set random seed.
    set_random_seed(self.model_config.seed)

def set_device_offset_patch():
    vllm_ascend.worker.worker.NPUWorker.init_device=init_device