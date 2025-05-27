import re
from abc import ABC, abstractmethod
from typing import Callable, Iterable, Dict, List, Tuple

import torch
from torch import Tensor

STATE_DICT_T = Dict[str, torch.Tensor]
TP_PATTERN_T = Dict[str, Callable[[int, int, torch.Tensor], torch.Tensor]]
DIGIT_FMT = r'(\d+)'


def interleaved_qkv_to_concated(megatron_qkv: Tensor, num_key_value_heads: int, split_size: List[int]) -> Tensor:
    """mindspeed-mm中的qkv排布和vllm中qkv的排布不同，因此需要重新转换，qwen2vl的转换逻辑如下：
    qkv = [nq1,k1,v1,  nq2,k2,v2,  nq3,k3,v3,  nq4,k4,v4]
    new_qkv = [nq1,nq2,nq3,nq4,k1,k2,k3,k4,v1,v2,v3,v4]
    """
    groups_qkv = [torch.split(group, split_size) for group in torch.chunk(megatron_qkv, num_key_value_heads)]
    return torch.cat([head for group_head in zip(*groups_qkv) for head in group_head])


def concated_qkv_to_interleaved(qkv: Tensor, num_key_value_heads: int, split_size: List[int]) -> Tensor:
    """
    qkv = [nq1,nq2,nq3,nq4,k1,k2,k3,k4]
    new_qkv =  [nq1,k1,v1,  nq2,k2,v2,  nq3,k3,v3,  nq4,k4,v4]
    """
    qkv_split = torch.split(qkv, split_size)
    qkv_chunk_group = [torch.chunk(i, num_key_value_heads) for i in qkv_split]
    return torch.cat([i for j in zip(*qkv_chunk_group) for i in j])


def merge_to_interleaved_qkv(q: Tensor, k: Tensor, v: Tensor, group: int) -> Tensor:
    """原顺序: q=[nq1,nq2,nq3,...], k=[k1,k2,k3,...], v=[v1,v2,v3,...]
    转换后顺序: qkv = [nq1,k1,v1,nq1,k2,v2,nq3,k3,v3,...]
    这里nq1，group query attention时，多个q共享一个k/v， k1/v1即一个key head对应的head_dim维度
    """
    qkv_chunks = [torch.chunk(x, chunks=group, dim=0) for x in [q, k, v]]
    # zip(*[])转换迭代顺序，qkv_group即一组qkv，包含一个k/v头以及对应的多个q，[nq1,k1,v1]
    return torch.concat([head for qkv_group in zip(*qkv_chunks) for head in qkv_group], dim=0)


def get_layer_num(pattern: str, names: Iterable[str]) -> int:
    """在names中找到pattern正则匹配到的数数字的最大值"""
    return max(int(re.findall(pattern, k)[0]) for k in names if re.match(pattern, k)) + 1


class Operator(ABC):
    @abstractmethod
    def handle(self, weights: STATE_DICT_T):
        """原地修改weights字典"""
        pass


class ZeroWeightOp(Operator):
    """
    权重置零操作，将权重值赋值为零或向权重字典添加全零权重
    权重转换需要把q,k,v权重合并成一个完整的qkv大矩阵。
    而由于在Qwen2.5-Omni模型的音频模型中，k没有bias，所以需要将k的bias以全零权重的形式添加到权重字典
    """

    def __init__(self, name: str, shape: List[int], layers: int):
        """
        初始化权重信息

        参数:
        - name (str): 权重的名称。
        - shape (List[int]): 权重的的形状，以整数列表形式表示。
        - layers (int): 模型的层数。
        """
        self.name = name
        self.shape = shape
        self.layers = layers

    def handle(self, weights: STATE_DICT_T):
        """向weights字典中添加权重或修改权重为0"""
        for layer_num in range(self.layers):
            tensor = torch.zeros(self.shape)
            name = self.name.replace(DIGIT_FMT, str(layer_num))
            weights[name] = tensor


class RenameOp(Operator):
    def __init__(self, patterns: Tuple[Tuple[str, str], ...]):
        self.patterns = patterns

    def handle(self, weights: STATE_DICT_T):
        mapping = {}
        for weight_name in weights:
            new_name = weight_name
            for pattern, replacement in self.patterns:
                new_name = re.sub(pattern, replacement, new_name)
            if new_name != weight_name:
                mapping[weight_name] = new_name
        for old_name, new_name in mapping.items():
            weights[new_name] = weights.pop(old_name)


class MergeOp(Operator):
    def __init__(self, raw_names: List[str], new_name: str):
        self.raw_names = raw_names
        self.new_name = new_name

    @abstractmethod
    def merge(self, tensors: List[Tensor]) -> Tensor:
        pass

    def handle(self, weights: STATE_DICT_T):
        layer_num = get_layer_num(self.raw_names[0], weights.keys())
        for num in range(layer_num):
            merged_tensor = self.merge([weights.pop(name.replace(DIGIT_FMT, str(num))) for name in self.raw_names])
            weights[self.new_name.replace(DIGIT_FMT, str(num))] = merged_tensor


class UpGateMergeOp(MergeOp):
    """GLU门控单元中权重合并逻辑，从transformers的gate up两个小矩阵到megatron的mlp中linear_fc1中的一个大矩阵是直接concat的"""

    def merge(self, tensors: List[Tensor]) -> Tensor:
        return torch.concat(tensors, dim=0)


class QKVMergeOp(MergeOp):
    """qkv中权重合并逻辑，从transformers的q1q2/k1k2/v1v2三个小矩阵到megatron的linear_qkv一个大矩阵是q1k1v1q2k2v2交织拼接的"""

    def __init__(self, raw_names: List[str], new_name: str, group: int):
        self.group = group
        super().__init__(raw_names, new_name)

    def merge(self, tensors: List[Tensor]) -> Tensor:
        return merge_to_interleaved_qkv(*tensors, group=self.group)


class RelocateOp(Operator):
    """对于vit，transformers和megatron都是一个大矩阵，但transformer中的qkv是q1q2k1k2v1v2直接拼接，而megatron中是q1k1v1q2k2v2交织拼接"""

    def __init__(self, name: str, group: int, split_size: List[int]):
        self.name = name
        self.group = group
        self.split_size = split_size

    def handle(self, weights: STATE_DICT_T):
        layer_num = get_layer_num(self.name, weights.keys())
        for num in range(layer_num):
            name = self.name.replace(DIGIT_FMT, str(num))
            weights[name] = concated_qkv_to_interleaved(weights.get(name), self.group, self.split_size)


class TieOp(Operator):
    """对于tie word embedding，其output layer是直接复用的word_embedding的权重"""

    def __init__(self, raw_name: str, new_name: str):
        self.raw_name = raw_name
        self.new_name = new_name

    def handle(self, weights: STATE_DICT_T):
        weights[self.new_name] = weights.get(self.raw_name)


def tp_split_col_weight(tp_size: int, tp_rank: int, value: Tensor) -> Tensor:
    size_per_tp = value.shape[1] // tp_size
    return value[:, tp_rank * size_per_tp:(tp_rank + 1) * size_per_tp]


def tp_split_row_bias(tp_size: int, tp_rank: int, value: Tensor) -> Tensor:
    size_per_tp = value.shape[0] // tp_size
    return value[tp_rank * size_per_tp:(tp_rank + 1) * size_per_tp]


def tp_split_row_weight(tp_size: int, tp_rank: int, value: Tensor) -> Tensor:
    size_per_tp = value.shape[0] // tp_size
    return value[tp_rank * size_per_tp:(tp_rank + 1) * size_per_tp, :]


def tp_split_glu_weight(tp_size: int, tp_rank: int, value: Tensor) -> Tensor:
    size_per_tp = value.shape[0] // tp_size // 2
    values = torch.chunk(value, 2, dim=0)
    gate_tp = values[0][tp_rank * size_per_tp:(tp_rank + 1) * size_per_tp, :]
    up_tp = values[1][tp_rank * size_per_tp:(tp_rank + 1) * size_per_tp, :]
    tensor = torch.cat((gate_tp, up_tp), dim=0)
    return tensor


def tp_split_glu_bias(tp_size: int, tp_rank: int, value: Tensor) -> Tensor:
    size_per_tp = value.shape[0] // tp_size // 2
    values = torch.chunk(value, 2, dim=0)
    gate_tp = values[0][tp_rank * size_per_tp:(tp_rank + 1) * size_per_tp]
    up_tp = values[1][tp_rank * size_per_tp:(tp_rank + 1) * size_per_tp]
    tensor = torch.cat((gate_tp, up_tp), dim=0)
    return tensor


def create_qwen2vl_ops(vit_embed_dim: int, vit_num_heads: int, llm_num_query_groups: int) -> List[Operator]:
    """qwen2vl权重转换逻辑"""
    ops = [
        UpGateMergeOp(raw_names=["model.layers.(\d+).mlp.gate_proj.weight", "model.layers.(\d+).mlp.up_proj.weight"],
                      new_name="text_decoder.decoder.layers.(\d+).mlp.linear_fc1.weight"),
        QKVMergeOp(raw_names=["model.layers.(\d+).self_attn.q_proj.weight",
                              "model.layers.(\d+).self_attn.k_proj.weight",
                              "model.layers.(\d+).self_attn.v_proj.weight"],
                   new_name="text_decoder.decoder.layers.(\d+).self_attention.linear_qkv.weight",
                   group=llm_num_query_groups),
        QKVMergeOp(raw_names=["model.layers.(\d+).self_attn.q_proj.bias",
                              "model.layers.(\d+).self_attn.k_proj.bias",
                              "model.layers.(\d+).self_attn.v_proj.bias"],
                   new_name="text_decoder.decoder.layers.(\d+).self_attention.linear_qkv.bias",
                   group=llm_num_query_groups),
        RenameOp(
            (
                # 定义多个正则替换规则（按处理顺序排列）
                # 处理 visual.blocks.{n} 路径
                (r'^visual\.blocks\.(\d+)\.', r'image_encoder.encoder.blocks.layers.\1.'),
                (r'\.attn\.proj\.', '.self_attention.linear_proj.'),
                (r'\.attn\.qkv\.', '.self_attention.linear_qkv.'),
                (r'\.mlp\.fc1\.', '.mlp.linear_fc1.'),
                (r'\.mlp\.fc2\.', '.mlp.linear_fc2.'),
                (r'\.norm1\.', '.input_layernorm.'),
                (r'\.norm2\.', '.pre_mlp_layernorm.'),
                # 处理 model.layers.{n} 路径
                (r'^model\.layers\.(\d+)\.', r'text_decoder.decoder.layers.\1.'),
                (r'\.mlp\.down_proj\.', '.mlp.linear_fc2.'),
                (r'\.post_attention_layernorm\.', '.pre_mlp_layernorm.'),
                (r'\.self_attn\.o_proj\.', '.self_attention.linear_proj.'),
                # 处理 visual.merger 相关
                (r'^visual\.merger\.ln_q', 'image_encoder.projector.layernorm'),
                (r'^visual\.merger\.mlp\.0', 'image_encoder.projector.encoder.linear_fc1'),
                (r'^visual\.merger\.mlp\.2', 'image_encoder.projector.encoder.linear_fc2'),
                # 其他固定映射
                (r'^visual\.patch_embed\.proj', 'image_encoder.encoder.patch_embed.proj'),
                (r'^model\.embed_tokens', 'text_decoder.embedding.word_embeddings'),
                (r'^model\.norm', 'text_decoder.decoder.final_layernorm'),
                (r'^lm_head', 'text_decoder.output_layer')
            )
        ),
        RelocateOp(name="image_encoder.encoder.blocks.layers.(\d+).self_attention.linear_qkv.weight",
                   group=vit_num_heads,
                   split_size=[vit_embed_dim] * 3,  # vit的qkv不是gqa，所以切分的三份是相同的
                   ),
        RelocateOp(name="image_encoder.encoder.blocks.layers.(\d+).self_attention.linear_qkv.bias",
                   group=vit_num_heads,
                   split_size=[vit_embed_dim] * 3,  # vit的qkv不是gqa，所以切分的三份是相同的
                   ),
    ]
    return ops


def create_qwen2_5_vl_ops(vit_embed_dim: int, vit_num_heads: int, llm_num_query_groups: int) -> List[Operator]:
    """qwen2.5vl在qwen2vl的基础上vit的mlp变成了glu模式、需要增加合并处理逻辑"""
    ops = [
              UpGateMergeOp(
                  raw_names=["visual.blocks.(\d+).mlp.gate_proj.weight", "visual.blocks.(\d+).mlp.up_proj.weight"],
                  new_name="image_encoder.encoder.blocks.layers.(\d+).mlp.linear_fc1.weight"),
              UpGateMergeOp(
                  raw_names=["visual.blocks.(\d+).mlp.gate_proj.bias", "visual.blocks.(\d+).mlp.up_proj.bias"],
                  new_name="image_encoder.encoder.blocks.layers.(\d+).mlp.linear_fc1.bias")
          ] + create_qwen2vl_ops(vit_embed_dim, vit_num_heads, llm_num_query_groups)
    return ops


def create_qwen2_5_omni_ops(vit_num_heads: int, llm_num_query_groups: int, audio_num_heads: int, audio_d_model: int,
                            audio_encoder_layers: int) -> List[Operator]:
    """qwen2.5-omni 权重转换逻辑"""
    ops = [
        # 音频模型中，k没有bias，所以需要将k的bias以全零权重的形式添加到权重字典，以便进行后续的qkv拼接
        ZeroWeightOp(
            name="thinker.audio_tower.layers.(\d+).self_attn.k_proj.bias",
            shape=[audio_d_model],
            layers=audio_encoder_layers
        ),
        UpGateMergeOp(
            raw_names=["thinker.visual.blocks.(\d+).mlp.gate_proj.weight",
                       "thinker.visual.blocks.(\d+).mlp.up_proj.weight"],
            new_name="image_encoder.encoder.blocks.layers.(\d+).mlp.linear_fc1.weight"),
        UpGateMergeOp(
            raw_names=["thinker.visual.blocks.(\d+).mlp.gate_proj.bias",
                       "thinker.visual.blocks.(\d+).mlp.up_proj.bias"],
            new_name="image_encoder.encoder.blocks.layers.(\d+).mlp.linear_fc1.bias"),
        UpGateMergeOp(raw_names=["thinker.model.layers.(\d+).mlp.gate_proj.weight",
                                 "thinker.model.layers.(\d+).mlp.up_proj.weight"],
                      new_name="text_decoder.decoder.layers.(\d+).mlp.linear_fc1.weight"),
        QKVMergeOp(raw_names=["thinker.model.layers.(\d+).self_attn.q_proj.weight",
                              "thinker.model.layers.(\d+).self_attn.k_proj.weight",
                              "thinker.model.layers.(\d+).self_attn.v_proj.weight"],
                   new_name="text_decoder.decoder.layers.(\d+).self_attention.linear_qkv.weight",
                   group=llm_num_query_groups),
        QKVMergeOp(raw_names=["thinker.model.layers.(\d+).self_attn.q_proj.bias",
                              "thinker.model.layers.(\d+).self_attn.k_proj.bias",
                              "thinker.model.layers.(\d+).self_attn.v_proj.bias"],
                   new_name="text_decoder.decoder.layers.(\d+).self_attention.linear_qkv.bias",
                   group=llm_num_query_groups),
        QKVMergeOp(raw_names=["thinker.visual.blocks.(\d+).attn.q.weight",
                              "thinker.visual.blocks.(\d+).attn.k.weight",
                              "thinker.visual.blocks.(\d+).attn.v.weight"],
                   new_name="image_encoder.encoder.blocks.layers.(\d+).self_attention.linear_qkv.weight",
                   group=vit_num_heads),
        QKVMergeOp(raw_names=["thinker.visual.blocks.(\d+).attn.q.bias",
                              "thinker.visual.blocks.(\d+).attn.k.bias",
                              "thinker.visual.blocks.(\d+).attn.v.bias"],
                   new_name="image_encoder.encoder.blocks.layers.(\d+).self_attention.linear_qkv.bias",
                   group=vit_num_heads),
        QKVMergeOp(raw_names=["thinker.audio_tower.layers.(\d+).self_attn.q_proj.weight",
                              "thinker.audio_tower.layers.(\d+).self_attn.k_proj.weight",
                              "thinker.audio_tower.layers.(\d+).self_attn.v_proj.weight"],
                   new_name="audio_encoder.encoder.blocks.layers.(\d+).self_attention.linear_qkv.weight",
                   group=audio_num_heads),
        QKVMergeOp(raw_names=["thinker.audio_tower.layers.(\d+).self_attn.q_proj.bias",
                              "thinker.audio_tower.layers.(\d+).self_attn.k_proj.bias",
                              "thinker.audio_tower.layers.(\d+).self_attn.v_proj.bias"],
                   new_name="audio_encoder.encoder.blocks.layers.(\d+).self_attention.linear_qkv.bias",
                   group=audio_num_heads),
        RenameOp(
            (
                # 定义多个正则替换规则（按处理顺序排列）
                # 处理 visual.blocks.{n} 路径
                (r'^thinker.visual\.blocks\.(\d+)\.', r'image_encoder.encoder.blocks.layers.\1.'),
                (r'\.attn\.proj\.', '.self_attention.linear_proj.'),
                (r'\.attn\.qkv\.', '.self_attention.linear_qkv.'),
                (r'\.norm1\.', '.input_layernorm.'),
                (r'\.norm2\.', '.pre_mlp_layernorm.'),
                # 处理 model.layers.{n} 路径
                (r'^thinker.model\.layers\.(\d+)\.', r'text_decoder.decoder.layers.\1.'),
                (r'\.mlp\.down_proj\.', '.mlp.linear_fc2.'),
                (r'\.post_attention_layernorm\.', '.pre_mlp_layernorm.'),
                (r'\.self_attn\.o_proj\.', '.self_attention.linear_proj.'),
                # 处理 visual.merger 相关
                (r'^thinker.visual\.merger\.ln_q', 'image_encoder.projector.layernorm'),
                (r'^thinker.visual\.merger\.mlp\.0', 'image_encoder.projector.encoder.linear_fc1'),
                (r'^thinker.visual\.merger\.mlp\.2', 'image_encoder.projector.encoder.linear_fc2'),
                # audio_tower相关
                (r'^thinker.audio_tower\.layers\.(\d+)\.', r'audio_encoder.encoder.blocks.layers.\1.'),
                (r'\.self_attn\.out_proj\.', '.self_attention.linear_proj.'),
                (r'\.fc1\.', '.mlp.linear_fc1.'),
                (r'\.fc2\.', '.mlp.linear_fc2.'),
                (r'\.self_attn_layer_norm\.', '.input_layernorm.'),
                (r'\.final_layer_norm\.', '.pre_mlp_layernorm.'),
                (r'^thinker.audio_tower\.', r'audio_encoder.encoder.'),
                # 其他固定映射
                (r'^thinker.visual\.patch_embed\.proj', 'image_encoder.encoder.patch_embed.proj'),
                (r'^thinker.model\.embed_tokens', 'text_decoder.embedding.word_embeddings'),
                (r'^thinker.model\.norm', 'text_decoder.decoder.final_layernorm'),
                (r'^thinker.lm_head', 'text_decoder.output_layer')
            )
        ),
    ]
    return ops


qwen2vl_tp_patterns = {
    "text_decoder.output_layer.weight": tp_split_row_weight,
    "text_decoder.embedding.word_embeddings.weight": tp_split_row_weight,
    'text_decoder.decoder.layers.(\d+).mlp.linear_fc1.weight': tp_split_glu_weight,
    'text_decoder.decoder.layers.(\d+).mlp.linear_fc2.weight': tp_split_col_weight,
    'text_decoder.decoder.layers.(\d+).self_attention.linear_qkv.weight': tp_split_row_weight,
    'text_decoder.decoder.layers.(\d+).self_attention.linear_qkv.bias': tp_split_row_bias,
    'text_decoder.decoder.layers.(\d+).self_attention.linear_proj.weight': tp_split_col_weight,
    "image_encoder.encoder.blocks.layers.(\d+).self_attention.linear_proj.weight": tp_split_col_weight,
    "image_encoder.encoder.blocks.layers.(\d+).self_attention.linear_qkv.bias": tp_split_row_bias,
    "image_encoder.encoder.blocks.layers.(\d+).self_attention.linear_qkv.weight": tp_split_row_bias,
    "image_encoder.encoder.blocks.layers.(\d+).mlp.linear_fc1.bias": tp_split_row_bias,
    "image_encoder.encoder.blocks.layers.(\d+).mlp.linear_fc1.weight": tp_split_row_weight,
    "image_encoder.encoder.blocks.layers.(\d+).mlp.linear_fc2.weight": tp_split_col_weight,
    "image_encoder.projector.encoder.linear_fc1.bias": tp_split_row_bias,
    "image_encoder.projector.encoder.linear_fc1.weight": tp_split_row_weight,
    "image_encoder.projector.encoder.linear_fc2.weight": tp_split_col_weight
}
#  qwen2.5vl的tp切分在qwen2vl的tp切分基础上，修改了vit中mlp的tp切分逻辑，适应glu结构
qwen2_5_vl_tp_patterns = {**qwen2vl_tp_patterns,
                          **{"image_encoder.encoder.blocks.layers.(\d+).mlp.linear_fc1.bias": tp_split_glu_bias,
                             "image_encoder.encoder.blocks.layers.(\d+).mlp.linear_fc1.weight": tp_split_glu_weight}
                          }

deepseekvl_tp_patterns = {
    "text_decoder.output_layer.weight": tp_split_row_weight,
    "text_decoder.embedding.word_embeddings.weight": tp_split_row_weight,
    'text_decoder.decoder.layers.(\d+).mlp.linear_fc1.weight': tp_split_glu_weight,
    'text_decoder.decoder.layers.(\d+).mlp.linear_fc2.weight': tp_split_col_weight,
    'text_decoder.decoder.layers.(\d+).self_attention.linear_qb.weight': tp_split_row_weight,
    'text_decoder.decoder.layers.(\d+).self_attention.linear_kvb.weight': tp_split_row_weight,
    'text_decoder.decoder.layers.(\d+).self_attention.linear_kvb.bias': tp_split_row_bias,
    'text_decoder.decoder.layers.(\d+).self_attention.linear_proj.weight': tp_split_col_weight,
    'text_decoder.decoder.layers.(\d+).mlp.experts.local_experts.(\d+).linear_fc1.weight': tp_split_glu_weight,
    "text_decoder.decoder.layers.(\d+).mlp.experts.local_experts.(\d+).linear_fc2.weight": tp_split_col_weight,
    "text_decoder.decoder.layers.(\d+).mlp.shared_experts.linear_fc1.weight": tp_split_glu_weight,
    "text_decoder.decoder.layers.(\d+).mlp.shared_experts.linear_fc2.weight": tp_split_col_weight
}
