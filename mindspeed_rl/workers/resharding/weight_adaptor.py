from abc import ABC, abstractmethod
import re
import torch


class BaseWeightAdaptor(ABC):
    def __init__(self):
        """
        Base class for weight adaptors.
        A weight adaptor provide a set of tools to transfer from training weight to inference weight.
        Currently, we support MegatronVLLMWeightAdaptor only.
        Args:
        """
        pass

    @abstractmethod
    def replace_name_i2t(self, inference_name):
        """
        transfer inference weight name to training weight name
        """
        pass

    @abstractmethod
    def convert_weight_t2i(self, weight_name, weight):
        """
        Transfer weight format to inference engine's format.
        """
        pass

    @abstractmethod
    def get_weight_buffer_meta(self, param_dict, valid_names=None):
        """
        Given inference param_dict, build a weight buffer meta data in train weight style.
        Needs model specific coding when multiple inference params correspond to one training param,
         or one inference param corresponds to multiple training params.
        Return a dictionary containing name to a shape and dtype.
        """
        pass


class MegatronVLLMWeightAdaptor(BaseWeightAdaptor):
    def __init__(self, model_config):
        super(MegatronVLLMWeightAdaptor, self).__init__()
        self.model_config = model_config
        self.params_mapping = [
            # (megatron core gpt model name, vllm model name)
            ("embedding.word_embeddings", "model.embed_tokens"),
            ("self_attention.linear_qkv", "self_attn.qkv_proj"),
            ("self_attention.linear_proj", "self_attn.o_proj"),
            ("input_layernorm", "input_layernorm"),
            ("pre_mlp_layernorm", "post_attention_layernorm"),
            ("mlp.linear_fc1", "mlp.gate_up_proj"),
            ("mlp.linear_fc2", "mlp.down_proj"),
            ("decoder.final_layernorm", "model.norm"),
            ("output_layer", "lm_head"),
        ]

    def replace_name_i2t(self, inference_name):
        """
        transfer inference weight name to training weight name
        """
        for m_name, v_name in self.params_mapping:
            if v_name not in inference_name:
                continue
            if "layers" in inference_name:  # deal with decoder layers
                inference_name = inference_name.replace("model", "decoder")
                vllm_name_list = inference_name.split(".")
                if "layer_norm_weight" in vllm_name_list or "layer_norm_bias" in vllm_name_list:
                    param_name_list = vllm_name_list[:3]
                    param_name_list.append(m_name)
                    param_name = ".".join(param_name_list)
                else:
                    param_name_list = vllm_name_list[:3]
                    weight_or_bias = vllm_name_list[-1]
                    param_name_list.append(m_name)
                    if weight_or_bias in ['weight', 'bias']:
                        param_name_list.append(weight_or_bias)
                    param_name = ".".join(param_name_list)
                return param_name
            else:
                param_name = inference_name.replace(v_name, m_name)
                return param_name

    # Identity operation here, can be rewritten model by model.
    def _transfer_loaded_weight(self, loaded_weight, name, infer_tp_size):
        return loaded_weight

    def convert_weight_t2i(self, actor_weights, vllm_model, **kargs):
        """
        Transfer weight format to inference engine's format, and load weight to inference engine.
        This will be implemented in the next version.
        """
        pass


    def get_weight_buffer_meta(self, model, valid_names=None):
        weight_buffer_meta = {}
        for name, param in sorted(model.named_parameters()):
            if valid_names and name not in valid_names:
                continue
            else:
                weight_buffer_meta[name] = {'shape': param.shape, 'dtype': param.dtype}
        return weight_buffer_meta

    @staticmethod
    def global2local_layer(name, num_layer_list):
        """
        Transform the model name in each model_chunk in global space to local space
        """
        layer_name = 'layers'

        if layer_name in name:  # belong to an intermediate layer
            split_name = name.split('.')

            # find the num next to split_name
            for layer_num_idx, name in enumerate(split_name, start=1):
                if name == layer_name:
                    break

            # check the name
            if len(split_name) < layer_num_idx + 1 or not split_name[layer_num_idx].isdigit():
                raise ValueError(f'split_name = {split_name}')

            # increment layer_num_idx by layer_offset
            global_idx = int(split_name[layer_num_idx])
            for layers_in_pp in num_layer_list:
                global_idx -= layers_in_pp
                if global_idx < 0:
                    local_index = global_idx + layers_in_pp
                    break

            split_name[layer_num_idx] = str(local_index)
            name = '.'.join(split_name)  # weight name in inference_tp_model

        return name

    @staticmethod
    def get_weight_names_per_pp(layer_list, vllm_names):

        end_layer = sum(layer_list) - 1

        def get_weight_names_in_range(layer_range, names: list, layer_name='layers') -> list:
            """
            Extract weights in a given range and also include the weights before and after the range as needed.
            """
            start, end = layer_range
            last_layer_index = end_layer
            names_in_range = []

            # add names before decoder layers
            if start == 0:
                for name in names:
                    if layer_name not in name:
                        names_in_range.append(name)
                    else:
                        break

            for name in names:
                # Extract layer number from weight
                match = re.match(r'.*\.layers\.(\d+)', name)
                if match:
                    layer_num = int(match.group(1))
                    if start <= layer_num <= end:
                        names_in_range.append(name)

            # add names after decode layers
            if end == last_layer_index:
                for name in reversed(names):
                    if layer_name not in name:
                        names_in_range.append(name)
                    else:
                        break
            return names_in_range

        pp_layers_range = []
        start_layer = 0
        for layers_in_pp_rank in layer_list:
            pp_layers_range.append((start_layer, start_layer + layers_in_pp_rank - 1))
            start_layer += layers_in_pp_rank
        weight_names_per_pp = [get_weight_names_in_range(layer_range, vllm_names) for layer_range in pp_layers_range]
        return weight_names_per_pp


class DeepSeekMVWeightAdaptor(MegatronVLLMWeightAdaptor):
    """
    Megatron-vLLM WeightAdaptor for DeepSeek model architectures.
    """
    def __init__(self, model_config):
        super(DeepSeekMVWeightAdaptor, self).__init__(model_config)
        self.params_mapping = [
            # (megatron core gpt model name, vllm model name)
            ("embedding.word_embeddings", "model.embed_tokens"),
            ("self_attention.linear_qkv", "self_attn.qkv_proj"),  # q_a_proj, kv_a_proj_with_mqa
            ("self_attention.linear_proj", "self_attn.o_proj"),
            ("input_layernorm", "input_layernorm"),
            ("pre_mlp_layernorm", "post_attention_layernorm"),
            ("mlp.linear_fc1", "mlp.gate_up_proj"),
            ("mlp.linear_fc2", "mlp.down_proj"),
            ("decoder.final_layernorm", "model.norm"),
            ("output_layer", "lm_head"),
            ("self_attention.linear_qb", "self_attn.q_b_proj"),
            ("self_attention.linear_kvb", "self_attn.kv_b_proj"),
            ("mlp.router.expert_bias", "mlp.gate.e_score_correction_bias"),
            ('mlp.router', 'mlp.gate'),  # weight, expert_bias
            ("mlp.shared_experts.linear_fc1", "mlp.shared_experts.gate_up_proj"),
            ("mlp.shared_experts.linear_fc2", "mlp.shared_experts.down_proj"),
            ("mlp.experts.weight1", "mlp.experts.w13_weight"),
            ("mlp.experts.weight2", "mlp.experts.w2_weight"),
            ("self_attention.q_layernorm", "self_attn.q_a_layernorm"),
            ("self_attention.k_layernorm", "self_attn.kv_a_layernorm"),
        ]

    def get_weight_buffer_meta(self, model, valid_names=None):
        weight_buffer_meta = {}
        for name, param in sorted(model.named_parameters()):
            if valid_names and name not in valid_names:
                continue
            if 'kv_a_proj_with_mqa' in name:
                q_param = dict(model.named_parameters()).get(name.replace('kv_a_proj_with_mqa', 'q_a_proj'))
                qkv_param_shape = torch.cat([q_param, param], dim=0).shape
                qkv_name = name.replace('kv_a_proj_with_mqa', 'qkv_proj')
                weight_buffer_meta[qkv_name] = {'shape': qkv_param_shape, 'dtype': param.dtype}
            elif 'q_a_proj' in name:
                continue
            else:
                weight_buffer_meta[name] = {'shape': param.shape, 'dtype': param.dtype}
        return weight_buffer_meta


class QwenMVWeightAdaptor(MegatronVLLMWeightAdaptor):
    """
    Megatron-vLLM WeightAdaptor for Qwen model architectures.
    """
    def __init__(self, model_config):
        super(QwenMVWeightAdaptor, self).__init__(model_config)


class Qwen2VLWeightAdaptor(MegatronVLLMWeightAdaptor):
    """
    Megatron-vLLM WeightAdaptor for Qwen2VL model architectures.
    """
    def __init__(self, model_config):
        super(Qwen2VLWeightAdaptor, self).__init__(model_config)
        self.params_mapping = [
            ("text_decoder.embedding.word_embeddings", "language_model.model.embed_tokens"),
            ("text_decoder.decoder.layers.*.self_attention.linear_qkv", "language_model.model.layers.*.self_attn.qkv_proj"),
            ("text_decoder.decoder.layers.*.self_attention.linear_proj", "language_model.model.layers.*.self_attn.o_proj"),
            ("text_decoder.decoder.layers.*.input_layernorm", "language_model.model.layers.*.input_layernorm"),
            ("text_decoder.decoder.layers.*.pre_mlp_layernorm", "language_model.model.layers.*.post_attention_layernorm"),
            ("text_decoder.decoder.layers.*.mlp.linear_fc1", "language_model.model.layers.*.mlp.gate_up_proj"),
            ("text_decoder.decoder.layers.*.mlp.linear_fc2", "language_model.model.layers.*.mlp.down_proj"),
            ("text_decoder.decoder.final_layernorm", "language_model.model.norm"),
            ("text_decoder.output_layer", "language_model.lm_head"),
            ("image_encoder.encoder.patch_embed.proj", "visual.patch_embed.proj"),
            ("image_encoder.encoder.blocks.layers.*.self_attention.linear_qkv", "visual.blocks.*.attn.qkv"),
            ("image_encoder.encoder.blocks.layers.*.self_attention.linear_proj", "visual.blocks.*.attn.proj"),
            ("image_encoder.encoder.blocks.layers.*.input_layernorm", "visual.blocks.*.norm1"),
            ("image_encoder.encoder.blocks.layers.*.pre_mlp_layernorm", "visual.blocks.*.norm2"),
            ("image_encoder.encoder.blocks.layers.*.mlp.linear_fc1", "visual.blocks.*.mlp.fc1"),
            ("image_encoder.encoder.blocks.layers.*.mlp.linear_fc2", "visual.blocks.*.mlp.fc2"),
            ("image_encoder.projector.layernorm", "visual.merger.ln_q"),
            ("image_encoder.projector.encoder.linear_fc1", "visual.merger.mlp.0"),
            ("image_encoder.projector.encoder.linear_fc2", "visual.merger.mlp.2"),
        ]
    
    def replace_name_i2t(self, inference_name):
        weight_suffix = ""
        if inference_name.endswith(".weight"):
            weight_suffix = ".weight"
            base_name = inference_name[:-7]
        elif inference_name.endswith(".bias"):
            weight_suffix = ".bias"
            base_name = inference_name[:-5]
        else:
            base_name = inference_name
        
        for megatron_pattern, vllm_pattern in self.params_mapping:
            vllm_regex = vllm_pattern.replace("*", "(\\d+)")
            match = re.match(f"^{vllm_regex}(.*)?$", base_name)
            if match:
                digits = match.groups()
                extra_suffix = match.groups()[-1] if match.groups() and match.groups()[-1] is not None else ""
                
                megatron_result = megatron_pattern
                for i, digit in enumerate(digits[:-1] if extra_suffix else digits):
                    if digit is not None:
                        megatron_result = megatron_result.replace("*", digit, 1)
                
                return megatron_result + extra_suffix + weight_suffix
        
        return inference_name
    
    @staticmethod
    def global2local_layer(name, num_layer_list):
        img_pp_layers, llm_pp_layers = num_layer_list

        if name.startswith('visual') and 'blocks' in name:
            split_name = name.split('.')
            for i, name_part in enumerate(split_name):
                if name_part == 'blocks':
                    break
            block_num_idx = i + 1
            if len(split_name) < block_num_idx + 1 or not split_name[block_num_idx].isdigit():
                raise ValueError(f'Invalid visual block name: {split_name}')
            
            global_idx = int(split_name[block_num_idx])
            local_index = -1
            
            cumulative_layers = 0
            for layers_in_pp_rank in img_pp_layers:
                if layers_in_pp_rank == 0:
                    continue
                if cumulative_layers <= global_idx < cumulative_layers + layers_in_pp_rank:
                    local_index = global_idx - cumulative_layers
                    break
                cumulative_layers += layers_in_pp_rank
            
            if local_index == -1:
                raise ValueError(f'Could not map visual block {global_idx} to a local index with distribution {img_pp_layers}')
            
            split_name[block_num_idx] = str(local_index)
            name = '.'.join(split_name)
        
        elif name.startswith('language_model') and 'layers' in name:
            split_name = name.split('.')
            for i, name_part in enumerate(split_name):
                if name_part == 'layers':
                    break
            layer_num_idx = i + 1
            if len(split_name) < layer_num_idx + 1 or not split_name[layer_num_idx].isdigit():
                raise ValueError(f'Invalid language model layer name: {split_name}')
            
            global_idx = int(split_name[layer_num_idx])
            local_index = -1
            
            cumulative_layers = 0
            for pp_rank, layers_in_pp_rank in enumerate(llm_pp_layers):
                if layers_in_pp_rank == 0:
                    continue
                if cumulative_layers <= global_idx < cumulative_layers + layers_in_pp_rank:
                    local_index = global_idx - cumulative_layers
                    break
                cumulative_layers += layers_in_pp_rank
            
            if local_index == -1:
                raise ValueError(f'Could not map language model layer {global_idx} to a local index with distribution {llm_pp_layers}')
            
            split_name[layer_num_idx] = str(local_index)
            name = '.'.join(split_name)
        
        return name
    
    @staticmethod
    def get_weight_names_per_pp(layer_list, vllm_names):
        img_pp_layers, llm_pp_layers = layer_list
        pp_size = len(img_pp_layers)
        
        visual_weights = []
        lang_weights = []
        visual_pre_layer_weights = []
        visual_post_layer_weights = []
        lang_pre_layer_weights = []
        lang_post_layer_weights = []
        
        for name in vllm_names:
            if name.startswith('visual'):
                if 'blocks' not in name:
                    if 'patch_embed' in name:
                        visual_pre_layer_weights.append(name)
                    elif 'merger' in name:
                        visual_post_layer_weights.append(name)
                else:
                    visual_weights.append(name)
            elif name.startswith('language_model'):
                if 'layers' not in name:
                    if 'embed_tokens' in name:
                        lang_pre_layer_weights.append(name)
                    else:
                        lang_post_layer_weights.append(name)
                else:
                    lang_weights.append(name)
        
        img_blocks_range = []
        llm_layers_range = []
        
        img_start_layer = 0
        for i, layers_in_pp_rank in enumerate(img_pp_layers):
            if layers_in_pp_rank > 0:
                img_blocks_range.append((img_start_layer, img_start_layer + layers_in_pp_rank - 1))
                img_start_layer += layers_in_pp_rank
            else:
                img_blocks_range.append((-1, -1))
        
        llm_start_layer = 0
        for i, layers_in_pp_rank in enumerate(llm_pp_layers):
            if layers_in_pp_rank > 0:
                llm_layers_range.append((llm_start_layer, llm_start_layer + layers_in_pp_rank - 1))
                llm_start_layer += layers_in_pp_rank
            else:
                llm_layers_range.append((-1, -1))
        
        weight_names_per_pp = [[] for _ in range(pp_size)]
        
        last_img_rank = -1
        for i in range(pp_size-1, -1, -1):
            if img_pp_layers[i] > 0:
                last_img_rank = i
                break
        
        first_llm_rank = -1
        for i in range(pp_size):
            if llm_pp_layers[i] > 0:
                first_llm_rank = i
                break
        
        for pp_rank in range(pp_size):
            start_layer, end_layer = img_blocks_range[pp_rank]
            
            if start_layer == 0 and end_layer >= 0:
                weight_names_per_pp[pp_rank].extend(visual_pre_layer_weights)
                
            if start_layer >= 0 and end_layer >= 0:
                for name in visual_weights:
                    match = re.match(r'.*\.blocks\.(\d+)', name)
                    if match:
                        block_num = int(match.group(1))
                        if start_layer <= block_num <= end_layer:
                            weight_names_per_pp[pp_rank].append(name)
            
            if pp_rank == last_img_rank:
                weight_names_per_pp[pp_rank].extend(visual_post_layer_weights)
        
        last_llm_rank = -1
        for i in range(pp_size-1, -1, -1):
            if llm_pp_layers[i] > 0:
                last_llm_rank = i
                break
                
        for pp_rank in range(pp_size):
            start_layer, end_layer = llm_layers_range[pp_rank]
            
            if pp_rank == first_llm_rank:
                weight_names_per_pp[pp_rank].extend(lang_pre_layer_weights)
                
            if start_layer >= 0 and end_layer >= 0:
                for name in lang_weights:
                    match = re.match(r'.*\.layers\.(\d+)', name)
                    if match:
                        layer_num = int(match.group(1))
                        if start_layer <= layer_num <= end_layer:
                            weight_names_per_pp[pp_rank].append(name)
            
            if pp_rank == last_llm_rank:
                weight_names_per_pp[pp_rank].extend(lang_post_layer_weights)
        
        return weight_names_per_pp


class Qwen2_5_VLWeightAdaptor(Qwen2VLWeightAdaptor):
    def __init__(self, model_config):
        super(Qwen2_5_VLWeightAdaptor, self).__init__(model_config)
        self.params_mapping = [
            ("text_decoder.embedding.word_embeddings", "language_model.model.embed_tokens"),
            ("text_decoder.decoder.layers.*.self_attention.linear_qkv", "language_model.model.layers.*.self_attn.qkv_proj"),
            ("text_decoder.decoder.layers.*.self_attention.linear_proj", "language_model.model.layers.*.self_attn.o_proj"),
            ("text_decoder.decoder.layers.*.input_layernorm", "language_model.model.layers.*.input_layernorm"),
            ("text_decoder.decoder.layers.*.pre_mlp_layernorm", "language_model.model.layers.*.post_attention_layernorm"),
            ("text_decoder.decoder.layers.*.mlp.linear_fc1", "language_model.model.layers.*.mlp.gate_up_proj"),
            ("text_decoder.decoder.layers.*.mlp.linear_fc2", "language_model.model.layers.*.mlp.down_proj"),
            ("text_decoder.decoder.final_layernorm", "language_model.model.norm"),
            ("text_decoder.output_layer", "language_model.lm_head"),
            ("image_encoder.encoder.patch_embed.proj", "visual.patch_embed.proj"),
            ("image_encoder.encoder.blocks.layers.*.self_attention.linear_qkv", "visual.blocks.*.attn.qkv"),
            ("image_encoder.encoder.blocks.layers.*.self_attention.linear_proj", "visual.blocks.*.attn.proj"),
            ("image_encoder.encoder.blocks.layers.*.input_layernorm", "visual.blocks.*.norm1"),
            ("image_encoder.encoder.blocks.layers.*.pre_mlp_layernorm", "visual.blocks.*.norm2"),
            ("image_encoder.encoder.blocks.layers.*.mlp.linear_fc1", "visual.blocks.*.mlp.gate_up_proj"),
            ("image_encoder.encoder.blocks.layers.*.mlp.linear_fc2", "visual.blocks.*.mlp.down_proj"),
            ("image_encoder.projector.layernorm", "visual.merger.ln_q"),
            ("image_encoder.projector.encoder.linear_fc1", "visual.merger.mlp.0"),
            ("image_encoder.projector.encoder.linear_fc2", "visual.merger.mlp.2"),
        ]


WEIGHT_ADAPTOR_REGISTRY = {
    "Qwen2ForCausalLM": QwenMVWeightAdaptor,
    "DeepseekV3ForCausalLM": DeepSeekMVWeightAdaptor,
    "DeepseekV2ForCausalLM": DeepSeekMVWeightAdaptor,
    "Qwen2VLForConditionalGeneration": Qwen2VLWeightAdaptor,
    "CustomQwen2VLForConditionalGeneration": Qwen2VLWeightAdaptor,
    "Qwen2_5_VLForConditionalGeneration": Qwen2_5_VLWeightAdaptor,
}


def get_weight_adaptor(arch: str):
    if arch in WEIGHT_ADAPTOR_REGISTRY:
        return WEIGHT_ADAPTOR_REGISTRY[arch]
    raise ValueError(f"Model architectures {arch} are not supported for now.")