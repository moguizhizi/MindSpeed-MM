import importlib

import torch
import torch.nn as nn

from mindspeed_mm.models.text_encoder.hunyuan_mllm_text_encoder import HunyuanMLLmModel
from mindspeed_mm.models.text_encoder.stepllm_text_encoder import StepLLmModel
from mindspeed_mm.utils.utils import get_dtype

TRANSFORMERS_TEXT_ENCODER_MAPPING = {
    "T5": "T5EncoderModel",
    "MT5": "MT5EncoderModel",
    "UMT5": "UMT5EncoderModel",
    "CLIP": "CLIPTextModel",
    "Auto": "AutoModel",
    "BertModel": "BertModel",
    "CLIPWithProjection": "CLIPTextModelWithProjection",
}

CUSTOM_TEXT_ENCODER_MAPPING = {
    "StepLLmModel": StepLLmModel,
    "HunyuanMLLmModel": HunyuanMLLmModel,
}


class TextEncoder(nn.Module):
    """
    Configuration for initializing one or more Text Encoder Model instances.

    Args:
        config (Optional[dict, list(dict)]): the general config for Text Encoder Model.
        - If `config` is a dictionary, it specifies the parameters for a single Text Encoder Model instance.
            e.g.
            {
                (1) args for our feautrues
                "backend": type-str, "hf" or "om",
                "model_id": type-str, "AutoModel" or other automodel name,
                "dtype": type-str, dtype of text encoder

                (2) args for automodel.from_pretrained() of transformers or openmind
                "pretrained_model_name_or_path": type-str, local path or hub path,
                "local_files_only": type-bool,
                ...
            }
        - If `config` is a list of dictionaries, each dictionary in the list will be used to instantiate a separate Text Encoder Model instance,
            effectively allowing the creation of multiple Text Encoder based on different configurations.
    """
    def __init__(self, config):
        super().__init__()

        if isinstance(config, list) or isinstance(config, tuple):
            self.text_encoders = nn.ModuleList()
            for config_i in config:
                text_encoder_i = self._init_text_encoder(config_i)
                self.text_encoders.append(text_encoder_i)
        else:
            self.text_encoders = self._init_text_encoder(config)

    def get_model(self):
        return self.text_encoders

    def encode(self, input_ids, mask, **kwargs):
        if isinstance(self.text_encoders, nn.ModuleList):
            outputs = []
            masks = []
            for i, text_encoder_i in enumerate(self.text_encoders):
                input_ids_i = input_ids[i]
                mask_i = mask[i]
                output, att_mask = self._single_encode(text_encoder_i, input_ids_i, mask_i, **kwargs)
                outputs.append(output)
                masks.append(att_mask)
        else:
            outputs, masks = self._single_encode(self.text_encoders, input_ids, mask)
        return outputs, masks

    def _single_encode(self, text_encoder, input_ids, attention_mask, **kwargs):
        *BN, L = input_ids.shape
        input_ids = input_ids.to(text_encoder.device).view(-1, L)
        attention_mask = attention_mask.to(text_encoder.device).view(-1, L)
        model_attention_mask = attention_mask if text_encoder.use_attention_mask else None
        model_kwargs = {}
        if text_encoder.using_kwargs:
            for k in text_encoder.using_kwargs:
                if k in kwargs.keys():
                    model_kwargs[k] = kwargs[k]
                else:
                    raise ValueError(f"{k} is not in kwargs")
        output = text_encoder(
            input_ids=input_ids,
            attention_mask=model_attention_mask,
            output_hidden_states=text_encoder.hidden_state_skip_layer is not None,
            **model_kwargs
        )

        emb = output[text_encoder.output_key]
        if text_encoder.hidden_state_skip_layer:
            emb = emb[-(text_encoder.hidden_state_skip_layer + 1)]

        if text_encoder.ucg_rate is not None and text_encoder.ucg_rate > 0.0:
            def expand_dims_like(x, y):
                while x.dim() != y.dim():
                    x = x.unsqueeze(-1)
                return x

            emb = (
                expand_dims_like(
                    torch.bernoulli(
                        (1.0 - text_encoder.ucg_rate) * torch.ones(emb.shape[0], device=emb.device, dtype=emb.dtype)),
                    emb,
                )
                * emb
            )

        if text_encoder.output_key in ["last_hidden_state", "hidden_states"]:
            emb = emb.view(*BN, emb.shape[-2], -1)
        elif text_encoder.output_key in ["pooler_output", "text_embeds"]:
            emb = emb.view(*BN, -1)
        else:
            raise NotImplementedError(f"Text encoder output_key: {text_encoder.output_key} is not implenmented! ")

        if text_encoder.use_attention_mask:
            attention_mask = model_attention_mask
        attention_mask = attention_mask.view(*BN, -1)

        return emb, attention_mask

    def _init_text_encoder(self, config):
        if not isinstance(config, dict):
            config = config.to_dict()

        backend = config.pop("hub_backend")
        use_attention_mask = config.pop("use_attention_mask", True)
        ucg_rate = config.pop("ucg_rate", None)
        output_key = config.pop("output_key", "last_hidden_state")
        hidden_state_skip_layer = config.pop("hidden_state_skip_layer", None)
        using_kwargs = config.pop("using_kwargs", None)

        config["pretrained_model_name_or_path"] = config.pop("from_pretrained")
        config["torch_dtype"] = get_dtype(config.pop("dtype"))
        config["local_files_only"] = True
        try:
            from megatron.training import get_args
            config["trust_remote_code"] = get_args().trust_remote_code
        except (ImportError, AssertionError, AttributeError):
            config["trust_remote_code"] = False

        # Only huggingface backend is supported, OpenMind backend will be supported soon.
        model_id = config.pop("model_id")
        if model_id in TRANSFORMERS_TEXT_ENCODER_MAPPING:
            module = importlib.import_module("transformers")
            self.automodel_name = TRANSFORMERS_TEXT_ENCODER_MAPPING[model_id]
            automodel = getattr(module, self.automodel_name)
            text_encoder = automodel.from_pretrained(**config)
        elif model_id in CUSTOM_TEXT_ENCODER_MAPPING:
            automodel = CUSTOM_TEXT_ENCODER_MAPPING[model_id]
            text_encoder = automodel.from_pretrained(**config)
        else:
            raise ValueError(f"Model ID {model_id} is not supported for text encoder")

        setattr(text_encoder, "ucg_rate", ucg_rate)
        setattr(text_encoder, "use_attention_mask", use_attention_mask)
        setattr(text_encoder, "output_key", output_key)
        setattr(text_encoder, "hidden_state_skip_layer", hidden_state_skip_layer)
        setattr(text_encoder, "using_kwargs", using_kwargs)

        if hidden_state_skip_layer and output_key not in ["hidden_states"]:
            raise ValueError("If use hidden_state_skip, the output_keys must in [`hidden_states`]")
        return text_encoder