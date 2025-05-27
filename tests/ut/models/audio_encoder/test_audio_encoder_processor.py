import pytest
import mindspeed.megatron_adaptor
import torch
import torch_npu

from mindspeed_mm.models.whisper.whisper_model import WhisperEncoder
from tests.ut.utils import judge_expression
from tests.ut.utils import TestConfig


AUDIO_INPUT_PATH = "/home/ci_resource/data/whisper-encoder/audio_input.pt"
WHISPER_ENCODER_PATH = "/home/ci_resource/models/whisper-encoder/whisperencoder.pt"
WHISPER_ENCODER_OUTPUT = -4.763552665710449


class TestAudioEncoder:
    """
    audio encoder processor test case
    """
    def test_whisper(self):
        """
        test whisper encoder processor
        """
        audio_encoder_dict = {
            "dropout": 0.0,
            "encoder_layerdrop": 0.0,
            "d_model": 1280,
            "num_mel_bins": 128,
            "pad_token_id": 50256,
            "max_source_positions": 1500,
            "scale_embedding": False,
            "encoder_layers": 32,
            "encoder_attention_heads": 20,
            "attention_dropout": 0.0,
            "activation_function": "gelu",
            "activation_dropout": 0.0,
            "encoder_ffn_dim": 5120,
        }
        audio_encoder_configs = TestConfig(audio_encoder_dict)
        audio_encoder = WhisperEncoder(audio_encoder_configs)
        audio_state_dict = torch.load(WHISPER_ENCODER_PATH)
        audio_encoder.load_state_dict(audio_state_dict)
        audio_encoder = audio_encoder.npu()
        audio_input = torch.load(AUDIO_INPUT_PATH)
        audio_output = audio_encoder(audio_input.npu())
        judge_expression(audio_output.min().item() == WHISPER_ENCODER_OUTPUT)
