import mindspeed.megatron_adaptor

from mindspeed_mm.models.common.embeddings.pos_embeddings import Rotary3DPositionEmbedding
from tests.ut.utils import judge_expression


class TestCogVideoXRope:
    def test_init_rope_1_0_t2v(self):
        rope = Rotary3DPositionEmbedding(30, 45, 1, 3072, 64, 226, learnable_pos_embed=False)
        judge_expression(isinstance(rope, Rotary3DPositionEmbedding))
