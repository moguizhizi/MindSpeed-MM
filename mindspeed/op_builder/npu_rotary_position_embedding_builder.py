from mindspeed.op_builder.builder import MindSpeedOpBuilder


class RotaryPositionEmbeddingOpBuilder(MindSpeedOpBuilder):
    OP_NAME = "npu_rotary_position_embedding"

    def __init__(self):
        super(RotaryPositionEmbeddingOpBuilder, self).__init__(self.OP_NAME)

    def sources(self):
        return ['ops/csrc/cann/npu_rotary_position_embedding.cpp']

    def include_paths(self):
        paths = super().include_paths()
        paths += ['ops/csrc/cann/inc']
        return paths

    def cxx_args(self):
        args = super().cxx_args()
        args += [
            '-Wno-sign-compare',
            '-Wno-deprecated-declarations',
            '-Wno-return-type',
            "-D__FILENAME__='\"$$(notdir $$(abspath $$<))\"'"
        ]
        return args
