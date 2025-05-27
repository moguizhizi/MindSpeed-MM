# coding=utf-8
# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import wraps


def extra_args_provider_decorator(extra_args_provider):
    @wraps(extra_args_provider)
    def wrapper(parser):
        if extra_args_provider is not None:
            parser = extra_args_provider(parser)
        parser = process_args(parser)
        return parser

    return wrapper


def process_args(parser):
    parser.conflict_handler = "resolve"
    parser = _add_lora_args(parser)
    parser = _add_training_args(parser)
    parser = _add_network_size_args(parser)
    parser = _add_dummy_optimizer_args(parser)
    parser = _add_logging_args(parser)
    parser = _add_security_args(parser)
    parser = _add_auto_parallel_mm_args(parser)
    parser = _add_rlfh_args(parser)
    parser = _add_network_args(parser)
    return parser


def _add_lora_args(parser):
    group = parser.add_argument_group(title='lora')

    group.add_argument('--lora-target-modules', nargs='+', type=str, default=[],
                       help='Use lora in target modules.')
    group.add_argument('--load-base-model', type=str, default=None,
                       help='Directory containing a base model checkpoint for lora.')
    group.add_argument('--lora-dropout', type=float, default=0.0, help="lora dropout rate")
    group.add_argument('--lora-r', type=int, default=8,
                       help='Lora rank.')
    group.add_argument('--lora-alpha', type=int, default=16,
                       help='Lora alpha.')
    group.add_argument('--lora-register-forward-hook', nargs='+', type=str,
                       default=['word_embeddings', 'input_layernorm', 'final_layernorm'],
                       help='Lora register forward hook.')

    return parser


def _add_training_args(parser):
    group = parser.add_argument_group(title='training')

    group.add_argument('--use-deter-comp',
                       action='store_true',
                       default=False,
                       help='Enable deterministic computing for npu')
    group.add_argument('--jit-compile',
                       action='store_true',
                       default=False,
                       help='Setting jit compile mode to True')
    group.add_argument('--allow-tf32',
                       action='store_true',
                       default=False,
                       help='Use tf32 to train')
    group.add_argument('--allow-internal-format',
                       action='store_true',
                       default=False,
                       help='Use internal format to train')
    group.add_argument('--virtual-pipeline-model-parallel-size',
                       type=int,
                       default=None,
                       help='vpp size')
    group.add_argument('--encoder-dp-balance',
                       action='store_true',
                       default=False,
                       help='Balance for encoder')
    group.add_argument('--recompute-skip-core-attention',
                       action='store_true',
                       default=False,
                       help='Recomputing will skip the Flash attention if True')
    group.add_argument('--recompute-num-layers-skip-core-attention',
                       type=int,
                       default=0)

    return parser


def _add_network_size_args(parser):
    group = parser.add_argument_group(title='network_size_args')

    group.add_argument('--padded-vocab-size',
                       type=int,
                       default=None,
                       help='set padded vocab size')

    return parser


def _add_dummy_optimizer_args(parser):
    group = parser.add_argument_group(title='dummy optimizer args')

    group.add_argument('--enable-dummy-optimizer',
                       action='store_true',
                       default=False,
                       help='enable dummy optimizer')

    return parser


def _add_logging_args(parser):
    group = parser.add_argument_group(title='logging')

    group.add_argument('--log-tps',
                       action='store_true',
                       default=False,
                       help='calculate and log average tokens per sample')

    return parser


def _add_security_args(parser):
    group = parser.add_argument_group(title='security configuration')

    group.add_argument('--trust-remote-code',
                       action='store_true',
                       default=False,
                       help='Whether or not to allow for custom models defined on the Hub in their own modeling files.')

    return parser


def _add_auto_parallel_mm_args(parser):
    group = parser.add_argument_group(title='auto_parallel_mm')
    group.add_argument('--profile-subgraph-seg', action='store_true', default=False, help='model segmentation')
    group.add_argument('--profile-stage', type=int, default=None, help='model profile stage')
    group.add_argument('--simulated-nnodes', type=int, default=None, help='the simulated number of node in the cluster')
    group.add_argument('--simulated-nproc-per-node', type=int, default=None, help='the simulated number of NPU on each node')

    return parser


def _add_rlfh_args(parser):
    group = parser.add_argument_group(title='dpo')

    group.add_argument(
        '--dpo-beta',
        type=float,
        default=0.1,
        help="The beta parameter for the DPO loss"
    )
    group.add_argument(
        '--dpo-loss-type',
        default="sigmoid",
        choices=["sigmoid"],
        help="The type of DPO loss to use"
    )
    group.add_argument(
        "--dpo-label-smoothing",
        type=float,
        default=0.0,
        help="The robust DPO label smoothing parameter in cDPO that should be between 0 and 0.5."
    )
    group.add_argument(
        '--ref-model',
        default=None,
        type=str,
        help='Path to the reference model used for the PPO or DPO training.'
    )
    group.add_argument(
        '--pref-ftx',
        default=0.0,
        type=float,
        help="The supervised fine-tuning loss coefficient in DPO training.",
    )

    return parser


def _add_network_args(parser):
    group = parser.add_argument_group(title='network')

    # MM_GRPO use，judging training methods
    group.add_argument(
        '--stage',
        default=None,
        choices=["ray_grpo"],
        help='Determine training mode'
    )

    return parser
