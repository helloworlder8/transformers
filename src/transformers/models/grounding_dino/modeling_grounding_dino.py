# coding=utf-8
# Copyright 2024 IDEA Research and The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch Grounding DINO model."""

import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from ...activations import ACT2FN
from ...file_utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_timm_available,
    is_torch_cuda_available,
    replace_return_docstrings,
    requires_backends,
)
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import meshgrid
from ...utils import is_ninja_available, logging
from ...utils.backbone_utils import load_backbone
from ..auto import AutoModel
from .configuration_grounding_dino import GroundingDinoConfig


if is_timm_available():
    from timm import create_model


logger = logging.get_logger(__name__)

MultiScaleDeformableAttention = None


# Copied from models.deformable_detr.load_cuda_kernels
def load_cuda_kernels():
    from torch.utils.cpp_extension import load

    global MultiScaleDeformableAttention

    root = Path(__file__).resolve().parent.parent.parent / "kernels" / "deformable_detr"
    src_files = [
        root / filename
        for filename in [
            "vision.cpp",
            os.path.join("cpu", "ms_deform_attn_cpu.cpp"),
            os.path.join("cuda", "ms_deform_attn_cuda.cu"),
        ]
    ]

    MultiScaleDeformableAttention = load(
        "MultiScaleDeformableAttention",
        src_files,
        with_cuda=True,
        extra_include_paths=[str(root)],
        extra_cflags=["-DWITH_CUDA=1"],
        extra_cuda_cflags=[
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ],
    )


# Copied from transformers.models.deformable_detr.modeling_deformable_detr.MultiScaleDeformableAttentionFunction
class MultiScaleDeformableAttentionFunction(Function):
    @staticmethod
    def forward(
        context,
        value,
        value_spatial_shapes,
        value_level_start_index,
        sampling_locations,
        attention_weights,
        im2col_step,
    ):
        context.im2col_step = im2col_step
        output = MultiScaleDeformableAttention.ms_deform_attn_forward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            context.im2col_step,
        )
        context.save_for_backward(
            value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights
        )
        return output

    @staticmethod
    @once_differentiable
    def backward(context, grad_output):
        (
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
        ) = context.saved_tensors
        grad_value, grad_sampling_loc, grad_attn_weight = MultiScaleDeformableAttention.ms_deform_attn_backward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            grad_output,
            context.im2col_step,
        )

        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "GroundingDinoConfig"
_CHECKPOINT_FOR_DOC = "IDEA-Research/grounding-dino-tiny"


@dataclass
class GroundingDinoDecoderOutput(ModelOutput):
    """
    Args:

    """
    decoder_query_embeds_states: torch.FloatTensor = None
    decoder_query_embeds_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_refine_coords_hidden_states: torch.FloatTensor = None
    decoder_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None

@dataclass
class GroundingDinoEncoderOutput(ModelOutput):
    """
    Args:
    
    """
    encoder_vision_features_states: torch.FloatTensor = None
    encoder_text_features_states: torch.FloatTensor = None
    encoder_vision_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_text_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None


@dataclass
class GroundingDinoModelOutput(ModelOutput):
    """
    Args:
    
    """
    decoder_query_embeds_states: torch.FloatTensor = None
    decoder_query_embeds_hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_refine_coords_hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None

    encoder_vision_features_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    encoder_text_features_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    encoder_vision_hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    encoder_text_hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    encoder_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    
    init_topk_coords: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    enc_outputs_class: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    enc_outputs_coord: Optional[Tuple[Tuple[torch.FloatTensor]]] = None


@dataclass
class GroundingDinoObjectDetectionOutput(ModelOutput):
    """
    Args:
    """
    loss: Optional[Tuple[torch.FloatTensor]] = None
    loss_dict: Optional[Tuple[torch.FloatTensor]] = None
    auxiliary_outputs: Optional[Tuple[torch.FloatTensor]] = None
    
    outputs_classes: Optional[Tuple[torch.FloatTensor]] = None
    outputs_coords: Optional[Tuple[torch.FloatTensor]] = None
    
    decoder_query_embeds_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_query_embeds_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_refine_coords_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    
    encoder_vision_features_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_text_features_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_vision_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_text_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    
    init_topk_coords: Optional[Tuple[torch.FloatTensor]] = None
    enc_outputs_class: Optional[Tuple[torch.FloatTensor]] = None
    enc_outputs_coord: Optional[Tuple[torch.FloatTensor]] = None


# Copied from transformers.models.detr.modeling_detr.DetrFrozenBatchNorm2d with Detr->GroundingDino
class GroundingDinoFrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt, without which any other models than
    torchvision.models.resnet[18,34,50,101] produce nans.
    """

    def __init__(self, n):
        super().__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def forward(self, x):
        # move reshapes to the beginning
        # to make it user-friendly
        weight = self.weight.reshape(1, -1, 1, 1)
        bias = self.bias.reshape(1, -1, 1, 1)
        running_var = self.running_var.reshape(1, -1, 1, 1)
        running_mean = self.running_mean.reshape(1, -1, 1, 1)
        epsilon = 1e-5
        scale = weight * (running_var + epsilon).rsqrt()
        bias = bias - running_mean * scale
        return x * scale + bias


# Copied from transformers.models.detr.modeling_detr.replace_batch_norm with Detr->GroundingDino
def replace_batch_norm(model):
    r"""
    Recursively replace all `torch.nn.BatchNorm2d` with `GroundingDinoFrozenBatchNorm2d`.

    Args:
        model (torch.nn.Module):
            input model
    """
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            new_module = GroundingDinoFrozenBatchNorm2d(module.num_features)

            if not module.weight.device == torch.device("meta"):
                new_module.weight.data.copy_(module.weight)
                new_module.bias.data.copy_(module.bias)
                new_module.running_mean.data.copy_(module.running_mean)
                new_module.running_var.data.copy_(module.running_var)

            model._modules[name] = new_module

        if len(list(module.children())) > 0:
            replace_batch_norm(module)


class GroundingDinoConvEncoder(nn.Module):
    """
    Convolutional backbone, using either the AutoBackbone API or one from the timm library.

    nn.BatchNorm2d layers are replaced by GroundingDinoFrozenBatchNorm2d as defined above.

    """

    def __init__(self, config):
        super().__init__()

        self.config = config

        if config.use_timm_backbone:
            requires_backends(self, ["timm"])
            backbone = create_model(
                config.backbone,
                pretrained=config.use_pretrained_backbone,
                features_only=True,
                **config.backbone_kwargs,
            )
        else:
            backbone = load_backbone(config)

        # replace batch norm by frozen batch norm
        with torch.no_grad():
            replace_batch_norm(backbone)
        self.model = backbone
        self.intermediate_channel_sizes = (
            self.model.feature_info.channels() if config.use_timm_backbone else self.model.channels
        )

        backbone_model_type = None
        if config.backbone is not None:
            backbone_model_type = config.backbone
        elif config.backbone_config is not None:
            backbone_model_type = config.backbone_config.model_type
        else:
            raise ValueError("Either `backbone` or `backbone_config` should be provided in the config")

        if "resnet" in backbone_model_type:
            for name, parameter in self.model.named_parameters():
                if config.use_timm_backbone:
                    if "layer2" not in name and "layer3" not in name and "layer4" not in name:
                        parameter.requires_grad_(False)
                else:
                    if "stage.1" not in name and "stage.2" not in name and "stage.3" not in name:
                        parameter.requires_grad_(False)

    # Copied from transformers.models.detr.modeling_detr.DetrConvEncoder.forward with Detr->GroundingDino
    def forward(self, pixel_values: torch.Tensor, pixel_mask: torch.Tensor):
        # send pixel_values through the model to get list of feature maps
        features = self.model(pixel_values) if self.config.use_timm_backbone else self.model(pixel_values).feature_maps

        out = []
        for feature_map in features:
            # downsample pixel_mask to match shape of corresponding feature_map
            mask = nn.functional.interpolate(pixel_mask[None].float(), size=feature_map.shape[-2:]).to(torch.bool)[0]
            out.append((feature_map, mask))
        return out


# Copied from transformers.models.detr.modeling_detr.DetrConvModel with Detr->GroundingDino
class GroundingDinoConvModel(nn.Module):
    """
    This module adds 2D position embeddings to all query_embeds_hidden feature maps of the convolutional encoder.
    """

    def __init__(self, conv_encoder, position_embedding):
        super().__init__()
        self.conv_encoder = conv_encoder
        self.position_embedding = position_embedding

    def forward(self, pixel_values, pixel_mask):
        # send pixel_values and pixel_mask through backbone to get list of (feature_map, pixel_mask) tuples
        out = self.conv_encoder(pixel_values, pixel_mask)
        pos = []
        for feature_map, mask in out:
            # position encoding
            pos.append(self.position_embedding(feature_map, mask).to(feature_map.dtype))

        return out, pos #输出 输出加上位置编码


class GroundingDinoSinePositionEmbedding(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one used by the Attention is all you
    need paper, generalized to work on images.
    """

    def __init__(self, config):
        super().__init__()
        self.embedding_dim = config.d_model // 2
        self.temperature = config.positional_embedding_temperature
        self.scale = 2 * math.pi

    def forward(self, pixel_values, pixel_mask):
        y_embed = pixel_mask.cumsum(1, dtype=torch.float32)
        x_embed = pixel_mask.cumsum(2, dtype=torch.float32)
        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.embedding_dim, dtype=torch.float32, device=pixel_values.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.embedding_dim)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class GroundingDinoLearnedPositionEmbedding(nn.Module):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, config):
        super().__init__()

        embedding_dim = config.d_model // 2
        self.row_embeddings = nn.Embedding(50, embedding_dim)
        self.column_embeddings = nn.Embedding(50, embedding_dim)

    def forward(self, pixel_values, pixel_mask=None):
        height, width = pixel_values.shape[-2:]
        width_values = torch.arange(width, device=pixel_values.device)
        height_values = torch.arange(height, device=pixel_values.device)
        x_emb = self.column_embeddings(width_values)
        y_emb = self.row_embeddings(height_values)
        pos = torch.cat([x_emb.unsqueeze(0).repeat(height, 1, 1), y_emb.unsqueeze(1).repeat(1, width, 1)], dim=-1)
        pos = pos.permute(2, 0, 1)
        pos = pos.unsqueeze(0)
        pos = pos.repeat(pixel_values.shape[0], 1, 1, 1)
        return pos


def build_position_encoding(config):
    if config.position_embedding_type == "sine":
        position_embedding = GroundingDinoSinePositionEmbedding(config)
    elif config.position_embedding_type == "learned":
        position_embedding = GroundingDinoLearnedPositionEmbedding(config)
    else:
        raise ValueError(f"Not supported {config.position_embedding_type}")

    return position_embedding


# Copied from transformers.models.deformable_detr.modeling_deformable_detr.multi_scale_deformable_attention
def multi_scale_deformable_attention(
    value: Tensor,
    value_spatial_shapes: Union[Tensor, List[Tuple]],
    sampling_locations: Tensor,
    attention_weights: Tensor,
) -> Tensor:
    batch_size, _, num_heads, hidden_dim = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
    value_list = value.split([height * width for height, width in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level_id, (height, width) in enumerate(value_spatial_shapes):
        # batch_size, height*width, num_heads, hidden_dim
        # -> batch_size, height*width, num_heads*hidden_dim
        # -> batch_size, num_heads*hidden_dim, height*width
        # -> batch_size*num_heads, hidden_dim, height, width
        value_l_ = (
            value_list[level_id].flatten(2).transpose(1, 2).reshape(batch_size * num_heads, hidden_dim, height, width)
        )
        # batch_size, num_queries, num_heads, num_points, 2
        # -> batch_size, num_heads, num_queries, num_points, 2
        # -> batch_size*num_heads, num_queries, num_points, 2
        sampling_grid_l_ = sampling_grids[:, :, :, level_id].transpose(1, 2).flatten(0, 1)
        # batch_size*num_heads, hidden_dim, num_queries, num_points
        sampling_value_l_ = nn.functional.grid_sample(
            value_l_, sampling_grid_l_, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        sampling_value_list.append(sampling_value_l_)
    # (batch_size, num_queries, num_heads, num_levels, num_points)
    # -> (batch_size, num_heads, num_queries, num_levels, num_points)
    # -> (batch_size, num_heads, 1, num_queries, num_levels*num_points)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        batch_size * num_heads, 1, num_queries, num_levels * num_points
    )
    output = (
        (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
        .sum(-1)
        .view(batch_size, num_heads * hidden_dim, num_queries)
    )
    return output.transpose(1, 2).contiguous()


# Copied from transformers.models.deformable_detr.modeling_deformable_detr.DeformableDetrMultiscaleDeformableAttention with DeformableDetr->GroundingDino, Deformable DETR->Grounding DINO
class GroundingDinoMultiscaleDeformableAttention(nn.Module):
    """
    Multiscale deformable attention as proposed in Deformable DETR.
    """

    def __init__(self, config: GroundingDinoConfig, num_heads: int, n_points: int):
        super().__init__()

        kernel_loaded = MultiScaleDeformableAttention is not None
        if is_torch_cuda_available() and is_ninja_available() and not kernel_loaded:
            try:
                load_cuda_kernels()
            except Exception as e:
                logger.warning(f"Could not load the custom kernel for multi-scale deformable attention: {e}")

        if config.d_model % num_heads != 0:
            raise ValueError(
                f"embed_dim (d_model) must be divisible by num_heads, but got {config.d_model} and {num_heads}"
            )
        dim_per_head = config.d_model // num_heads
        # check if dim_per_head is power of 2
        if not ((dim_per_head & (dim_per_head - 1) == 0) and dim_per_head != 0):
            warnings.warn(
                "You'd better set embed_dim (d_model) in GroundingDinoMultiscaleDeformableAttention to make the"
                " dimension of each attention head a power of 2 which is more efficient in the authors' CUDA"
                " implementation."
            )

        self.im2col_step = 64

        self.d_model = config.d_model
        self.n_levels = config.num_feature_levels
        self.n_heads = num_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(config.d_model, num_heads * self.n_levels * n_points * 2)
        self.attention_weights = nn.Linear(config.d_model, num_heads * self.n_levels * n_points)
        self.value_proj = nn.Linear(config.d_model, config.d_model)
        self.output_proj = nn.Linear(config.d_model, config.d_model)

        self.disable_custom_kernels = config.disable_custom_kernels

    def with_pos_embed(self, tensor: torch.Tensor, posi_embed: Optional[Tensor]):
        return tensor if posi_embed is None else tensor + posi_embed





    def forward(
        self,
        query_embeds: torch.Tensor,
        query_posi_embeds: Optional[torch.Tensor] = None,
        expand_topk_coords=None,
        
        vision_features=None,
        vision_attention_mask: Optional[torch.Tensor] = None,
        
        vision_map_shapes=None,
        level_start_index=None,
        output_attentions=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Multi-scale deformable attention mechanism"""

        # Add positional embeddings to queries
        if query_posi_embeds is not None:
            query_embeds = self.with_pos_embed(query_embeds, query_posi_embeds)

        batch_size, num_queries, _ = query_embeds.shape
        batch_size, vision_seq_len, _ = vision_features.shape

        # Validate vision map shapes
        if (vision_map_shapes[:, 0] * vision_map_shapes[:, 1]).sum() != vision_seq_len:
            raise ValueError("Spatial shapes and sequence length of vision features do not match")

        # Project values and apply mask
        value = self.value_proj(vision_features)
        if vision_attention_mask is not None:
            value = value.masked_fill(~vision_attention_mask[..., None], 0.0)
        value = value.view(batch_size, vision_seq_len, self.n_heads, self.d_model // self.n_heads) #->torch.Size([1, 17821, 8, 32])

        # Compute sampling offsets and attention weights
        sampling_offsets = self.sampling_offsets(query_embeds).view(
            batch_size, num_queries, self.n_heads, self.n_levels, self.n_points, 2
        ) #->torch.Size([1, 17821, 8, 4, 4, 2])
        attention_weights = F.softmax(
            self.attention_weights(query_embeds).view(
                batch_size, num_queries, self.n_heads, self.n_levels * self.n_points
            ),
            dim=-1
        ).view(batch_size, num_queries, self.n_heads, self.n_levels, self.n_points) #->torch.Size([1, 17821, 8, 4, 4])

        # Compute sampling locations
        coords_dim = expand_topk_coords.shape[-1]
        if coords_dim == 2:
            offset_normalizer = torch.stack([vision_map_shapes[..., 1], vision_map_shapes[..., 0]], -1)
            sampling_locations = (
                expand_topk_coords[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif coords_dim == 4:
            sampling_locations = (
                expand_topk_coords[:, :, None, :, None, :2]
                + sampling_offsets / self.n_points * expand_topk_coords[:, :, None, :, None, 2:] * 0.5
            )
        else:
            raise ValueError(f"Last dimension of expand_topk_coords must be 2 or 4, but got {coords_dim}")

        # Perform multi-scale deformable attention
        if self.disable_custom_kernels or MultiScaleDeformableAttention is None:
            output = multi_scale_deformable_attention(value, vision_map_shapes, sampling_locations, attention_weights)
        else:
            try:
                output = MultiScaleDeformableAttentionFunction.apply(
                    value,
                    vision_map_shapes,
                    level_start_index,
                    sampling_locations,
                    attention_weights,
                    self.im2col_step,
                )
            except Exception:
                output = multi_scale_deformable_attention(value, vision_map_shapes, sampling_locations, attention_weights)

        # Project output
        output = self.output_proj(output)

        return output, attention_weights

class GroundingDinoTextEnhancerLayer(nn.Module):
    """Vanilla Transformer with text embeddings as input"""

    def __init__(self, config):
        super().__init__()
        self.self_attn = GroundingDinoMultiheadAttention(
            config, num_attention_heads=config.encoder_attention_heads // 2
        )

        # Implementation of Feedforward model
        self.fc1 = nn.Linear(config.d_model, config.encoder_ffn_dim // 2)
        self.fc2 = nn.Linear(config.encoder_ffn_dim // 2, config.d_model)

        self.layer_norm_before = nn.LayerNorm(config.d_model, config.layer_norm_eps)
        self.layer_norm_after = nn.LayerNorm(config.d_model, config.layer_norm_eps)

        self.activation = ACT2FN[config.activation_function]
        self.num_heads = config.encoder_attention_heads // 2
        self.dropout = config.text_enhancer_dropout

    def with_pos_embed(self, hidden_state: Tensor, posi_embed: Optional[Tensor]):
        return hidden_state if posi_embed is None else hidden_state + posi_embed

    def forward(
        self,
        text_features: torch.FloatTensor,
        text_self_attention_mask: Optional[torch.BoolTensor] = None,
        text_posi_embed: Optional[torch.FloatTensor] = None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Self-attention mechanism for text features"""

        # Process attention mask
        if text_self_attention_mask is not None and text_self_attention_mask.dim() == 3:
            text_self_attention_mask = text_self_attention_mask[:, None, :, :].repeat(1, self.num_heads, 1, 1)
            text_self_attention_mask = (1.0 - text_self_attention_mask.to(dtype=text_features.dtype)) * torch.finfo(text_features.dtype).min #->torch.Size([1, 4, 9, 9])

        """ 加入位置编码 """
        queries = keys = self.with_pos_embed(text_features, text_posi_embed)
        """ 执行自注意力 """
        attention_output, attention_weights = self.self_attn(
            queries=queries,
            keys=keys,
            values=text_features,
            attention_mask=text_self_attention_mask,
            output_attentions=True,
        )

        """ dropout 残差连接 层归一化 """
        attention_output = nn.functional.dropout(attention_output, p=self.dropout, training=self.training)
        text_features = self.layer_norm_before(text_features + attention_output)

        """ MLP加残差连接 """
        residual = text_features
        text_features = self.activation(self.fc1(text_features))
        text_features = nn.functional.dropout(text_features, p=self.dropout, training=self.training)
        text_features = self.fc2(text_features)
        text_features = nn.functional.dropout(text_features, p=self.dropout, training=self.training)
        text_features = self.layer_norm_after(text_features + residual)

        return text_features, attention_weights


class GroundingDinoBiMultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        vision_dim = text_dim = config.d_model
        embed_dim = config.encoder_ffn_dim // 2
        num_heads = config.encoder_attention_heads // 2
        dropout = config.fusion_dropout

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.vision_dim = vision_dim
        self.text_dim = text_dim

        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by `num_heads` (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
            )
        self.scale = self.head_dim ** (-0.5)
        self.dropout = dropout

        self.vision_proj = nn.Linear(self.vision_dim, self.embed_dim)
        self.text_proj = nn.Linear(self.text_dim, self.embed_dim)
        self.values_vision_proj = nn.Linear(self.vision_dim, self.embed_dim)
        self.values_text_proj = nn.Linear(self.text_dim, self.embed_dim)

        self.out_vision_proj = nn.Linear(self.embed_dim, self.vision_dim)
        self.out_text_proj = nn.Linear(self.embed_dim, self.text_dim)

    def _reshape(self, tensor: torch.Tensor, seq_len: int, batch_size: int):
        return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        vision_features: torch.FloatTensor,
        text_features: torch.FloatTensor,
        vision_attention_mask: Optional[torch.BoolTensor] = None,
        text_attention_mask: Optional[torch.BoolTensor] = None,
    ) -> Tuple[Tuple[torch.FloatTensor, torch.FloatTensor], Tuple[torch.FloatTensor, torch.FloatTensor]]:
        """Image-to-text and text-to-image cross-attention

        Args:

        Returns:

        """
        batch_size, vision_seq_len, _ = vision_features.size() #-> 1 17821
        _, text_seq_len, _ = text_features.size() #->9
        
        vision_query = self._reshape(self.vision_proj(vision_features) * self.scale, -1, batch_size) #->torch.Size([1, 4, 17821, 256]) 批 头 宽高 维
        vision_value = self._reshape(self.values_vision_proj(vision_features), -1, batch_size)
        text_key = self._reshape(self.text_proj(text_features), -1, batch_size) #->torch.Size([1, 4, 9, 256]) 批 头 宽高 维
        text_value = self._reshape(self.values_text_proj(text_features), -1, batch_size)

        # Reshape for multi-head attention
        proj_shape = (batch_size * self.num_heads, -1, self.head_dim)
        vision_query, vision_value = vision_query.view(*proj_shape), vision_value.view(*proj_shape) #torch.Size([4, 17821, 256]) 批头 宽高 维
        text_key, text_value = text_key.view(*proj_shape), text_value.view(*proj_shape) #torch.Size([4, 9, 256]) 批头 宽高 维


        # Compute attention weights
        vision_fused_attn = torch.bmm(vision_query, text_key.transpose(1, 2)) #torch.Size([4, 17821, 9])
        vision_fused_attn = torch.clamp(vision_fused_attn - vision_fused_attn.max(), min=-50000, max=50000)

        text_fused_attn = torch.clamp(
            vision_fused_attn.transpose(1, 2) - torch.max(vision_fused_attn.transpose(1, 2), dim=-1, keepdim=True)[0],
            min=-50000,
            max=50000,
        ) #->torch.Size([4, 9, 17821])

        # Apply masks
        if vision_attention_mask is not None:
            vision_attention_mask = vision_attention_mask[:, None, None, :].repeat(1, self.num_heads, 1, 1).flatten(0, 1)
            text_fused_attn.masked_fill_(vision_attention_mask, float("-inf"))
        text_fused_attn = text_fused_attn.softmax(dim=-1) #文字，你重点要关注哪些信息#->torch.Size([4, 9, 17821])
        if text_attention_mask is not None:
            text_attention_mask = text_attention_mask[:, None, None, :].repeat(1, self.num_heads, 1, 1).flatten(0, 1)
            vision_fused_attn.masked_fill_(text_attention_mask, float("-inf"))
        vision_fused_attn = vision_fused_attn.softmax(dim=-1) #->torch.Size([4, 17821, 9])


        # Normalize attention weights
        vision_attn_probs = F.dropout(vision_fused_attn.softmax(dim=-1), p=self.dropout, training=self.training)
        text_attn_probs = F.dropout(text_fused_attn.softmax(dim=-1), p=self.dropout, training=self.training)

        # Compute attention outputs
        vision_fused_attn_output = torch.bmm(vision_attn_probs, text_value) #torch.Size([4, 17821, 256])
        text_fused_attn_output = torch.bmm(text_attn_probs, vision_value) #->torch.Size([4, 9, 256])


        # Reshape and combine heads
        vision_fused_attn_output = self._combine_heads(vision_fused_attn_output, batch_size, vision_seq_len) #->torch.Size([1, 17821, 1024])
        text_fused_attn_output = self._combine_heads(text_fused_attn_output, batch_size, text_seq_len) #->torch.Size([1, 9, 1024])

        # Final linear projections (FFN)
        vision_fused_attn_output = self.out_vision_proj(vision_fused_attn_output)
        text_fused_attn_output = self.out_text_proj(text_fused_attn_output)


        return (vision_fused_attn_output, vision_fused_attn), (text_fused_attn_output, text_fused_attn)
    
    # Helper function to combine heads
    def _combine_heads(self, attn_output, batch_size, seq_len):
        attn_output = attn_output.view(batch_size, self.num_heads, seq_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
        return attn_output

# Copied from transformers.models.beit.modeling_beit.drop_path
def drop_path(input: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    Comment by Ross Wightman: This is the same as the DropConnect impl I created for EfficientNet, etc networks,
    however, the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the
    layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use 'survival rate' as the
    argument.
    """
    if drop_prob == 0.0 or not training:
        return input
    keep_prob = 1 - drop_prob
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    random_tensor.floor_()  # binarize
    output = input.div(keep_prob) * random_tensor
    return output


# Copied from transformers.models.beit.modeling_beit.BeitDropPath with Beit->GroundingDino
class GroundingDinoDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, query_embeds: torch.Tensor) -> torch.Tensor:
        return drop_path(query_embeds, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class GroundingDinoFusionLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        drop_path = config.fusion_droppath

        # pre layer norm
        self.layer_norm_vision = nn.LayerNorm(config.d_model, config.layer_norm_eps)
        self.layer_norm_text = nn.LayerNorm(config.d_model, config.layer_norm_eps)
        self.attn = GroundingDinoBiMultiHeadAttention(config)

        # add layer scale for training stability
        self.drop_path = GroundingDinoDropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        init_values = 1e-4
        self.vision_param = nn.Parameter(init_values * torch.ones((config.d_model)), requires_grad=True)
        self.text_param = nn.Parameter(init_values * torch.ones((config.d_model)), requires_grad=True)

    def forward(
        self,
        vision_features: torch.FloatTensor,
        text_features: torch.FloatTensor,
        vision_attention_mask: Optional[torch.BoolTensor] = None,
        text_attention_mask: Optional[torch.BoolTensor] = None,
    ) -> Tuple[Tuple[torch.FloatTensor, torch.FloatTensor], Tuple[torch.FloatTensor, torch.FloatTensor]]:
        """Image and text features fusion

        Args:
            vision_features (`torch.FloatTensor` of shape `(batch_size, vision_sequence_length, hidden_dim)`):
                Projected flattened image features generated by the vision backbone.
            text_features (`torch.FloatTensor` of shape `(batch_size, text_sequence_length, hidden_dim)`):
                Projected text features generated by the text encoder.
            vision_attention_mask (`torch.BoolTensor`, **optional**):
                Attention mask for image-to-text cross-attention. False for real tokens and True for padding tokens.
            text_attention_mask (`torch.BoolTensor`, **optional**):
                Attention mask for text-to-image cross-attention. False for real tokens and True for padding tokens.

        Returns:
            `tuple(tuple(torch.FloatTensor), tuple(torch.FloatTensor))` where each inner tuple comprises an enhanced
            feature and attention output and weights:
            - **vision_features** (`torch.FloatTensor` of shape `(batch_size, vision_sequence_length, vision_dim)`) --
                Updated vision features with attention output from image-to-text cross-attention layer.
            - **vision_fused_attn** (`torch.FloatTensor` of shape `(batch_size, num_heads, vision_sequence_length,
              vision_sequence_length)`) --
                Attention weights of the image-to-text cross-attention layer.
            - **text_features** (`torch.FloatTensor` of shape `(batch_size, text_sequence_length, text_dim)`) --
                Updated text features with attention output from text-to-image cross-attention layer.
            - **text_fused_attn** (`torch.FloatTensor` of shape `(batch_size, num_heads, text_sequence_length,
              text_sequence_length)`) --
                Attention weights of the text-to-image cross-attention layer.
        """
        """ 2. A Feature Enhancer Layer """
        vision_features = self.layer_norm_vision(vision_features) #层归一化
        text_features = self.layer_norm_text(text_features) #层归一化
        (vision_fused_attn_output, vision_fused_attn), (text_fused_attn_output, text_fused_attn) = self.attn(
            vision_features,
            text_features,
            vision_attention_mask=vision_attention_mask,
            text_attention_mask=text_attention_mask,
        )
        vision_features = vision_features + self.drop_path(self.vision_param * vision_fused_attn_output)
        text_features = text_features + self.drop_path(self.text_param * text_fused_attn_output)

        return (vision_features, vision_fused_attn), (text_features, text_fused_attn)

class GroundingDinoDeformableLayer(nn.Module):
    def __init__(self, config: GroundingDinoConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = GroundingDinoMultiscaleDeformableAttention(
            config, num_heads=config.encoder_attention_heads, n_points=config.encoder_n_points
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim, config.layer_norm_eps)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim, config.layer_norm_eps)

    def forward(
        self,
        vision_features: torch.Tensor,
        vision_attention_mask: torch.Tensor,
        vision_posi_embed: torch.Tensor = None,
        
        anchor_points=None,
        vision_map_shapes=None,
        level_start_index=None,
        output_attentions: bool = False,
    ):
        """
        Args: vision_deformable_attn

        """
        residual = vision_features

        # Apply Multi-scale Deformable Attention Module on the multi-scale feature maps.
        vision_features, vision_deformable_attn = self.self_attn(
            query_embeds=vision_features,
            query_posi_embeds=vision_posi_embed,
            expand_topk_coords=anchor_points,
            
            vision_features=vision_features,  
            vision_attention_mask=vision_attention_mask,


            vision_map_shapes=vision_map_shapes,
            level_start_index=level_start_index,
            output_attentions=output_attentions,
        )

        vision_features = nn.functional.dropout(vision_features, p=self.dropout, training=self.training)
        vision_features = residual + vision_features
        vision_features = self.self_attn_layer_norm(vision_features)

        residual = vision_features
        vision_features = self.activation_fn(self.fc1(vision_features))
        vision_features = nn.functional.dropout(vision_features, p=self.activation_dropout, training=self.training)

        vision_features = self.fc2(vision_features)
        vision_features = nn.functional.dropout(vision_features, p=self.dropout, training=self.training)

        vision_features = residual + vision_features
        vision_features = self.final_layer_norm(vision_features)

        if self.training:
            if torch.isinf(vision_features).any() or torch.isnan(vision_features).any():
                clamp_value = torch.finfo(vision_features.dtype).max - 1000
                vision_features = torch.clamp(vision_features, min=-clamp_value, max=clamp_value)

        return vision_features, vision_deformable_attn


# Based on https://github.com/IDEA-Research/GroundingDINO/blob/2b62f419c292ca9c518daae55512fabc3fead4a4/groundingdino/models/GroundingDINO/utils.py#L24
def get_sine_pos_embed(
    pos_tensor: torch.Tensor, num_pos_feats: int = 128, temperature: int = 10000, exchange_xy: bool = True
) -> Tensor:
    """
    Generate sine position embeddings from a position tensor.

    Args:
        pos_tensor (torch.Tensor):
            Tensor containing positions. Shape: [..., n].
        num_pos_feats (`int`, *optional*, defaults to 128):
            Projected shape for each float in the tensor.
        temperature (`int`, *optional*, defaults to 10000):
            Temperature in the sine/cosine function.
        exchange_xy (`bool`, *optional*, defaults to `True`):
            Exchange pos x and pos y. For example, input tensor is [x,y], the results will be [pos(y), pos(x)].

    Returns:
        posi_embed (torch.Tensor): shape: [..., n * hidden_size].
    """
    scale = 2 * math.pi
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos_tensor.device)
    dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / num_pos_feats)

    def sine_func(x: torch.Tensor):
        sin_x = x * scale / dim_t
        sin_x = torch.stack((sin_x[..., 0::2].sin(), sin_x[..., 1::2].cos()), dim=3).flatten(2)
        return sin_x

    pos_tensor = pos_tensor.split([1] * pos_tensor.shape[-1], dim=-1)
    posi_embed = [sine_func(x) for x in pos_tensor]
    if exchange_xy:
        posi_embed[0], posi_embed[1] = posi_embed[1], posi_embed[0]
    posi_embed = torch.cat(posi_embed, dim=-1)
    return posi_embed


class GroundingDinoEncoderLayer(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        self.d_model = config.d_model

        self.text_enhancer_layer = GroundingDinoTextEnhancerLayer(config)
        self.fusion_layer = GroundingDinoFusionLayer(config)
        self.deformable_layer = GroundingDinoDeformableLayer(config)

    def get_text_position_embeddings(
        self,
        text_features: Tensor,
        text_posi_embed: Optional[torch.Tensor],
        text_position_ids: Optional[torch.Tensor],
    ) -> Tensor:
        batch_size, seq_length, _ = text_features.shape
        if text_posi_embed is None and text_position_ids is None:
            text_posi_embed = torch.arange(seq_length, device=text_features.device)
            text_posi_embed = text_posi_embed.float()
            text_posi_embed = text_posi_embed.unsqueeze(0).unsqueeze(-1)
            text_posi_embed = text_posi_embed.repeat(batch_size, 1, 1)
            text_posi_embed = get_sine_pos_embed(
                text_posi_embed, num_pos_feats=self.d_model, exchange_xy=False
            )
        if text_position_ids is not None:
            text_posi_embed = get_sine_pos_embed(
                text_position_ids[..., None], num_pos_feats=self.d_model, exchange_xy=False
            )

        return text_posi_embed #根据单词的位置加入位置编码 torch.Size([1, 9, 256])



                
    def forward(
        self,
        vision_features, #->torch.Size([1, 17821, 256])
        vision_attention_mask, #->torch.Size([1, 17821])
        vision_posi_embed, #->torch.Size([1, 17821, 256])
        
        vision_map_shapes, #torch.Size([4, 2])
        level_start_index, #torch.Size([4]) tensor([    0, 13400, 16750, 17600], device='cuda:0')
        anchor_points, #torch.Size([1, 17821, 4, 2])
        
        text_features, #torch.Size([1, 9, 256])
        text_attention_mask, #torch.Size([1, 9])
        text_posi_embed, #None
        
        text_self_attention_mask, #torch.Size([1, 9, 9])
        text_position_ids, #torch.Size([1, 9])
    ):
        text_posi_embed = self.get_text_position_embeddings( #=>torch.Size([1, 9, 256])
            text_features, text_posi_embed, text_position_ids
        ) #位置编码torch.Size([1, 9, 256])
        """ 特征融合 """
        (vision_features, vision_fused_attn), (text_features, text_fused_attn) = self.fusion_layer(
            vision_features=vision_features,
            text_features=text_features,
            vision_attention_mask=vision_attention_mask,
            text_attention_mask=text_attention_mask,
        )
        
        """ 文本特征增强 """
        (text_features, text_enhanced_attn) = self.text_enhancer_layer(
            text_features=text_features,
            text_self_attention_mask=~text_self_attention_mask,  # note we use ~ for mask here
            text_posi_embed=(text_posi_embed if text_posi_embed is not None else None),
        )

        """ 视觉特征增强 """
        (vision_features, vision_deformable_attn) = self.deformable_layer(
            vision_features=vision_features,
            vision_attention_mask=~vision_attention_mask,
            vision_posi_embed=vision_posi_embed, #视觉位置嵌入
            
            anchor_points=anchor_points,
            vision_map_shapes=vision_map_shapes,
            level_start_index=level_start_index,
        )

        return (
            (vision_features, text_features),
            (vision_fused_attn, text_fused_attn, text_enhanced_attn, vision_deformable_attn),
        )


class GroundingDinoMultiheadAttention(nn.Module):
    """Equivalent implementation of nn.MultiheadAttention with `batch_first=True`."""

    def __init__(self, config, num_attention_heads=None):
        super().__init__()
        if config.hidden_size % num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({num_attention_heads})"
            )

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(config.hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)

        self.dropout = nn.Dropout(config.attention_dropout)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        query_layer = self.transpose_for_scores(self.query(queries)) #torch.Size([1, 4, 9, 64])
        key_layer = self.transpose_for_scores(self.key(keys)) #torch.Size([1, 4, 9, 64])
        value_layer = self.transpose_for_scores(self.value(values)) #torch.Size([1, 4, 9, 64])

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) #torch.Size([1, 4, 9, 9])

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in GroundingDinoModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        context_layer = self.out_proj(context_layer)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


class GroundingDinoDecoderLayer(nn.Module):
    def __init__(self, config: GroundingDinoConfig):
        super().__init__()
        self.embed_dim = config.d_model

        # self-attention
        self.self_attn = GroundingDinoMultiheadAttention(config, num_attention_heads=config.decoder_attention_heads)

        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim, config.layer_norm_eps)
        # cross-attention text
        self.encoder_attn_text = GroundingDinoMultiheadAttention(
            config, num_attention_heads=config.decoder_attention_heads
        )
        self.encoder_attn_text_layer_norm = nn.LayerNorm(self.embed_dim, config.layer_norm_eps)
        # cross-attention
        self.encoder_attn = GroundingDinoMultiscaleDeformableAttention(
            config,
            num_heads=config.decoder_attention_heads,
            n_points=config.decoder_n_points,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim, config.layer_norm_eps)
        # feedforward neural networks
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim, config.layer_norm_eps)

    def with_pos_embed(self, tensor: torch.Tensor, query_posi_embeds: Optional[Tensor]):
        return tensor if query_posi_embeds is None else tensor + query_posi_embeds

    def forward(
        self,
        query_embeds,
        query_posi_embeds,
        expand_topk_coords,

        encoder_vision_features,
        vision_attention_mask,
        encoder_text_features,
        text_attention_mask,

        vision_map_shapes,
        level_start_index,
        output_attentions,
        self_attn_mask,
    ):
        first_residual = query_embeds

        """ 自注意力 """
        queries = keys = self.with_pos_embed(query_embeds, query_posi_embeds)
        query_embeds, query_self_attn = self.self_attn(
            queries=queries,
            keys=keys,
            values=query_embeds,
            attention_mask=self_attn_mask,
            output_attentions=True,
        )

        query_embeds = nn.functional.dropout(query_embeds, p=self.dropout, training=self.training)
        query_embeds = first_residual + query_embeds
        query_embeds = self.self_attn_layer_norm(query_embeds)

        second_residual = query_embeds

        """ 查询文本交叉注意力 """
        queries = self.with_pos_embed(query_embeds, query_posi_embeds)
        query_embeds, query2text_cross_attn = self.encoder_attn_text(
            queries=queries,
            keys=encoder_text_features,
            values=encoder_text_features,
            attention_mask=text_attention_mask,
            output_attentions=True,
        )

        query_embeds = nn.functional.dropout(query_embeds, p=self.dropout, training=self.training)
        query_embeds = second_residual + query_embeds
        query_embeds = self.encoder_attn_text_layer_norm(query_embeds)

        third_residual = query_embeds

        """  查询视觉交叉注意力 """
        query2vision_cross_attn = None
        query_embeds, query2vision_cross_attn = self.encoder_attn(
            query_embeds=query_embeds,
            query_posi_embeds=query_posi_embeds,
            expand_topk_coords=expand_topk_coords,

            vision_features=encoder_vision_features,
            vision_attention_mask=vision_attention_mask,


            vision_map_shapes=vision_map_shapes,
            level_start_index=level_start_index,
            output_attentions=output_attentions,
        )

        query_embeds = nn.functional.dropout(query_embeds, p=self.dropout, training=self.training)
        query_embeds = third_residual + query_embeds
        query_embeds = self.encoder_attn_layer_norm(query_embeds)

        # Fully Connected
        residual = query_embeds
        query_embeds = self.activation_fn(self.fc1(query_embeds))
        query_embeds = nn.functional.dropout(query_embeds, p=self.activation_dropout, training=self.training)
        query_embeds = self.fc2(query_embeds)
        query_embeds = nn.functional.dropout(query_embeds, p=self.dropout, training=self.training)
        query_embeds = residual + query_embeds
        query_embeds = self.final_layer_norm(query_embeds)

        outputs = (query_embeds,)

        if output_attentions:
            outputs += (query_self_attn, query2text_cross_attn, query2vision_cross_attn)

        return outputs


class GroundingDinoContrastiveEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.max_text_len = config.max_text_len

    def forward(self, vision_hidden_state, text_hidden_state, attention_mask):
        # Compute similarity between vision and text hidden states
        output = vision_hidden_state @ text_hidden_state.transpose(-1, -2) #->torch.size([1, 900, 9])
        output = output.masked_fill(~attention_mask[:, None, :], float("-inf"))

        # Pad output to max_text_len
        padded_output = torch.full(
            (*output.shape[:-1], self.max_text_len), float("-inf"), device=output.device
        )
        padded_output[..., : output.shape[-1]] = output

        return padded_output #->torch.Size([1, 900, 256])


class GroundingDinoPreTrainedModel(PreTrainedModel):
    config_class = GroundingDinoConfig
    base_model_prefix = "model"
    main_input_name = "pixel_values"

    def _init_weights(self, module):
        std = self.config.init_std

        if isinstance(module, GroundingDinoLearnedPositionEmbedding):
            nn.init.uniform_(module.row_embeddings.weight)
            nn.init.uniform_(module.column_embeddings.weight)
        elif isinstance(module, GroundingDinoMultiscaleDeformableAttention):
            nn.init.constant_(module.sampling_offsets.weight.data, 0.0)
            default_dtype = torch.get_default_dtype()
            thetas = torch.arange(module.n_heads, dtype=torch.int64).to(default_dtype) * (
                2.0 * math.pi / module.n_heads
            )
            grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
            grid_init = (
                (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
                .view(module.n_heads, 1, 1, 2)
                .repeat(1, module.n_levels, module.n_points, 1)
            )
            for i in range(module.n_points):
                grid_init[:, :, i, :] *= i + 1
            with torch.no_grad():
                module.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
            nn.init.constant_(module.attention_weights.weight.data, 0.0)
            nn.init.constant_(module.attention_weights.bias.data, 0.0)
            nn.init.xavier_uniform_(module.value_proj.weight.data)
            nn.init.constant_(module.value_proj.bias.data, 0.0)
            nn.init.xavier_uniform_(module.output_proj.weight.data)
            nn.init.constant_(module.output_proj.bias.data, 0.0)
        elif isinstance(module, GroundingDinoBiMultiHeadAttention):
            nn.init.xavier_uniform_(module.vision_proj.weight)
            module.vision_proj.bias.data.fill_(0)
            nn.init.xavier_uniform_(module.text_proj.weight)
            module.text_proj.bias.data.fill_(0)
            nn.init.xavier_uniform_(module.values_vision_proj.weight)
            module.values_vision_proj.bias.data.fill_(0)
            nn.init.xavier_uniform_(module.values_text_proj.weight)
            module.values_text_proj.bias.data.fill_(0)
            nn.init.xavier_uniform_(module.out_vision_proj.weight)
            module.out_vision_proj.bias.data.fill_(0)
            nn.init.xavier_uniform_(module.out_text_proj.weight)
            module.out_text_proj.bias.data.fill_(0)
        elif isinstance(module, (GroundingDinoEncoderLayer, GroundingDinoDecoderLayer)):
            for p in module.parameters():
                if p.dim() > 1:
                    nn.init.normal_(p, mean=0.0, std=std)
        elif isinstance(module, (nn.Linear, nn.Conv2d, nn.BatchNorm2d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, GroundingDinoMLPPredictionHead):
            nn.init.constant_(module.layers[-1].weight.data, 0)
            nn.init.constant_(module.layers[-1].bias.data, 0)

        if hasattr(module, "anchor_points") and not self.config.two_stage:
            nn.init.xavier_uniform_(module.anchor_points.weight.data, gain=1.0)
            nn.init.constant_(module.anchor_points.bias.data, 0.0)
        if hasattr(module, "level_embed"):
            nn.init.normal_(module.level_embed)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, GroundingDinoDecoder):
            module.gradient_checkpointing = value


GROUNDING_DINO_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`GroundingDinoConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

GROUNDING_DINO_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Padding will be ignored by default should you provide it.

            Pixel values can be obtained using [`AutoImageProcessor`]. See [`GroundingDinoImageProcessor.__call__`] for
            details.

        input_ids (`torch.LongTensor` of shape `(batch_size, text_sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`BertTokenizer.__call__`] for details.

        token_type_ids (`torch.LongTensor` of shape `(batch_size, text_sequence_length)`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`: 0 corresponds to a `sentence A` token, 1 corresponds to a `sentence B` token

            [What are token type IDs?](../glossary#token-type-ids)

        attention_mask (`torch.LongTensor` of shape `(batch_size, text_sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are real (i.e. **not masked**),
            - 0 for tokens that are padding (i.e. **masked**).

            [What are attention masks?](../glossary#attention-mask)

        pixel_mask (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*):
            Mask to avoid performing attention on padding pixel values. Mask values selected in `[0, 1]`:

            - 1 for pixels that are real (i.e. **not masked**),
            - 0 for pixels that are padding (i.e. **masked**).

            [What are attention masks?](../glossary#attention-mask)

        encoder_outputs (`tuple(tuple(torch.FloatTensor)`, *optional*):
            Tuple consists of (`encoder_vision_features_states`, *optional*: `encoder_text_features_states`, *optional*:
            `vision_hidden_states`, *optional*: `text_hidden_states`, *optional*: `attentions`)
            `encoder_vision_features_states` of shape `(batch_size, vision_seq_len, hidden_size)`, *optional*) is a sequence
            of hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the
            decoder.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `query_embeds` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
"""


class GroundingDinoEncoder(GroundingDinoPreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* deformable attention layers. Each layer is a
    [`GroundingDinoEncoderLayer`].

    The encoder updates the flattened multi-scale feature maps through multiple deformable attention layers.

    Args:
        config: GroundingDinoConfig
    """

    def __init__(self, config: GroundingDinoConfig):
        super().__init__(config)

        self.dropout = config.dropout
        self.layers = nn.ModuleList([GroundingDinoEncoderLayer(config) for _ in range(config.encoder_layers)])

        # Initialize weights and apply final processing
        self.post_init()

    @staticmethod
    def generate_anchor_points(vision_map_shapes, valid_ratios, device):

        anchor_points = []
        for level, (height, width) in enumerate(vision_map_shapes):
            # Create a meshgrid for the current feature map level
            grid_y, grid_x = torch.meshgrid(
                torch.linspace(0.5, height - 0.5, height, device=device),
                torch.linspace(0.5, width - 0.5, width, device=device),
                indexing="ij"
            )

            # Normalize the grid coordinates based on valid ratios and spatial dimensions
            grid_y = grid_y.reshape(-1)[None, :] / (valid_ratios[:, level, 1].unsqueeze(1) * height)
            grid_x = grid_x.reshape(-1)[None, :] / (valid_ratios[:, level, 0].unsqueeze(1) * width)

            # Stack the normalized coordinates and append to the list
            ref = torch.stack((grid_x, grid_y), dim=-1)  # Shape: (batch_size, num_queries, 2)
            anchor_points.append(ref)

        # Concatenate reference points from all levels
        anchor_points = torch.cat(anchor_points, dim=1)  # Shape: (batch_size, num_queries_total, 2)

        # Reshape and scale by valid ratios to get the final reference points
        anchor_points = anchor_points[:, :, None] * valid_ratios[:, None]  # Shape: (batch_size, num_queries_total, num_feature_levels, 2)

        return anchor_points





    def forward(
        self,
        vision_features: Tensor,
        vision_attention_mask: Tensor,
        vision_posi_embed: Tensor,
        
        vision_map_shapes: Tensor,
        level_start_index: Tensor,
        valid_ratios: Optional[Tensor] = None,
        
        text_features: Optional[Tensor] = None,
        text_attention_mask: Optional[Tensor] = None,
        text_posi_embed: Optional[Tensor] = None,
        
        text_self_attention_mask: Optional[Tensor] = None,
        text_position_ids: Optional[Tensor] = None,

        return_dict: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,

    ):
        r"""
        Args:

        """
        # Set default configuration values if arguments are None
        return_dict = return_dict or self.config.use_return_dict
        output_attentions = output_attentions or self.config.output_attentions
        output_hidden_states = output_hidden_states or self.config.output_hidden_states

        # Generate reference points based on spatial shapes and valid ratios
        anchor_points = self.generate_anchor_points(vision_map_shapes, valid_ratios, device=vision_features.device) #torch.Size([4, 2]) torch.Size([1, 4, 2]) cuda


        encoder_attentions = () if output_attentions else None
        text_fused_attn = () if output_attentions else None
        vision_fused_attn = () if output_attentions else None
        text_enhanced_attn = () if output_attentions else None
        vision_deformable_attn = () if output_attentions else None
        
        vision_hidden_states = () if output_hidden_states else None
        text_hidden_states = () if output_hidden_states else None
        for i, layers_i in enumerate(self.layers):
            """ 无需了解编码器中的隐藏层状态 """
            if output_hidden_states:
                vision_hidden_states += (vision_features,)
                text_hidden_states += (text_features,)

            (vision_features, text_features), attentions = layers_i(
                vision_features=vision_features, #torch.Size([1, 17821, 256])
                vision_attention_mask=vision_attention_mask, #torch.Size([1, 17821])
                vision_posi_embed=vision_posi_embed, #torch.Size([1, 17821, 256])
                
                vision_map_shapes=vision_map_shapes, #torch.Size([4, 2])
                level_start_index=level_start_index, #torch.Size([4])
                anchor_points=anchor_points, #torch.Size([1, 17821, 4, 2])
                
                text_features=text_features, #torch.Size([1, 9, 256])
                text_attention_mask=text_attention_mask, #torch.Size([1, 9])
                text_posi_embed=text_posi_embed, #None
                
                text_self_attention_mask=text_self_attention_mask, #torch.Size([1, 9, 9])
                text_position_ids=text_position_ids, #torch.Size([1, 9])
            )
            """ 无需了解编码器中的注意力状态 """
            if output_attentions:
                vision_fused_attn += (attentions[0],)
                text_fused_attn += (attentions[1],)
                text_enhanced_attn += (attentions[2],)
                vision_deformable_attn += (attentions[3],)

        if output_hidden_states:
            vision_hidden_states += (vision_features,)
            text_hidden_states += (text_features,)

        if output_attentions:
            encoder_attentions = (vision_fused_attn, text_fused_attn, text_enhanced_attn, vision_deformable_attn)

        if not return_dict:
            enc_outputs = [vision_features, text_features, vision_hidden_states, text_hidden_states, encoder_attentions]
            return tuple(v for v in enc_outputs if v is not None)
        return GroundingDinoEncoderOutput(
            encoder_vision_features_states=vision_features,
            encoder_text_features_states=text_features,

            encoder_vision_hidden_states=vision_hidden_states,
            encoder_text_hidden_states=text_hidden_states,
            encoder_attentions=encoder_attentions,
        )




class GroundingDinoDecoder(GroundingDinoPreTrainedModel):
    """
    Args:
        config: GroundingDinoConfig
    """

    def __init__(self, config: GroundingDinoConfig):
        super().__init__(config)

        self.dropout = config.dropout
        self.layer_norm = nn.LayerNorm(config.d_model, config.layer_norm_eps)
        self.layers = nn.ModuleList([GroundingDinoDecoderLayer(config) for _ in range(config.decoder_layers)])
        self.reference_points_head = GroundingDinoMLPPredictionHead(
            config.query_dim // 2 * config.d_model, config.d_model, config.d_model, 2
        )
        self.gradient_checkpointing = False

        # hack implementation for iterative bounding box refinement as in two-stage Deformable DETR
        self.bbox_embed = None
        self.class_embed = None
        self.query_scale = None

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        query_embeds,
        topk_coords=None,
        #encoder_vision_features
        encoder_vision_features=None,
        vision_attention_mask=None ,
        encoder_text_features=None,
        text_attention_mask=None,

        vision_map_shapes=None,
        level_start_index=None,
        valid_ratios=None,

        return_dict=None,
        output_attentions=None,
        output_hidden_states=None,

        self_attn_mask=None,
    ):
        r"""
        Args:
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
        """
        return_dict = return_dict or self.config.use_return_dict
        output_attentions = output_attentions or self.config.output_attentions
        output_hidden_states = output_hidden_states or self.config.output_hidden_states


        # decoder layers
        query_embeds_hidden = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        decoder_attentions = () if output_attentions else None
        all_cross_attns_vision = () if (output_attentions and encoder_vision_features is not None) else None
        all_cross_attns_text = () if (output_attentions and encoder_text_features is not None) else None
        query_embeds_hidden = ()
        refine_coords_hidden = ()

        if text_attention_mask is not None:
            dtype = encoder_text_features.dtype

            text_attention_mask = text_attention_mask[:, None, None, :]
            text_attention_mask = text_attention_mask.repeat(
                1, self.config.decoder_attention_heads, self.config.num_queries, 1
            ) #->torch.Size([1, 8, 900, 9])
            text_attention_mask = text_attention_mask.to(dtype=dtype)
            text_attention_mask = text_attention_mask * torch.finfo(dtype).min

        for idx, decoder_layer in enumerate(self.layers):
            coords_dim = topk_coords.shape[-1]
            if coords_dim == 4:
                expand_topk_coords = (
                    topk_coords[:, :, None] * torch.cat([valid_ratios, valid_ratios], -1)[:, None]
                )
            elif coords_dim == 2:
                expand_topk_coords = topk_coords[:, :, None] * valid_ratios[:, None]
            else:
                raise ValueError("Last dim of topk_coords must be 2 or 4, but got {topk_coords.shape[-1]}")
            embed_posi_topk_coords = get_sine_pos_embed(expand_topk_coords[:, :, 0, :], num_pos_feats=self.config.d_model // 2)
            query_posi_embeds = self.reference_points_head(embed_posi_topk_coords) #->torch.Size([1, 900, 256])


            if output_hidden_states:
                query_embeds_hidden += (self.layer_norm(query_embeds),)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    query_embeds,
                    query_posi_embeds,
                    expand_topk_coords,
                    vision_map_shapes,
                    level_start_index,
                    encoder_vision_features,
                    vision_attention_mask,
                    encoder_text_features,
                    text_attention_mask,
                    self_attn_mask,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    query_embeds=query_embeds,
                    query_posi_embeds=query_posi_embeds,
                    expand_topk_coords=expand_topk_coords,

                    encoder_vision_features=encoder_vision_features,
                    vision_attention_mask=vision_attention_mask,
                    encoder_text_features=encoder_text_features,
                    text_attention_mask=text_attention_mask,

                    vision_map_shapes=vision_map_shapes,
                    level_start_index=level_start_index,
                    output_attentions=output_attentions,
                    self_attn_mask=self_attn_mask,
                )

            query_embeds = layer_outputs[0]

            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[idx](query_embeds)
                coords_dim = topk_coords.shape[-1]
                if coords_dim == 4:
                    refine_coords = tmp + torch.special.logit(topk_coords, eps=1e-5)
                    refine_coords = refine_coords.sigmoid()
                elif coords_dim == 2:
                    refine_coords = tmp
                    refine_coords[..., :2] = tmp[..., :2] + torch.special.logit(topk_coords, eps=1e-5)
                    refine_coords = refine_coords.sigmoid()
                else:
                    raise ValueError(
                        f"Last dim of topk_coords must be 2 or 4, but got {topk_coords.shape[-1]}"
                    )
                refine_coords = refine_coords.detach()

            query_embeds_hidden += (self.layer_norm(query_embeds),)
            refine_coords_hidden += (refine_coords,)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_text_features is not None:
                    all_cross_attns_text += (layer_outputs[2],)

                if encoder_vision_features is not None:
                    all_cross_attns_vision += (layer_outputs[3],)

        # Keep batch_size as first dimension
        query_embeds_hidden = torch.stack(query_embeds_hidden, dim=1)
        refine_coords_hidden = torch.stack(refine_coords_hidden, dim=1)
        query_embeds = self.layer_norm(query_embeds)


        if output_attentions:
            decoder_attentions += (all_self_attns, all_cross_attns_text, all_cross_attns_vision)

        if not return_dict:
            return tuple(
                v
                for v in [
                    query_embeds,
                    query_embeds_hidden,
                    refine_coords_hidden,
                    decoder_attentions,
                ]
                if v is not None
            )
        return GroundingDinoDecoderOutput(
            decoder_query_embeds_states=query_embeds,
            decoder_query_embeds_hidden_states=query_embeds_hidden,
            decoder_refine_coords_hidden_states=refine_coords_hidden,
            decoder_attentions=decoder_attentions,
        )


# these correspond to [CLS], [SEP], . and ?
SPECIAL_TOKENS = [101, 102, 1012, 1029]


def generate_text_self_attention_masks(input_ids: torch.LongTensor) -> Tuple[torch.Tensor, torch.Tensor]:

    batch_size, seq_len = input_ids.shape

    # Identify positions of special tokens
    special_tokens_mask = torch.isin(input_ids, torch.tensor(SPECIAL_TOKENS, device=input_ids.device))
    special_tokens_posi = special_tokens_mask.nonzero(as_tuple=False)  # Shape: (num_special_tokens, 2)

    # Initialize attention mask and position IDs
    text_self_attention_mask = torch.eye(seq_len, device=input_ids.device).bool().unsqueeze(0).repeat(batch_size, 1, 1)
    position_ids = torch.zeros((batch_size, seq_len), device=input_ids.device, dtype=torch.long)

    # Generate attention mask and position IDs based on special tokens
    previous_col = 0
    for row, col in special_tokens_posi: #哪一句话的哪一个位置
        if col in (0, seq_len - 1):  # 在这两个为中国i上的话
            text_self_attention_mask[row, col, col] = True
            position_ids[row, col] = 0
        else:
            text_self_attention_mask[row, previous_col + 1 : col + 1, previous_col + 1 : col + 1] = True
            position_ids[row, previous_col + 1 : col + 1] = torch.arange(0, col - previous_col, device=input_ids.device)

        previous_col = col

    return text_self_attention_mask, position_ids
 #text_self_attention_mask

@add_start_docstrings(
    """
    The bare Grounding DINO Model (consisting of a backbone and encoder-decoder Transformer) outputting raw
    hidden-states without any specific head on top.
    """,
    GROUNDING_DINO_START_DOCSTRING,
)
class GroundingDinoModel(GroundingDinoPreTrainedModel):
    def __init__(self, config: GroundingDinoConfig):
        super().__init__(config)

        # Create backbone + positional encoding
        backbone = GroundingDinoConvEncoder(config)
        posi_embed = build_position_encoding(config)
        self.backbone = GroundingDinoConvModel(backbone, posi_embed)

        # Create input projection layers
        if config.num_feature_levels > 1:
            num_backbone_outs = len(backbone.intermediate_channel_sizes)
            input_proj_list = []
            for i in range(num_backbone_outs):
                in_channels = backbone.intermediate_channel_sizes[i]
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, config.d_model, kernel_size=1),
                        nn.GroupNorm(32, config.d_model),
                    )
                )
            for _ in range(config.num_feature_levels - num_backbone_outs):
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, config.d_model, kernel_size=3, stride=2, padding=1),
                        nn.GroupNorm(32, config.d_model),
                    )
                )
                in_channels = config.d_model
            self.input_proj_vision = nn.ModuleList(input_proj_list)
        else:
            self.input_proj_vision = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(backbone.intermediate_channel_sizes[-1], config.d_model, kernel_size=1),
                        nn.GroupNorm(32, config.d_model),
                    )
                ]
            )

        # Create text backbone
        self.text_backbone = AutoModel.from_config(
            config.text_config, add_pooling_layer=False, attn_implementation=config._attn_implementation
        )
        self.text_projection = nn.Linear(config.text_config.hidden_size, config.d_model)

        if config.embedding_init_target or not config.two_stage:
            self.query_position_embeddings = nn.Embedding(config.num_queries, config.d_model)

        self.encoder = GroundingDinoEncoder(config)
        self.decoder = GroundingDinoDecoder(config)

        self.level_embed = nn.Parameter(torch.Tensor(config.num_feature_levels, config.d_model))

        if config.two_stage:
            self.enc_output = nn.Linear(config.d_model, config.d_model)
            self.enc_output_norm = nn.LayerNorm(config.d_model, config.layer_norm_eps)
            if (
                config.two_stage_bbox_embed_share
                and config.decoder_bbox_embed_share
                and self.decoder.bbox_embed is not None
            ):
                self.encoder_output_bbox_embed = self.decoder.bbox_embed
            else:
                self.encoder_output_bbox_embed = GroundingDinoMLPPredictionHead(
                    input_dim=config.d_model, hidden_dim=config.d_model, output_dim=4, num_layers=3
                )

            self.encoder_output_class_embed = GroundingDinoContrastiveEmbedding(config)
        else:
            self.topk_coords_embed = nn.Embedding(config.num_queries, 4)

        self.post_init()

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def freeze_backbone(self):
        for name, param in self.backbone.conv_encoder.model.named_parameters():
            param.requires_grad_(False)

    def unfreeze_backbone(self):
        for name, param in self.backbone.conv_encoder.model.named_parameters():
            param.requires_grad_(True)

    def get_valid_ratio(self, mask):
        """Get the valid ratio of all feature maps."""

        _, height, width = mask.shape
        valid_height = torch.sum(mask[:, :, 0], 1)
        valid_width = torch.sum(mask[:, 0, :], 1)
        valid_ratio_heigth = valid_height.float() / height
        valid_ratio_width = valid_width.float() / width
        valid_ratio = torch.stack([valid_ratio_width, valid_ratio_heigth], -1)
        return valid_ratio

    def encoder_query_proposals(self, enc_output, vision_attention_mask, vision_map_shapes):
        batch_size = enc_output.shape[0]
        proposals = []
        current_position = 0

        for level, (height, width) in enumerate(vision_map_shapes):
            # Reshape and process the attention mask for the current level
            lvl_vision_attention_mask = vision_attention_mask[:, current_position : current_position + height * width]
            lvl_vision_attention_mask = lvl_vision_attention_mask.view(batch_size, height, width, 1)

            # Calculate valid height and width based on the mask
            valid_height = torch.sum(~lvl_vision_attention_mask[:, :, 0, 0], dim=1)
            valid_width = torch.sum(~lvl_vision_attention_mask[:, 0, :, 0], dim=1)

            # Generate a normalized grid for the current feature map level
            grid_y, grid_x = meshgrid(
                torch.linspace(0, height - 1, height, dtype=torch.float32, device=enc_output.device),
                torch.linspace(0, width - 1, width, dtype=torch.float32, device=enc_output.device),
                indexing="ij",
            )
            lvl_grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], dim=-1)

            # Scale and normalize the lvl_grid
            lvl_scale = torch.cat([valid_width.unsqueeze(-1), valid_height.unsqueeze(-1)], dim=1).view(batch_size, 1, 1, 2)
            lvl_grid = (lvl_grid.unsqueeze(0).expand(batch_size, -1, -1, -1) + 0.5) / lvl_scale

            # Calculate proposal widths and heights
            width_height = torch.ones_like(lvl_grid) * 0.05 * (2.0 ** level)
            proposal = torch.cat((lvl_grid, width_height), dim=-1).view(batch_size, -1, 4)
            proposals.append(proposal)

            current_position += height * width

        # Concatenate proposals from all levels
        valid_proposals = torch.cat(proposals, dim=1)

        # Filter proposals based on validity
        output_proposals_valid = ((valid_proposals > 0.01) & (valid_proposals < 0.99)).all(dim=-1, keepdim=True)
        valid_proposals = torch.log(valid_proposals / (1 - valid_proposals))  # Apply inverse sigmoid
        valid_proposals = valid_proposals.masked_fill(vision_attention_mask.unsqueeze(-1), float("inf"))
        valid_proposals = valid_proposals.masked_fill(~output_proposals_valid, float("inf"))

        # Assign each pixel as an object query
        valid_object_query = enc_output
        valid_object_query = valid_object_query.masked_fill(vision_attention_mask.unsqueeze(-1), 0.0)
        valid_object_query = valid_object_query.masked_fill(~output_proposals_valid, 0.0)
        valid_object_query = self.enc_output_norm(self.enc_output(valid_object_query))

        return valid_object_query, valid_proposals


    def _handle_params(
        self,
        return_dict: Optional[bool],
        output_attentions: Optional[bool],
        output_hidden_states: Optional[bool],
    ) -> Tuple[bool, bool, bool]:
        return_dict = return_dict or self.config.use_return_dict
        output_attentions = output_attentions or self.config.output_attentions
        output_hidden_states = output_hidden_states or self.config.output_hidden_states
        return return_dict, output_attentions, output_hidden_states
    
    
    def _prepare_text_inputs(
        self,
        input_ids,
        token_type_ids,
        text_attention_mask,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        # Handle token_type_ids
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)   
        # Handle text_attention_mask
        if text_attention_mask is None:
            text_attention_mask = torch.ones_like(input_ids)
        # Create text token mask
        text_attention_mask = text_attention_mask.bool()

        # Generate masks and position IDs
        text_self_attention_mask, position_ids = generate_text_self_attention_masks(input_ids) #torch.Size([1, 9])


        # Trim inputs if exceeding max_text_len
        max_text_len = self.config.max_text_len
        if text_self_attention_mask.shape[1] > max_text_len:
            input_ids = input_ids[:, :max_text_len]
            token_type_ids = token_type_ids[:, :max_text_len]
            text_attention_mask = text_attention_mask[:, :max_text_len]
            text_self_attention_mask = text_self_attention_mask[:, :max_text_len, :max_text_len]
            position_ids = position_ids[:, :max_text_len]
            
        return (
            input_ids, 
            token_type_ids, #哪句话
            text_attention_mask, # 好像都值得关注
            
            text_self_attention_mask, #大矩阵
            position_ids, #一句话的分隔
        )

    def _extract_text_features(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        
        text_self_attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        # Extract text features from text backbone
        text_outputs = self.text_backbone( #使用bert进行文本特征提取
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            
            attention_mask=text_self_attention_mask,
            position_ids=position_ids,
            return_dict=True,
        )
        text_features = text_outputs.last_hidden_state #torch.Size([1, 9, 768])
        text_features = self.text_projection(text_features)  #torch.Size([1, 9, 256])
        return text_features



    def _extract_vision_features(
        self,
        pixel_values: torch.Tensor,
        pixel_mask: Optional[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        batch_size, _, height, width = pixel_values.shape #torch.Size([1, 3, 800, 1066])
        device = pixel_values.device

        if pixel_mask is None: #torch.Size([1, 800, 1066])
            pixel_mask = torch.ones((batch_size, height, width), dtype=torch.long, device=device)

        feature_maps, vision_posi_embed = self.backbone(pixel_values, pixel_mask)

        # 命名规范 特征图将层级 输出讲隐藏层
        vision_features = []
        vision_attention_mask = []
        for level, (feature_maps_i, feature_maps_mask_i) in enumerate(feature_maps):
            vision_features.append(self.input_proj_vision[level](feature_maps_i))
            vision_attention_mask.append(feature_maps_mask_i)

        """ 这一步想到于手动加了一层 """
        if self.config.num_feature_levels > len(vision_features):
            for level in range(len(vision_features), self.config.num_feature_levels): #
                if level == len(vision_features):
                    feature_maps_i = self.input_proj_vision[level](feature_maps[-1][0])
                else:
                    feature_maps_i = self.input_proj_vision[level](vision_features[-1])
                    
                feature_maps_mask_i = nn.functional.interpolate(
                    pixel_mask.unsqueeze(1).float(),
                    size=feature_maps_i.shape[-2:],
                    mode="nearest"
                ).to(torch.bool).squeeze(1)
                
                vision_posi_embed_i = self.backbone.position_embedding(feature_maps_i, feature_maps_mask_i).to(feature_maps_i.dtype)

                vision_features.append(feature_maps_i)
                vision_attention_mask.append(feature_maps_mask_i)
                vision_posi_embed.append(vision_posi_embed_i)
                


        return vision_features, vision_attention_mask, vision_posi_embed


    def _vision_features_handle(
        self,
        vision_features: List[torch.Tensor],
        vision_attention_mask: List[torch.Tensor],
        vision_posi_embed: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        vision_features_flatten  = []
        vision_attention_mask_flatten  = []
        lvl_vision_posi_embed_flatten = []
        
        vision_map_shapes = []
        
        for level, (vision_features_i, vision_attention_mask_i, vision_posi_embed_i) in enumerate(zip(vision_features, vision_attention_mask, vision_posi_embed)):

            batch_size, _, height, width = vision_features_i.shape
            vision_features_i = vision_features_i.flatten(2).transpose(1, 2) 
            vision_attention_mask_i = vision_attention_mask_i.flatten(1)  # Shape: [batch_size, height*width]
            
            vision_posi_embed_i = vision_posi_embed_i.flatten(2).transpose(1, 2) 
            lvl_vision_posi_embed_i = vision_posi_embed_i + self.level_embed[level].view(1, 1, -1)

            
            vision_features_flatten.append(vision_features_i)
            vision_attention_mask_flatten.append(vision_attention_mask_i)
            lvl_vision_posi_embed_flatten.append(lvl_vision_posi_embed_i)
            
            spatial_shape = (height, width)
            vision_map_shapes.append(spatial_shape)
            
        vision_features_flatten  = torch.cat(vision_features_flatten , 1) 
        vision_attention_mask_flatten  = torch.cat(vision_attention_mask_flatten , 1)
        lvl_vision_posi_embed_flatten = torch.cat(lvl_vision_posi_embed_flatten, 1)
        
        vision_map_shapes = torch.as_tensor(vision_map_shapes, dtype=torch.long, device=vision_features_flatten .device)
        level_start_index = torch.cat((vision_map_shapes.new_zeros((1,)), vision_map_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in vision_attention_mask], 1).float()

        return (
            batch_size,
            
            vision_features_flatten ,
            vision_attention_mask_flatten ,
            lvl_vision_posi_embed_flatten,
            
            vision_map_shapes,
            level_start_index,
            valid_ratios,
        )

            

    def _prepare_decoder_inputs(
        self,
        query_embeds,
        encoder_outputs: Union[GroundingDinoEncoderOutput, Tuple[torch.Tensor, ...]],
        vision_attention_mask : torch.Tensor,
        vision_map_shapes: torch.Tensor,
        text_attention_mask : torch.Tensor,
        batch_size: int,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        # Initialize outputs
        enc_outputs_class = None
        enc_outputs_coord = None

        if self.config.two_stage:
            # Generate proposals and object query embeddings
            valid_object_query, valid_proposals = self.encoder_query_proposals(
                encoder_outputs[0], ~vision_attention_mask, vision_map_shapes
            ) #torch.Size([1, 17821, 256]) torch.Size([1, 17821]) torch.Size([4, 2])

            # Classification and bounding box regression (Two-Stage DETR)
            enc_outputs_class = self.encoder_output_class_embed(
                valid_object_query, encoder_outputs[1], text_attention_mask
            ) #torch.Size([1, 17821, 256])
            enc_outputs_coord = self.encoder_output_bbox_embed(valid_object_query) + valid_proposals #torch.Size([1, 17821, 4])

            # Select top scoring proposals
            topk = self.config.num_queries
            topk_indices = torch.topk(enc_outputs_class.max(-1)[0], topk, dim=1)[1]

            topk_coords_logits = torch.gather(
                enc_outputs_coord, 1, topk_indices.unsqueeze(-1).expand(-1, -1, 4)
            ).detach()

            topk_coords = topk_coords_logits.sigmoid()
            init_topk_coords = topk_coords

            # Prepare query_embeds embeddings
            query_embeds = (
                query_embeds.unsqueeze(0).repeat(batch_size, 1, 1)
                if query_embeds is not None
                else torch.gather(
                    valid_object_query, 1, topk_indices.unsqueeze(-1).expand(-1, -1, self.d_model)
                ).detach()
            )
        else:
            # Single-stage case
            query_embeds = query_embeds.unsqueeze(0).repeat(batch_size, 1, 1)
            topk_coords = self.topk_coords_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1).sigmoid()
            init_topk_coords = topk_coords

        return query_embeds, topk_coords, init_topk_coords, enc_outputs_class, enc_outputs_coord

            
    @add_start_docstrings_to_model_forward(GROUNDING_DINO_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=GroundingDinoModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        
        pixel_values: Tensor,
        pixel_mask: Optional[Tensor] = None,
        
        input_ids: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        text_attention_mask: Optional[Tensor] = None,

        return_dict=None,
        encoder_outputs=None,
        output_attentions=None,
        output_hidden_states=None,

    ):
        r"""
        Returns:
        ```"""
        # Step 1: 参数处理 因为模型才有 self.config
        (
            return_dict, #true
            output_attentions, #false
            output_hidden_states, #false
        ) = self._handle_params(return_dict, output_attentions, output_hidden_states)


        """ Vanilla Text Features """

        (
            input_ids, 
            token_type_ids, #哪句话
            text_attention_mask, # 好像都值得关注
            
            text_self_attention_mask, #大矩阵
            position_ids, #一句话的分隔
        ) = self._prepare_text_inputs(input_ids, token_type_ids, text_attention_mask)
        # Step 3: 文本特征提取  我的一切操作都是为了可以调用bert的api  bert特征提取加特征映射
        text_features = self._extract_text_features(input_ids, token_type_ids, text_self_attention_mask, position_ids) #-》torch.Size([1, 9, 256])


        """ Vanilla Image Features """
        # Step 4: 提取视觉特征
        vision_features, vision_attention_mask, vision_posi_embed = self._extract_vision_features(pixel_values, pixel_mask) #torch.Size([1, 256, 100, 134]) torch.Size([1, 256, 50, 67]) torch.Size([1, 256, 25, 34]) torch.Size([1, 256, 13, 17]) 总之就是图像处理成金字塔  
        # Step 5: 视觉特征进行处理
        (
            batch_size,
            vision_features ,
            vision_attention_mask ,
            vision_posi_embed,
            
            vision_map_shapes,
            level_start_index,
            valid_ratios,
        ) = self._vision_features_handle(vision_features, vision_attention_mask, vision_posi_embed) #简答理解为展平后的结果


        """ Feature Enhancer """
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                vision_features=vision_features, #torch.Size([1, 17821, 256])
                vision_attention_mask=~vision_attention_mask, #torch.Size([1, 17821])
                vision_posi_embed=vision_posi_embed, #torch.Size([1, 17821, 256])
                
                vision_map_shapes=vision_map_shapes, #torch.Size([4, 2])
                level_start_index=level_start_index, #torch.Size([4])
                valid_ratios=valid_ratios, #torch.Size([1, 4, 2])
                
                text_features=text_features,
                text_attention_mask=~text_attention_mask,
                text_posi_embed=None,
                
                text_self_attention_mask=~text_self_attention_mask,
                text_position_ids=position_ids,

                return_dict=return_dict,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
        elif return_dict and not isinstance(encoder_outputs, GroundingDinoEncoderOutput):
            encoder_outputs = GroundingDinoEncoderOutput(
                encoder_vision_features_states=encoder_outputs[0],
                encoder_text_features_states=encoder_outputs[1],
                vision_hidden_states=encoder_outputs[2] if output_hidden_states else None,
                text_hidden_states=encoder_outputs[3] if output_hidden_states else None,
                attentions=encoder_outputs[-1] if output_attentions else None,
            )


        """ 解码器 """
        # Create queries
        query_embeds = None
        if self.config.embedding_init_target or self.config.two_stage:
            query_embeds = self.query_position_embeddings.weight
        # Step 7: Prepare decoder inputs
        (
            query_embeds,
            topk_coords,
            init_topk_coords,
            enc_outputs_class,
            enc_outputs_coord,
        ) = self._prepare_decoder_inputs(query_embeds, encoder_outputs, vision_attention_mask , vision_map_shapes, text_attention_mask, batch_size) #torch.Size([1, 17821]) torch.Size([4, 2]) 1 torch.Size([900, 256]) torch.Size([1, 9])
        # Step 8: Pass through decoder
        decoder_outputs = self.decoder(
            query_embeds=query_embeds,
            topk_coords=topk_coords,
            
            encoder_vision_features=encoder_outputs[0],
            vision_attention_mask=vision_attention_mask ,
            encoder_text_features=encoder_outputs[1],
            text_attention_mask=~text_attention_mask,

            vision_map_shapes=vision_map_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,

            return_dict=return_dict,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,

            self_attn_mask=None,
        )

        """ 输出结果 """
        # Step 9: outputs
        if not return_dict:
            enc_outputs = tuple(value for value in [enc_outputs_class, enc_outputs_coord] if value is not None)
            tuple_outputs = (
                (decoder_outputs[0], init_topk_coords) + decoder_outputs[1:] + encoder_outputs + enc_outputs
            )
            return tuple_outputs

        return GroundingDinoModelOutput(

            decoder_query_embeds_states=decoder_outputs.decoder_query_embeds_states,
            decoder_query_embeds_hidden_states=decoder_outputs.decoder_query_embeds_hidden_states,
            decoder_refine_coords_hidden_states=decoder_outputs.decoder_refine_coords_hidden_states,
            decoder_attentions=decoder_outputs.decoder_attentions,


            encoder_vision_features_states=encoder_outputs.encoder_vision_features_states,
            encoder_text_features_states=encoder_outputs.encoder_text_features_states,
            encoder_vision_hidden_states=encoder_outputs.encoder_vision_hidden_states,
            encoder_text_hidden_states=encoder_outputs.encoder_text_hidden_states,
            encoder_attentions=encoder_outputs.encoder_attentions,
            
            init_topk_coords=init_topk_coords, 
            enc_outputs_class=enc_outputs_class,
            enc_outputs_coord=enc_outputs_coord,
        )


# Copied from transformers.models.detr.modeling_detr.DetrMLPPredictionHead
class GroundingDinoMLPPredictionHead(nn.Module):
    """
    Very simple multi-layer perceptron (MLP, also called FFN), used to predict the normalized center coordinates,
    height and width of a bounding box w.r.t. an image.

    Copied from https://github.com/facebookresearch/detr/blob/master/models/detr.py

    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = nn.functional.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


@add_start_docstrings(
    """
    Grounding DINO Model (consisting of a backbone and encoder-decoder Transformer) with object detection heads on top,
    for tasks such as COCO detection.
    """,
    GROUNDING_DINO_START_DOCSTRING,
)
class GroundingDinoForObjectDetection(GroundingDinoPreTrainedModel): #理解为整个项目
    # When using clones, all layers > 0 will be clones, but layer 0 *is* required
    # the bbox_embed in the decoder are all clones though
    _tied_weights_keys = [r"bbox_embed\.[1-9]\d*", r"model\.decoder\.bbox_embed\.[0-9]\d*"]

    def __init__(self, config: GroundingDinoConfig):
        super().__init__(config)

        self.model = GroundingDinoModel(config)
        _class_embed = GroundingDinoContrastiveEmbedding(config)

        if config.decoder_bbox_embed_share:
            _bbox_embed = GroundingDinoMLPPredictionHead(
                input_dim=config.d_model, hidden_dim=config.d_model, output_dim=4, num_layers=3
            )
            self.bbox_embed = nn.ModuleList([_bbox_embed for _ in range(config.decoder_layers)])
        else:
            for _ in range(config.decoder_layers):
                _bbox_embed = GroundingDinoMLPPredictionHead(
                    input_dim=config.d_model, hidden_dim=config.d_model, output_dim=4, num_layers=3
                )
                self.bbox_embed = nn.ModuleList([_bbox_embed for _ in range(config.decoder_layers)])
        self.class_embed = nn.ModuleList([_class_embed for _ in range(config.decoder_layers)])
        # hack for box-refinement
        self.model.decoder.bbox_embed = self.bbox_embed
        # hack implementation for two-stage
        self.model.decoder.class_embed = self.class_embed

        # Initialize weights and apply final processing
        self.post_init()

    # taken from https://github.com/facebookresearch/detr/blob/master/models/detr.py
    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{"logits": a, "pred_boxes": b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]



    def _handle_params(self,return_dict, input_ids, attention_mask):
        return_dict = return_dict or self.config.use_return_dict
        if attention_mask is None:
            return torch.ones_like(input_ids)
        return return_dict, attention_mask


    def _process_encoder_outputs(self, return_dict, outputs, output_attentions, output_hidden_states):
        if return_dict:
            encoder_text_features_states = outputs.encoder_text_features_states
            decoder_query_embeds_hidden_states = outputs.decoder_query_embeds_hidden_states
            init_topk_coords = outputs.init_topk_coords
            decoder_refine_coords_hidden_states = outputs.decoder_refine_coords_hidden_states
        else:
            idx = 5 + int(output_attentions) + int(output_hidden_states)
            encoder_text_features_states = outputs[idx]
            decoder_query_embeds_hidden_states = outputs[2]
            init_topk_coords = outputs[1]
            decoder_refine_coords_hidden_states = outputs[3]
        return encoder_text_features_states, decoder_query_embeds_hidden_states, init_topk_coords, decoder_refine_coords_hidden_states
    

    def _compute_predictions(
        self, encoder_text_features_states, decoder_query_embeds_hidden_states, init_topk_coords, decoder_refine_coords_hidden_states, attention_mask
    ):
        outputs_hidden_classes, outputs_hidden_coords = [], []
        num_levels = decoder_query_embeds_hidden_states.shape[1]

        for level in range(num_levels):
            refine_coords = init_topk_coords if level == 0 else decoder_refine_coords_hidden_states[:, level - 1]
            refine_coords = torch.special.logit(refine_coords, eps=1e-5)

            class_logits = self.class_embed[level](
                vision_hidden_state=decoder_query_embeds_hidden_states[:, level], #解码器的图像信息
                text_hidden_state=encoder_text_features_states, #编码器的文本信息
                attention_mask=attention_mask.bool(),
            )
            outputs_hidden_classes.append(class_logits) #->torch.Size([1, 900, 256])

            delta_bbox = self.bbox_embed[level](decoder_query_embeds_hidden_states[:, level]) #解码器的图像信息预测坐标偏差 torch.Size([1, 6, 900, 256]) ->torch.Size([1, 900, 4])
            coord_dim = refine_coords.shape[-1]

            if coord_dim == 4:
                coord_logits = delta_bbox + refine_coords #实际坐标
            elif coord_dim == 2:
                coord_logits = delta_bbox.clone()
                coord_logits[..., :2] += refine_coords
            else:
                raise ValueError(f"Reference points should have 2 or 4 dimensions, got {coord_dim}")

            outputs_hidden_coords.append(coord_logits.sigmoid())

        outputs_classes = outputs_hidden_classes[-1]
        outputs_coords = outputs_hidden_coords[-1]
        return outputs_classes, outputs_coords, outputs_hidden_classes, outputs_hidden_coords
    
# outputs_classes, outputs_coords, outputs_hidden_classes, outputs_hidden_coords, labels
    def _compute_loss(self, outputs_classes, outputs_coords, outputs_hidden_classes, outputs_hidden_coords, labels):
        loss, loss_dict, auxiliary_outputs = None, None, None
        if labels is not None:
            loss, loss_dict, auxiliary_outputs = self.loss_function(
                outputs_classes, labels, self.device, outputs_coords, self.config, outputs_hidden_classes, outputs_hidden_coords
            )
        return loss, loss_dict, auxiliary_outputs 



    @add_start_docstrings_to_model_forward(GROUNDING_DINO_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=GroundingDinoObjectDetectionOutput, config_class=_CONFIG_FOR_DOC
    )
    def forward(
        self,
        pixel_values: torch.FloatTensor, #torch.Size([1, 3, 800, 1066])
        pixel_mask: Optional[torch.BoolTensor] = None, #torch.Size([1, 800, 1066])
        
        input_ids: torch.LongTensor = None, #torch.Size([1, 9])
        token_type_ids: torch.LongTensor = None, #torch.Size([1, 9])
        attention_mask: torch.LongTensor = None, #torch.Size([1, 9])

        return_dict: Optional[bool] = None, #torch.Size([1, 9])
        encoder_outputs: Optional[Union[GroundingDinoEncoderOutput, Tuple]] = None, #None
        output_attentions: Optional[bool] = None, #None
        output_hidden_states: Optional[bool] = None, #None

        labels: List[Dict[str, Union[torch.LongTensor, torch.FloatTensor]]] = None, #None
    ):
        r"""
        Returns:
        """
        return_dict, text_attention_mask = self._handle_params(return_dict, input_ids, attention_mask)

        """ 重点 """
        outputs = self.model(
            pixel_values=pixel_values, #图像 torch.Size([1, 3, 800, 1066])
            pixel_mask=pixel_mask, #图像 torch.Size([1, 800, 1066])
            
            input_ids=input_ids, #torch.Size([1, 9])
            token_type_ids=token_type_ids, #torch.Size([1, 9]) 区分是哪一个句子 0 
            text_attention_mask=text_attention_mask, #torch.Size([1, 9]) 全有用 1

            return_dict=return_dict, #True
            encoder_outputs=encoder_outputs, #None
            output_attentions=output_attentions, #None
            output_hidden_states=output_hidden_states, #None
        ) #输入参数原封不动的传进去了

        encoder_text_features_states, decoder_query_embeds_hidden_states, init_topk_coords, decoder_refine_coords_hidden_states = \
            self._process_encoder_outputs(return_dict, outputs, output_attentions, output_hidden_states)


        outputs_classes, outputs_coords, outputs_hidden_classes, outputs_hidden_coords = self._compute_predictions(
            encoder_text_features_states, decoder_query_embeds_hidden_states, init_topk_coords, decoder_refine_coords_hidden_states, text_attention_mask
        )
        
        loss, loss_dict, auxiliary_outputs = self._compute_loss(outputs_classes, outputs_coords, outputs_hidden_classes, outputs_hidden_coords, labels)
        
        if return_dict:
            dict_outputs = GroundingDinoObjectDetectionOutput(
                loss=loss,
                loss_dict=loss_dict,
                auxiliary_outputs=auxiliary_outputs,
                
                outputs_classes=outputs_classes,
                outputs_coords=outputs_coords,
                
                decoder_query_embeds_states=outputs.decoder_query_embeds_states,
                decoder_query_embeds_hidden_states=outputs.decoder_query_embeds_hidden_states,
                decoder_refine_coords_hidden_states=outputs.decoder_refine_coords_hidden_states,
                decoder_attentions=outputs.decoder_attentions,
                
                encoder_vision_features_states=outputs.encoder_vision_features_states,
                encoder_text_features_states=outputs.encoder_text_features_states,
                encoder_vision_hidden_states=outputs.encoder_vision_hidden_states,
                encoder_text_hidden_states=outputs.encoder_text_hidden_states,
                encoder_attentions=outputs.encoder_attentions,
                
                init_topk_coords=outputs.init_topk_coords,
                enc_outputs_class=outputs.enc_outputs_class,
                enc_outputs_coord=outputs.enc_outputs_coord,
            )
            return dict_outputs
        else:
            base_output = (outputs_classes, outputs_coords) + (auxiliary_outputs or ()) + tuple(outputs)
            return ((loss, loss_dict) + base_output) if loss is not None else base_output