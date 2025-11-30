import math
import random
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

import torch
import torch.distributed
from omegaconf import DictConfig, ListConfig, open_dict
from torch import nn

from nemo.collections.asr.modules.conformer_encoder import ConformerEncoder
from nemo.collections.asr.models.configs import CacheAwareStreamingConfig
from nemo.collections.asr.parts.mixins.streaming import StreamingEncoder
from nemo.collections.asr.parts.submodules.causal_convs import CausalConv1D
from nemo.collections.asr.parts.submodules.conformer_modules import ConformerLayer
from nemo.collections.asr.parts.submodules.multi_head_attention import (
    LocalAttRelPositionalEncoding,
    MultiHeadAttention,
    PositionalEncoding,
    RelPositionalEncoding,
    RelPositionMultiHeadAttention,
    RelPositionMultiHeadAttentionLongformer,
)
from nemo.collections.asr.parts.submodules.subsampling import (
    ConvSubsampling,
    StackingSubsampling,
    SubsamplingReductionModule,
)
from nemo.collections.asr.parts.utils import adapter_utils
from nemo.collections.asr.parts.utils.regularization_utils import compute_stochastic_depth_drop_probs
from nemo.core.classes.common import typecheck
from nemo.core.classes.exportable import Exportable
from nemo.core.classes.mixins import AccessMixin, adapter_mixins
from nemo.core.classes.module import NeuralModule
from nemo.core.neural_types import (
    AcousticEncodedRepresentation,
    BoolType,
    ChannelType,
    LengthsType,
    NeuralType,
    SpectrogramType,
)
from nemo.utils import logging

__all__ = ['ConformerEncoder']


class MidSeqConformerEncoder(ConformerEncoder):
    def __init__(
        self,
        feat_in,
        n_layers,
        d_model,
        n_recur=3,
        feat_out=-1,
        causal_downsampling=False,
        subsampling='striding',
        subsampling_factor=4,
        subsampling_conv_chunking_factor=1,
        subsampling_conv_channels=-1,
        reduction=None,
        reduction_position=None,
        reduction_factor=1,
        ff_expansion_factor=4,
        self_attention_model='rel_pos',
        n_heads=4,
        att_context_size=None,
        att_context_probs=None,
        att_context_style='regular',
        xscaling=True,
        untie_biases=True,
        pos_emb_max_len=5000,
        conv_kernel_size=31,
        conv_norm_type='batch_norm',
        conv_context_size=None,
        use_bias=True,
        dropout=0.1,
        dropout_pre_encoder=0.1,
        dropout_emb=0.1,
        dropout_att=0.0,
        stochastic_depth_drop_prob: float = 0.0,
        stochastic_depth_mode: str = "linear",
        stochastic_depth_start_layer: int = 1,
        global_tokens: int = 0,
        global_tokens_spacing: int = 1,
        global_attn_separate: bool = False,
        use_pytorch_sdpa: bool = False,
        use_pytorch_sdpa_backends=None,
        sync_max_audio_length: bool = True,
    ):
        super().__init__()
        d_ff = d_model * ff_expansion_factor
        self.d_model = d_model
        self.n_layers = n_layers
        self._feat_in = feat_in
        self.n_recur = n_recur
        self.att_context_style = att_context_style
        self.subsampling_factor = subsampling_factor
        self.subsampling_conv_chunking_factor = subsampling_conv_chunking_factor

        self.self_attention_model = self_attention_model
        self.global_tokens = global_tokens
        self.global_attn_separate = global_attn_separate
        self.global_tokens_spacing = global_tokens_spacing
        self.use_pytorch_sdpa = use_pytorch_sdpa
        if use_pytorch_sdpa_backends is None:
            use_pytorch_sdpa_backends = []
        self.use_pytorch_sdpa_backends = use_pytorch_sdpa_backends
        self.sync_max_audio_length = sync_max_audio_length

        # Setting up the att_context_size
        (
            self.att_context_size_all,
            self.att_context_size,
            self.att_context_probs,
            self.conv_context_size,
        ) = self._calc_context_sizes(
            att_context_style=att_context_style,
            att_context_size=att_context_size,
            att_context_probs=att_context_probs,
            conv_context_size=conv_context_size,
            conv_kernel_size=conv_kernel_size,
        )

        if xscaling:
            self.xscale = math.sqrt(d_model)
        else:
            self.xscale = None

        # Subsampling
        if subsampling_conv_channels == -1:
            subsampling_conv_channels = d_model
        if subsampling and subsampling_factor > 1:
            if subsampling in ['stacking', 'stacking_norm']:
                # stacking_norm has an extra layer norm after stacking comparing to stacking
                self.pre_encode = StackingSubsampling(
                    subsampling_factor=subsampling_factor,
                    feat_in=feat_in,
                    feat_out=d_model,
                    norm=True if subsampling == 'stacking_norm' else False,
                )
            else:
                self.pre_encode = ConvSubsampling(
                    subsampling=subsampling,
                    subsampling_factor=subsampling_factor,
                    feat_in=feat_in,
                    feat_out=d_model,
                    conv_channels=subsampling_conv_channels,
                    subsampling_conv_chunking_factor=subsampling_conv_chunking_factor,
                    activation=nn.ReLU(True),
                    is_causal=causal_downsampling,
                )
        else:
            self.pre_encode = nn.Linear(feat_in, d_model)

        # Reduction
        if reduction and reduction_factor > 1:
            assert reduction_position >= -1 and reduction_position < n_layers
            self.reduction_subsampling = SubsamplingReductionModule(
                reduction=reduction,
                d_model=d_model,
                reduction_factor=reduction_factor,
            )
            self.reduction_position = reduction_position
        else:
            self.reduction_subsampling = None
            self.reduction_position = None

        self._feat_out = d_model

        # Biases for relative positional encoding
        if not untie_biases and self_attention_model == "rel_pos":
            d_head = d_model // n_heads
            pos_bias_u = nn.Parameter(torch.Tensor(n_heads, d_head))
            pos_bias_v = nn.Parameter(torch.Tensor(n_heads, d_head))
            nn.init.zeros_(pos_bias_u)
            nn.init.zeros_(pos_bias_v)
        else:
            pos_bias_u = None
            pos_bias_v = None

        # Positional encodings
        self.pos_emb_max_len = pos_emb_max_len
        if self_attention_model == "rel_pos":
            self.pos_enc = RelPositionalEncoding(
                d_model=d_model,
                dropout_rate=dropout_pre_encoder,
                max_len=pos_emb_max_len,
                xscale=self.xscale,
                dropout_rate_emb=dropout_emb,
            )
        elif self_attention_model == 'rel_pos_local_attn':
            if max(att_context_size) <= 0:
                raise ValueError("When using local attention, context size must be set > 0")
            self.pos_enc = LocalAttRelPositionalEncoding(
                att_context_size=att_context_size,
                d_model=d_model,
                dropout_rate=dropout,
                max_len=pos_emb_max_len,
                xscale=self.xscale,
                dropout_rate_emb=dropout_emb,
            )
        elif self_attention_model == "abs_pos":
            pos_bias_u = None
            pos_bias_v = None
            self.pos_enc = PositionalEncoding(
                d_model=d_model, dropout_rate=dropout_pre_encoder, max_len=pos_emb_max_len, xscale=self.xscale
            )
        else:
            raise ValueError(f"Not valid self_attention_model: '{self_attention_model}'!")

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            layer = ConformerLayer(
                d_model=d_model,
                d_ff=d_ff,
                self_attention_model=self_attention_model,
                global_tokens=global_tokens,
                global_tokens_spacing=global_tokens_spacing,
                global_attn_separate=global_attn_separate,
                n_heads=n_heads,
                conv_kernel_size=conv_kernel_size,
                conv_norm_type=conv_norm_type,
                conv_context_size=self.conv_context_size,
                dropout=dropout,
                dropout_att=dropout_att,
                pos_bias_u=pos_bias_u,
                pos_bias_v=pos_bias_v,
                att_context_size=self.att_context_size,
                use_bias=use_bias,
                use_pytorch_sdpa=self.use_pytorch_sdpa,
                use_pytorch_sdpa_backends=self.use_pytorch_sdpa_backends,
            )
            self.layers.append(layer)

        if feat_out > 0 and feat_out != self._feat_out:
            self.out_proj = nn.Linear(self._feat_out, feat_out)
            self._feat_out = feat_out
        else:
            self.out_proj = None
            self._feat_out = d_model
        self.set_max_audio_length(self.pos_emb_max_len)
        self.use_pad_mask = True

        self.setup_streaming_params()
        self.export_cache_support = False

        # will be set in self.forward() if defined in AccessMixin config
        self.interctc_capture_at_layers = None

    def forward_internal(
        self,
        audio_signal,
        length,
        cache_last_channel=None,
        cache_last_time=None,
        cache_last_channel_len=None,
        bypass_pre_encode=False,
    ):
        """
        The `audio_signal` input supports two formats depending on the `bypass_pre_encode` boolean flag.
        This determines the required format of the input variable `audio_signal`:
        (1) bypass_pre_encode = False (default):
            `audio_signal` must be a tensor containing audio features.
            Shape: (batch, self._feat_in, n_frames)
        (2) bypass_pre_encode = True:
            `audio_signal` must be a tensor containing pre-encoded embeddings.
            Shape: (batch, n_frame, self.d_model)

        `bypass_pre_encode=True` is used in cases where frame-level, context-independent embeddings are
        needed to be saved or reused (e.g., speaker cache in streaming speaker diarization).
        """
        if length is None:
            length = audio_signal.new_full(
                (audio_signal.size(0),), audio_signal.size(-1), dtype=torch.int64, device=audio_signal.device
            )

        # select a random att_context_size with the distribution specified by att_context_probs during training
        # for non-validation cases like test, validation or inference, it uses the first mode in self.att_context_size
        if self.training and len(self.att_context_size_all) > 1:
            cur_att_context_size = random.choices(self.att_context_size_all, weights=self.att_context_probs)[0]
        else:
            cur_att_context_size = self.att_context_size

        if not bypass_pre_encode:
            audio_signal = torch.transpose(audio_signal, 1, 2)

            if isinstance(self.pre_encode, nn.Linear):
                audio_signal = self.pre_encode(audio_signal)
            else:
                audio_signal, length = self.pre_encode(x=audio_signal, lengths=length)
                length = length.to(torch.int64)
                # `self.streaming_cfg` is set by setup_streaming_cfg(), called in the init
                if self.streaming_cfg.drop_extra_pre_encoded > 0 and cache_last_channel is not None:
                    audio_signal = audio_signal[:, self.streaming_cfg.drop_extra_pre_encoded :, :]
                    length = (length - self.streaming_cfg.drop_extra_pre_encoded).clamp(min=0)

            if self.reduction_position is not None and cache_last_channel is not None:
                raise ValueError("Caching with reduction feature is not supported yet!")

        max_audio_length = audio_signal.size(1)
        if cache_last_channel is not None:
            cache_len = self.streaming_cfg.last_channel_cache_size
            cache_keep_size = max_audio_length - self.streaming_cfg.cache_drop_size
            max_audio_length = max_audio_length + cache_len
            padding_length = length + cache_len
            offset = torch.neg(cache_last_channel_len) + cache_len
        else:
            padding_length = length
            cache_last_channel_next = None
            cache_len = 0
            offset = None

        audio_signal, pos_emb = self.pos_enc(x=audio_signal, cache_len=cache_len)

        # Create the self-attention and padding masks
        pad_mask, att_mask = self._create_masks(
            att_context_size=cur_att_context_size,
            padding_length=padding_length,
            max_audio_length=max_audio_length,
            offset=offset,
            device=audio_signal.device,
        )

        if cache_last_channel is not None:
            pad_mask = pad_mask[:, cache_len:]
            if att_mask is not None:
                att_mask = att_mask[:, cache_len:]
            # Convert caches from the tensor to list
            cache_last_time_next = []
            cache_last_channel_next = []

        # layer 0
        for lth, layer in enumerate(self.layers[0:3]):
            if cache_last_channel is not None:
                cache_last_channel_cur = cache_last_channel[lth]
                cache_last_time_cur = cache_last_time[lth]
            else:
                cache_last_channel_cur = None
                cache_last_time_cur = None
            audio_signal = layer(
                x=audio_signal,
                att_mask=att_mask,
                pos_emb=pos_emb,
                pad_mask=pad_mask,
                cache_last_channel=cache_last_channel_cur,
                cache_last_time=cache_last_time_cur,
            )

        # inter layer
        for lth, layer in enumerate(self.layers[3:-1]):
            for _ in range(self.n_recur):
                if cache_last_channel is not None:
                    cache_last_channel_cur = cache_last_channel[lth]
                    cache_last_time_cur = cache_last_time[lth]
                else:
                    cache_last_channel_cur = None
                    cache_last_time_cur = None
                audio_signal = layer(
                    x=audio_signal,
                    att_mask=att_mask,
                    pos_emb=pos_emb,
                    pad_mask=pad_mask,
                    cache_last_channel=cache_last_channel_cur,
                    cache_last_time=cache_last_time_cur,
                )
        # last layer
        if cache_last_channel is not None:
            cache_last_channel_cur = cache_last_channel[lth]
            cache_last_time_cur = cache_last_time[lth]
        else:
            cache_last_channel_cur = None
            cache_last_time_cur = None
        audio_signal = self.layers[-1](
            x=audio_signal,
            att_mask=att_mask,
            pos_emb=pos_emb,
            pad_mask=pad_mask,
            cache_last_channel=cache_last_channel_cur,
            cache_last_time=cache_last_time_cur,
        ) 

        if self.out_proj is not None:
            audio_signal = self.out_proj(audio_signal)

        # Reduction
        if self.reduction_position == -1:
            audio_signal, length = self.reduction_subsampling(x=audio_signal, lengths=length)

        audio_signal = torch.transpose(audio_signal, 1, 2)
        length = length.to(dtype=torch.int64)

        if cache_last_channel is not None:
            cache_last_channel_next = torch.stack(cache_last_channel_next, dim=0)
            cache_last_time_next = torch.stack(cache_last_time_next, dim=0)
            return (
                audio_signal,
                length,
                cache_last_channel_next,
                cache_last_time_next,
                torch.clamp(cache_last_channel_len + cache_keep_size, max=cache_len),
            )
        else:
            return audio_signal, length
