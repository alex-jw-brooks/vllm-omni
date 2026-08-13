# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""StyleTTS2 model — nn.Module wrapper for vLLM Omni.

All component module definitions are ported with __init__ methods matching
the reference exactly so state_dict keys align with exported checkpoints.

Expected checkpoint format (produced by export_styletts2.py):
  Flat state_dict with keys like:
    bert.embeddings.word_embeddings.weight
    bert_encoder.weight
    decoder.decode.0.conv1.weight_g
    predictor.text_encoder.lstms.0.weight_ih_l0
    ...
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as audio_F
from torch.nn.utils import spectral_norm, weight_norm
from transformers import AlbertConfig, AlbertModel
from vllm.config import VllmConfig

from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.transformers_utils.configs.granite_styletts2 import (
    GraniteStyleTTS2Config,
)

logger = logging.getLogger(__name__)

_PLBERT_HIDDEN_SIZE = 768


def _length_to_mask(lengths: torch.Tensor) -> torch.Tensor:
    mask = torch.arange(lengths.max(), device=lengths.device).unsqueeze(0)
    mask = mask.expand(lengths.shape[0], -1).type_as(lengths)
    return torch.gt(mask + 1, lengths.unsqueeze(1))


# ─── Shared utility layers ─────────────────────────────────────────────────────


def _init_weights(m, mean=0.0, std=0.01):
    if m.__class__.__name__.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def _get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def _get_activation_fn(active):
    if active == "relu":
        return nn.ReLU()
    elif active == "lrelu":
        return nn.LeakyReLU(0.2)
    raise RuntimeError(f"Unexpected active type {active}")


class LinearNorm(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain="linear"):
        super().__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)
        nn.init.xavier_uniform_(self.linear_layer.weight, gain=nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class ConvNorm(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=None,
        dilation=1,
        bias=True,
        w_init_gain="linear",
        param=None,
    ):
        super().__init__()
        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        nn.init.xavier_uniform_(
            self.conv.weight,
            gain=nn.init.calculate_gain(w_init_gain, param=param),
        )

    def forward(self, signal):
        return self.conv(signal)


# ─── JDC (pitch extractor) ─────────────────────────────────────────────────────


class JDCResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, leaky_relu_slope=0.01):
        super().__init__()
        self.downsample = in_channels != out_channels
        self.pre_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(leaky_relu_slope, inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2)),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(leaky_relu_slope, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        )
        self.conv1by1 = None
        if self.downsample:
            self.conv1by1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        res = x
        x = self.pre_conv(x)
        x = self.conv(x)
        if self.downsample:
            res = self.conv1by1(res)
            res = F.max_pool2d(res, kernel_size=(1, 2))
        return x + res


class JDCNet(nn.Module):
    def __init__(self, num_class, seq_len, leaky_relu_slope=0.01):
        super().__init__()
        self.num_class = num_class
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(leaky_relu_slope, inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
        )
        self.res_block1 = JDCResBlock(64, 128)
        self.res_block2 = JDCResBlock(128, 192)
        self.res_block3 = JDCResBlock(192, 256)
        self.pool_block = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.LeakyReLU(leaky_relu_slope, inplace=True),
            nn.MaxPool2d(kernel_size=(1, 4)),
            nn.Dropout(p=0.2),
        )
        self.maxpool1 = nn.MaxPool2d(kernel_size=(1, 40))
        self.maxpool2 = nn.MaxPool2d(kernel_size=(1, 20))
        self.maxpool3 = nn.MaxPool2d(kernel_size=(1, 10))
        self.detector_conv = nn.Sequential(
            nn.Conv2d(640, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(leaky_relu_slope, inplace=True),
            nn.Dropout(p=0.2),
        )
        self.bilstm_classifier = nn.LSTM(
            512,
            256,
            batch_first=True,
            bidirectional=True,
        )
        self.bilstm_detector = nn.LSTM(
            512,
            256,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Linear(512, self.num_class)
        self.detector = nn.Linear(512, 2)

    def forward(self, x):
        # Not used in inference — pitch_extractor is only for training
        raise NotImplementedError("JDCNet.forward() not needed for inference")


# ─── ASR (text aligner) ────────────────────────────────────────────────────────


class ConvBlock(nn.Module):
    def __init__(self, hidden_dim, n_conv=3, dropout_p=0.2, active="relu"):
        super().__init__()
        self._n_groups = 8
        self.blocks = nn.ModuleList(
            [self._get_conv(hidden_dim, dilation=3**i, active=active, dropout_p=dropout_p) for i in range(n_conv)]
        )

    def _get_conv(self, hidden_dim, dilation, active="relu", dropout_p=0.2):
        return nn.Sequential(
            ConvNorm(hidden_dim, hidden_dim, kernel_size=3, padding=dilation, dilation=dilation),
            _get_activation_fn(active),
            nn.GroupNorm(self._n_groups, hidden_dim),
            nn.Dropout(p=dropout_p),
            ConvNorm(hidden_dim, hidden_dim, kernel_size=3, padding=1, dilation=1),
            _get_activation_fn(active),
            nn.Dropout(p=dropout_p),
        )

    def forward(self, x):
        for block in self.blocks:
            x = x + block(x)
        return x


class MFCC(nn.Module):
    def __init__(self, n_mfcc=40, n_mels=80):
        super().__init__()
        dct_mat = audio_F.create_dct(n_mfcc, n_mels, "ortho")
        self.register_buffer("dct_mat", dct_mat)

    def forward(self, mel_specgram):
        return torch.matmul(mel_specgram, self.dct_mat)


class LocationLayer(nn.Module):
    def __init__(self, attention_n_filters, attention_kernel_size, attention_dim):
        super().__init__()
        padding = int((attention_kernel_size - 1) / 2)
        self.location_conv = ConvNorm(
            2,
            attention_n_filters,
            kernel_size=attention_kernel_size,
            padding=padding,
            bias=False,
            stride=1,
            dilation=1,
        )
        self.location_dense = LinearNorm(
            attention_n_filters,
            attention_dim,
            bias=False,
            w_init_gain="tanh",
        )

    def forward(self, x):
        x = self.location_conv(x)
        x = x.transpose(1, 2)
        return self.location_dense(x)


class ASRAttention(nn.Module):
    def __init__(
        self,
        attention_rnn_dim,
        embedding_dim,
        attention_dim,
        attention_location_n_filters,
        attention_location_kernel_size,
    ):
        super().__init__()
        self.query_layer = LinearNorm(
            attention_rnn_dim,
            attention_dim,
            bias=False,
            w_init_gain="tanh",
        )
        self.memory_layer = LinearNorm(
            embedding_dim,
            attention_dim,
            bias=False,
            w_init_gain="tanh",
        )
        self.v = LinearNorm(attention_dim, 1, bias=False)
        self.location_layer = LocationLayer(
            attention_location_n_filters,
            attention_location_kernel_size,
            attention_dim,
        )

    def forward(self, *args, **kwargs):
        # Not used in inference — text_aligner attention is training-only
        raise NotImplementedError("ASRAttention.forward() not needed for inference")


class ASRS2S(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, n_location_filters=32, location_kernel_size=63, *, n_token):
        super().__init__()
        self.embedding = nn.Embedding(n_token, embedding_dim)
        val_range = math.sqrt(6 / hidden_dim)
        self.embedding.weight.data.uniform_(-val_range, val_range)
        self.decoder_rnn_dim = hidden_dim
        self.project_to_n_symbols = nn.Linear(self.decoder_rnn_dim, n_token)
        self.attention_layer = ASRAttention(
            self.decoder_rnn_dim,
            hidden_dim,
            hidden_dim,
            n_location_filters,
            location_kernel_size,
        )
        self.decoder_rnn = nn.LSTMCell(
            self.decoder_rnn_dim + embedding_dim,
            self.decoder_rnn_dim,
        )
        self.project_to_hidden = nn.Sequential(
            LinearNorm(self.decoder_rnn_dim * 2, hidden_dim),
            nn.Tanh(),
        )

    def forward(self, *args, **kwargs):
        # Not used in inference
        raise NotImplementedError("ASRS2S.forward() not needed for inference")


class ASRCNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_token, n_layers=6, *, token_embedding_dim):
        super().__init__()
        self.n_token = n_token
        self.n_down = 1
        self.to_mfcc = MFCC()
        self.init_cnn = ConvNorm(
            input_dim // 2,
            hidden_dim,
            kernel_size=7,
            padding=3,
            stride=2,
        )
        self.cnns = nn.Sequential(
            *[nn.Sequential(ConvBlock(hidden_dim), nn.GroupNorm(1, hidden_dim)) for _ in range(n_layers)]
        )
        self.projection = ConvNorm(hidden_dim, hidden_dim // 2)
        self.ctc_linear = nn.Sequential(
            LinearNorm(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            LinearNorm(hidden_dim, n_token),
        )
        self.asr_s2s = ASRS2S(
            embedding_dim=token_embedding_dim,
            hidden_dim=hidden_dim // 2,
            n_token=n_token,
        )

    def forward(self, x, **kwargs):
        # Not used in inference
        raise NotImplementedError("ASRCNN.forward() not needed for inference")


# ─── PLBERT ─────────────────────────────────────────────────────────────────────


class CustomAlbert(AlbertModel):
    def forward(self, *args, **kwargs):
        outputs = super().forward(*args, **kwargs)
        return outputs.last_hidden_state


# ─── StyleTTS2 core modules ────────────────────────────────────────────────────


class DownSample(nn.Module):
    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == "none":
            return x
        elif self.layer_type == "timepreserve":
            return F.avg_pool2d(x, (2, 1))
        elif self.layer_type == "half":
            if x.shape[-1] % 2 != 0:
                x = torch.cat([x, x[..., -1:]], dim=-1)
            return F.avg_pool2d(x, 2)
        raise RuntimeError(f"Unexpected downsample type {self.layer_type}")


class LearnedDownSample(nn.Module):
    def __init__(self, layer_type, dim_in):
        super().__init__()
        self.layer_type = layer_type
        if layer_type == "none":
            self.conv = nn.Identity()
        elif layer_type == "timepreserve":
            self.conv = spectral_norm(
                nn.Conv2d(
                    dim_in,
                    dim_in,
                    kernel_size=(3, 1),
                    stride=(2, 1),
                    groups=dim_in,
                    padding=(1, 0),
                )
            )
        elif layer_type == "half":
            self.conv = spectral_norm(
                nn.Conv2d(
                    dim_in,
                    dim_in,
                    kernel_size=(3, 3),
                    stride=(2, 2),
                    groups=dim_in,
                    padding=1,
                )
            )
        else:
            raise RuntimeError(f"Unexpected downsample type {layer_type}")

    def forward(self, x):
        return self.conv(x)


class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2), normalize=False, *, downsample):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = DownSample(downsample)
        self.downsample_res = LearnedDownSample(downsample, dim_in)
        self.learned_sc = dim_in != dim_out
        self.conv1 = spectral_norm(nn.Conv2d(dim_in, dim_in, 3, 1, 1))
        self.conv2 = spectral_norm(nn.Conv2d(dim_in, dim_out, 3, 1, 1))
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = spectral_norm(
                nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False),
            )

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        x = self.downsample(x)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        x = self.downsample_res(x)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        return (self._shortcut(x) + self._residual(x)) / math.sqrt(2)


class StyleEncoder(nn.Module):
    def __init__(self, dim_in, style_dim, max_conv_dim):
        super().__init__()
        blocks = [spectral_norm(nn.Conv2d(1, dim_in, 3, 1, 1))]
        for _ in range(4):
            dim_out = min(dim_in * 2, max_conv_dim)
            blocks.append(ResBlk(dim_in, dim_out, downsample="half"))
            dim_in = dim_out
        blocks.append(nn.LeakyReLU(0.2))
        blocks.append(spectral_norm(nn.Conv2d(dim_out, dim_out, 5, 1, 0)))
        blocks.append(nn.AdaptiveAvgPool2d(1))
        blocks.append(nn.LeakyReLU(0.2))
        self.shared = nn.Sequential(*blocks)
        self.unshared = nn.Linear(dim_out, style_dim)

    def forward(self, x):
        h = self.shared(x)
        h = h.view(h.size(0), -1)
        return self.unshared(h)


class LayerNorm1d(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)


class TextEncoder(nn.Module):
    def __init__(self, channels, kernel_size, depth, n_symbols, actv=nn.LeakyReLU(0.2)):
        super().__init__()
        self.embedding = nn.Embedding(n_symbols, channels)
        padding = (kernel_size - 1) // 2
        self.cnn = nn.ModuleList()
        for _ in range(depth):
            self.cnn.append(
                nn.Sequential(
                    weight_norm(
                        nn.Conv1d(
                            channels,
                            channels,
                            kernel_size=kernel_size,
                            padding=padding,
                        )
                    ),
                    LayerNorm1d(channels),
                    actv,
                    nn.Dropout(0.2),
                )
            )
        self.lstm = nn.LSTM(
            channels,
            channels // 2,
            1,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x, input_lengths, m):
        x = self.embedding(x)
        x = x.transpose(1, 2)
        m = m.to(input_lengths.device).unsqueeze(1)
        x.masked_fill_(m, 0.0)
        for c in self.cnn:
            x = c(x)
            x.masked_fill_(m, 0.0)
        x = x.transpose(1, 2)
        input_lengths_np = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(x, input_lengths_np, batch_first=True, enforce_sorted=False)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x = x.transpose(-1, -2)
        x_pad = torch.zeros([x.shape[0], x.shape[1], m.shape[-1]], device=x.device)
        x_pad[:, :, : x.shape[-1]] = x
        x_pad.masked_fill_(m, 0.0)
        return x_pad


class AdaIN1d(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm1d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features * 2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


class UpSample1d(nn.Module):
    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == "none":
            return x
        return F.interpolate(x, scale_factor=2, mode="nearest")


class AdainResBlk1d(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim, actv=nn.LeakyReLU(0.2), upsample="none", dropout_p=0.0):
        super().__init__()
        self.actv = actv
        self.upsample_type = upsample
        self.upsample = UpSample1d(upsample)
        self.learned_sc = dim_in != dim_out
        self.dropout = nn.Dropout(dropout_p)
        self.conv1 = weight_norm(nn.Conv1d(dim_in, dim_out, 3, 1, 1))
        self.conv2 = weight_norm(nn.Conv1d(dim_out, dim_out, 3, 1, 1))
        self.norm1 = AdaIN1d(style_dim, dim_in)
        self.norm2 = AdaIN1d(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = weight_norm(
                nn.Conv1d(dim_in, dim_out, 1, 1, 0, bias=False),
            )
        if upsample == "none":
            self.pool = nn.Identity()
        else:
            self.pool = weight_norm(
                nn.ConvTranspose1d(
                    dim_in,
                    dim_in,
                    kernel_size=3,
                    stride=2,
                    groups=dim_in,
                    padding=1,
                    output_padding=1,
                )
            )

    def _shortcut(self, x):
        x = self.upsample(x)
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.actv(x)
        x = self.pool(x)
        x = self.conv1(self.dropout(x))
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(self.dropout(x))
        return x

    def forward(self, x, s):
        out = self._residual(x, s)
        return (out + self._shortcut(x)) / math.sqrt(2)


class AdaLayerNorm(nn.Module):
    def __init__(self, style_dim, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.fc = nn.Linear(style_dim, channels * 2)

    def forward(self, x, s):
        x = x.transpose(-1, -2)
        x = x.transpose(1, -1)
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        gamma, beta = gamma.transpose(1, -1), beta.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), eps=self.eps)
        x = (1 + gamma) * x + beta
        return x.transpose(1, -1).transpose(-1, -2)


class DurationEncoder(nn.Module):
    def __init__(self, sty_dim, d_model, nlayers, dropout):
        super().__init__()
        self.lstms = nn.ModuleList()
        for _ in range(nlayers):
            self.lstms.append(
                nn.LSTM(
                    d_model + sty_dim,
                    d_model // 2,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True,
                    dropout=dropout,
                )
            )
            self.lstms.append(AdaLayerNorm(sty_dim, d_model))
        self.dropout = dropout
        self.d_model = d_model
        self.sty_dim = sty_dim

    def forward(self, x, style, text_lengths, m):
        masks = m.to(text_lengths.device)
        x = x.permute(2, 0, 1)
        s = style.expand(x.shape[0], x.shape[1], -1)
        x = torch.cat([x, s], axis=-1)
        x.masked_fill_(masks.unsqueeze(-1).transpose(0, 1), 0.0)
        x = x.transpose(0, 1)
        input_lengths = text_lengths.cpu().numpy()
        x = x.transpose(-1, -2)
        for block in self.lstms:
            if isinstance(block, AdaLayerNorm):
                x = block(x.transpose(-1, -2), style).transpose(-1, -2)
                x = torch.cat([x, s.permute(1, -1, 0)], axis=1)
                x.masked_fill_(masks.unsqueeze(-1).transpose(-1, -2), 0.0)
            else:
                x = x.transpose(-1, -2)
                x = nn.utils.rnn.pack_padded_sequence(x, input_lengths, batch_first=True, enforce_sorted=False)
                block.flatten_parameters()
                x, _ = block(x)
                x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = x.transpose(-1, -2)
                x_pad = torch.zeros([x.shape[0], x.shape[1], m.shape[-1]], device=x.device)
                x_pad[:, :, : x.shape[-1]] = x
                x = x_pad
        return x.transpose(-1, -2)


class ProsodyPredictor(nn.Module):
    def __init__(self, style_dim, d_hid, nlayers, max_dur, dropout):
        super().__init__()
        self.text_encoder = DurationEncoder(
            sty_dim=style_dim,
            d_model=d_hid,
            nlayers=nlayers,
            dropout=dropout,
        )
        self.lstm = nn.LSTM(
            d_hid + style_dim,
            d_hid // 2,
            1,
            batch_first=True,
            bidirectional=True,
        )
        self.duration_proj = LinearNorm(d_hid, max_dur)
        self.shared = nn.LSTM(
            d_hid + style_dim,
            d_hid // 2,
            1,
            batch_first=True,
            bidirectional=True,
        )
        self.F0 = nn.ModuleList(
            [
                AdainResBlk1d(d_hid, d_hid, style_dim, dropout_p=dropout),
                AdainResBlk1d(d_hid, d_hid // 2, style_dim, upsample=True, dropout_p=dropout),
                AdainResBlk1d(d_hid // 2, d_hid // 2, style_dim, dropout_p=dropout),
            ]
        )
        self.N = nn.ModuleList(
            [
                AdainResBlk1d(d_hid, d_hid, style_dim, dropout_p=dropout),
                AdainResBlk1d(d_hid, d_hid // 2, style_dim, upsample=True, dropout_p=dropout),
                AdainResBlk1d(d_hid // 2, d_hid // 2, style_dim, dropout_p=dropout),
            ]
        )
        self.F0_proj = nn.Conv1d(d_hid // 2, 1, 1, 1, 0)
        self.N_proj = nn.Conv1d(d_hid // 2, 1, 1, 1, 0)

    def forward(self, texts, style, text_lengths, alignment, m):
        d = self.text_encoder(texts, style, text_lengths, m)
        input_lengths = text_lengths.cpu().numpy()
        m = m.to(text_lengths.device).unsqueeze(1)
        x = nn.utils.rnn.pack_padded_sequence(d, input_lengths, batch_first=True, enforce_sorted=False)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x_pad = torch.zeros([x.shape[0], m.shape[-1], x.shape[-1]], device=x.device)
        x_pad[:, : x.shape[1], :] = x
        duration = self.duration_proj(F.dropout(x_pad, 0.5, training=self.training))
        en = d.transpose(-1, -2) @ alignment
        return duration.squeeze(-1), en

    def f0n_train(self, x, s, f0_branch_emb=None, n_branch_emb=None):
        x, _ = self.shared(x.transpose(-1, -2))
        f0 = x.transpose(-1, -2)
        if f0_branch_emb is not None:
            coarse = f0_branch_emb[0] if isinstance(f0_branch_emb, tuple) else f0_branch_emb
            f0 = f0 + coarse
        for i, block in enumerate(self.F0):
            f0 = block(f0, s)
            if i == 0 and isinstance(f0_branch_emb, tuple) and f0_branch_emb[1] is not None:
                f0 = f0 + f0_branch_emb[1]
        f0 = self.F0_proj(f0)
        n = x.transpose(-1, -2)
        if n_branch_emb is not None:
            coarse = n_branch_emb[0] if isinstance(n_branch_emb, tuple) else n_branch_emb
            n = n + coarse
        for i, block in enumerate(self.N):
            n = block(n, s)
            if i == 0 and isinstance(n_branch_emb, tuple) and n_branch_emb[1] is not None:
                n = n + n_branch_emb[1]
        n = self.N_proj(n)
        return f0.squeeze(1), n.squeeze(1)


# ─── HiFiGAN decoder ───────────────────────────────────────────────────────────


class SineGen(nn.Module):
    def __init__(
        self,
        samp_rate,
        upsample_scale,
        harmonic_num,
        sine_amp=0.1,
        noise_std=0.003,
        voiced_threshold=0,
        flag_for_pulse=False,
    ):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold
        self.flag_for_pulse = flag_for_pulse
        self.upsample_scale = upsample_scale

    def _f02uv(self, f0):
        return (f0 > self.voiced_threshold).float()

    def _f02sine(self, f0_values):
        rad_values = (f0_values / self.sampling_rate) % 1
        rand_ini = torch.rand(f0_values.shape[0], f0_values.shape[2], device=f0_values.device)
        rand_ini[:, 0] = 0
        rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini
        if not self.flag_for_pulse:
            rad_values = F.interpolate(
                rad_values.transpose(1, 2),
                scale_factor=1 / self.upsample_scale,
                mode="linear",
            ).transpose(1, 2)
            phase = torch.cumsum(rad_values, dim=1) * 2 * np.pi
            phase = F.interpolate(
                phase.transpose(1, 2) * self.upsample_scale,
                scale_factor=self.upsample_scale,
                mode="linear",
            ).transpose(1, 2)
            sines = torch.sin(phase)
        else:
            uv = self._f02uv(f0_values)
            uv_1 = torch.roll(uv, shifts=-1, dims=1)
            uv_1[:, -1, :] = 1
            u_loc = (uv < 1) * (uv_1 > 0)
            tmp_cumsum = torch.cumsum(rad_values, dim=1)
            for idx in range(f0_values.shape[0]):
                temp_sum = tmp_cumsum[idx, u_loc[idx, :, 0], :]
                temp_sum[1:, :] = temp_sum[1:, :] - temp_sum[0:-1, :]
                tmp_cumsum[idx, :, :] = 0
                tmp_cumsum[idx, u_loc[idx, :, 0], :] = temp_sum
            i_phase = torch.cumsum(rad_values - tmp_cumsum, dim=1)
            sines = torch.cos(i_phase * 2 * np.pi)
        return sines

    def forward(self, f0):
        fn = torch.multiply(
            f0,
            torch.FloatTensor([[range(1, self.harmonic_num + 2)]]).to(f0.device),
        )
        sine_waves = self._f02sine(fn) * self.sine_amp
        uv = self._f02uv(f0)
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        noise = noise_amp * torch.randn_like(sine_waves)
        sine_waves = sine_waves * uv + noise
        return sine_waves, uv, noise


class SourceModuleHnNSF(nn.Module):
    def __init__(
        self, sampling_rate, upsample_scale, harmonic_num, sine_amp=0.1, add_noise_std=0.003, voiced_threshold=0
    ):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = add_noise_std
        self.l_sin_gen = SineGen(
            sampling_rate,
            upsample_scale,
            harmonic_num,
            sine_amp,
            add_noise_std,
            voiced_threshold,
        )
        self.l_linear = nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = nn.Tanh()

    def forward(self, x):
        with torch.no_grad():
            sine_wavs, uv, _ = self.l_sin_gen(x)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))
        noise = torch.randn_like(uv) * self.sine_amp / 3
        return sine_merge, noise, uv


class AdaINResBlock1(nn.Module):
    def __init__(self, channels, kernel_size, dilation, style_dim):
        super().__init__()
        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=_get_padding(kernel_size, dilation[0]),
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=_get_padding(kernel_size, dilation[1]),
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[2],
                        padding=_get_padding(kernel_size, dilation[2]),
                    )
                ),
            ]
        )
        self.convs1.apply(_init_weights)
        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=_get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=_get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=_get_padding(kernel_size, 1),
                    )
                ),
            ]
        )
        self.convs2.apply(_init_weights)
        self.adain1 = nn.ModuleList([AdaIN1d(style_dim, channels) for _ in range(3)])
        self.adain2 = nn.ModuleList([AdaIN1d(style_dim, channels) for _ in range(3)])
        self.alpha1 = nn.ParameterList([nn.Parameter(torch.ones(1, channels, 1)) for _ in range(3)])
        self.alpha2 = nn.ParameterList([nn.Parameter(torch.ones(1, channels, 1)) for _ in range(3)])

    def forward(self, x, s):
        for c1, c2, n1, n2, a1, a2 in zip(
            self.convs1,
            self.convs2,
            self.adain1,
            self.adain2,
            self.alpha1,
            self.alpha2,
        ):
            xt = n1(x, s)
            xt = xt + (1 / a1) * (torch.sin(a1 * xt) ** 2)
            xt = c1(xt)
            xt = n2(xt, s)
            xt = xt + (1 / a2) * (torch.sin(a2 * xt) ** 2)
            xt = c2(xt)
            x = xt + x
        return x


class HiFiGANGenerator(nn.Module):
    def __init__(
        self,
        style_dim,
        resblock_kernel_sizes,
        upsample_rates,
        upsample_initial_channel,
        resblock_dilation_sizes,
        upsample_kernel_sizes,
        sr,
    ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.m_source = SourceModuleHnNSF(
            sampling_rate=sr,
            upsample_scale=np.prod(upsample_rates),
            harmonic_num=8,
            voiced_threshold=10,
        )
        self.f0_upsamp = nn.Upsample(scale_factor=np.prod(upsample_rates))
        self.noise_convs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.noise_res = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            c_cur = upsample_initial_channel // (2 ** (i + 1))
            self.ups.append(
                weight_norm(
                    nn.ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(u // 2 + u % 2),
                        output_padding=u % 2,
                    )
                )
            )
            if i + 1 < len(upsample_rates):
                stride_f0 = np.prod(upsample_rates[i + 1 :])
                self.noise_convs.append(
                    nn.Conv1d(
                        1,
                        c_cur,
                        kernel_size=stride_f0 * 2,
                        stride=stride_f0,
                        padding=(stride_f0 + 1) // 2,
                    )
                )
                self.noise_res.append(
                    AdaINResBlock1(c_cur, 7, [1, 3, 5], style_dim),
                )
            else:
                self.noise_convs.append(
                    nn.Conv1d(1, c_cur, kernel_size=1),
                )
                self.noise_res.append(
                    AdaINResBlock1(c_cur, 11, [1, 3, 5], style_dim),
                )
        self.resblocks = nn.ModuleList()
        self.alphas = nn.ParameterList()
        self.alphas.append(
            nn.Parameter(torch.ones(1, upsample_initial_channel, 1)),
        )
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            self.alphas.append(nn.Parameter(torch.ones(1, ch, 1)))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(AdaINResBlock1(ch, k, d, style_dim))
        self.conv_post = weight_norm(nn.Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(_init_weights)
        self.conv_post.apply(_init_weights)

    def forward(self, x, s, f0):
        f0 = self.f0_upsamp(f0[:, None]).transpose(1, 2)
        har_source, noi_source, uv = self.m_source(f0)
        har_source = har_source.transpose(1, 2)
        for i in range(self.num_upsamples):
            x = x + (1 / self.alphas[i]) * (torch.sin(self.alphas[i] * x) ** 2)
            x_source = self.noise_convs[i](har_source)
            x_source = self.noise_res[i](x_source, s)
            x = self.ups[i](x)
            x = x + x_source
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x, s)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x, s)
            x = xs / self.num_kernels
        x = x + (1 / self.alphas[i + 1]) * (torch.sin(self.alphas[i + 1] * x) ** 2)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x


class HiFiGANDecoder(nn.Module):
    def __init__(
        self,
        dim_in,
        style_dim,
        resblock_kernel_sizes,
        upsample_rates,
        upsample_initial_channel,
        resblock_dilation_sizes,
        upsample_kernel_sizes,
        sr,
    ):
        super().__init__()
        self.encode = AdainResBlk1d(dim_in + 2, 1024, style_dim)
        self.decode = nn.ModuleList(
            [
                AdainResBlk1d(1024 + 2 + 64, 1024, style_dim),
                AdainResBlk1d(1024 + 2 + 64, 1024, style_dim),
                AdainResBlk1d(1024 + 2 + 64, 1024, style_dim),
                AdainResBlk1d(1024 + 2 + 64, 512, style_dim, upsample=True),
            ]
        )
        self.F0_conv = weight_norm(
            nn.Conv1d(1, 1, kernel_size=3, stride=2, groups=1, padding=1),
        )
        self.N_conv = weight_norm(
            nn.Conv1d(1, 1, kernel_size=3, stride=2, groups=1, padding=1),
        )
        self.asr_res = nn.Sequential(
            weight_norm(nn.Conv1d(512, 64, kernel_size=1)),
        )
        self.generator = HiFiGANGenerator(
            style_dim,
            resblock_kernel_sizes,
            upsample_rates,
            upsample_initial_channel,
            resblock_dilation_sizes,
            upsample_kernel_sizes,
            sr,
        )

    def forward(self, asr, f0_curve, n, s):
        f0 = self.F0_conv(f0_curve.unsqueeze(1))
        n = self.N_conv(n.unsqueeze(1))
        x = torch.cat([asr, f0, n], axis=1)
        x = self.encode(x, s)
        asr_res = self.asr_res(asr)
        res = True
        for block in self.decode:
            if res:
                x = torch.cat([x, asr_res, f0, n], axis=1)
            x = block(x, s)
            if block.upsample_type != "none":
                res = False
        x = self.generator(x, s, f0_curve)
        return x


# ─── Top-level wrapper ─────────────────────────────────────────────────────────


class StyleTTS2Model(nn.Module):
    """Unified nn.Module wrapping all StyleTTS2 components.

    Submodule names match the checkpoint component keys:
    bert, bert_encoder, predictor, decoder, text_encoder,
    predictor_encoder, style_encoder, text_aligner, pitch_extractor,
    embed_dur, embed_f0N.
    """

    def __init__(self, config: GraniteStyleTTS2Config | None = None):
        super().__init__()
        if config is None:
            config = GraniteStyleTTS2Config()
        self.config = config

        self.bert = CustomAlbert(
            AlbertConfig(
                vocab_size=178,
                hidden_size=_PLBERT_HIDDEN_SIZE,
                num_attention_heads=12,
                intermediate_size=2048,
                max_position_embeddings=512,
                num_hidden_layers=12,
            )
        )

        self.bert_encoder = nn.Linear(
            _PLBERT_HIDDEN_SIZE,
            config.hidden_dim,
        )

        self.predictor = ProsodyPredictor(
            style_dim=config.style_dim,
            d_hid=config.hidden_dim,
            nlayers=config.n_layer,
            max_dur=config.max_dur,
            dropout=config.dropout,
        )

        self.decoder = HiFiGANDecoder(
            dim_in=config.hidden_dim,
            style_dim=config.style_dim,
            resblock_kernel_sizes=config.resblock_kernel_sizes,
            upsample_rates=config.upsample_rates,
            upsample_initial_channel=config.upsample_initial_channel,
            resblock_dilation_sizes=config.resblock_dilation_sizes,
            upsample_kernel_sizes=config.upsample_kernel_sizes,
            sr=config.sr,
        )

        self.text_encoder = TextEncoder(
            channels=config.hidden_dim,
            kernel_size=5,
            depth=config.n_layer,
            n_symbols=config.n_token,
        )

        self.predictor_encoder = StyleEncoder(
            dim_in=config.dim_in,
            style_dim=config.style_dim,
            max_conv_dim=config.hidden_dim,
        )
        self.style_encoder = StyleEncoder(
            dim_in=config.dim_in,
            style_dim=config.style_dim,
            max_conv_dim=config.hidden_dim,
        )

        self.text_aligner = ASRCNN(
            input_dim=80,
            hidden_dim=256,
            n_token=178,
            token_embedding_dim=512,
        )

        self.pitch_extractor = JDCNet(num_class=1, seq_len=192)

        self.embed_dur = nn.Embedding(
            514,
            config.hidden_dim,
            padding_idx=-1,
        )
        self.embed_f0N = nn.Embedding(
            514,
            (config.hidden_dim + config.style_dim) // 4,
            padding_idx=-1,
        )

    @torch.inference_mode()
    def inference(
        self,
        tokens: torch.Tensor,
        ref_s: torch.Tensor,
        ref_p: torch.Tensor,
        prsinf: torch.Tensor,
        boundaries: torch.Tensor,
        ref_p_f0n: torch.Tensor | None = None,
        duration_scale: float = 1.0,
    ) -> torch.Tensor:
        """Run full StyleTTS2 inference pipeline.

        Args:
            tokens: (1, T_phon) phoneme token IDs
            ref_s: (1, style_dim) acoustic style embedding (speaker)
            ref_p: (1, style_dim) predictor style embedding (duration)
            prsinf: (n_words+1, 5) prosody bins [dur, h1, h2, h3, h4]
            boundaries: (2, n_words) word start/end phoneme indices
            ref_p_f0n: (1, style_dim) optional separate F0/N style
            duration_scale: multiply predicted durations (>1 = slower)

        Returns:
            1-D waveform tensor at 24 kHz
        """
        device = tokens.device
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
        text_mask = _length_to_mask(input_lengths).to(device)

        t_en = self.text_encoder(tokens, input_lengths, text_mask)
        bert_dur = self.bert(tokens, attention_mask=(~text_mask).int())
        d_en = self.bert_encoder(bert_dur).transpose(-1, -2)

        starts, ends = boundaries[0], boundaries[1] - 1
        num_phon = ends - starts + 1
        prsinf_d = prsinf[:, 0].repeat_interleave(num_phon, dim=0)
        dur_wd_emb = self.embed_dur(prsinf_d)
        d_en = d_en + dur_wd_emb.transpose(-1, -2)

        d = self.predictor.text_encoder(d_en, ref_p, input_lengths, text_mask)
        x, _ = self.predictor.lstm(d)
        duration = self.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1)
        pred_dur = torch.round(duration.squeeze()).clamp(min=1)
        if duration_scale != 1.0:
            pred_dur = torch.round(pred_dur * duration_scale).clamp(min=1)
        num_sil = int(pred_dur[-2:].sum())

        pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data), device=device)
        c_frame = 0
        for i in range(pred_aln_trg.size(0)):
            dur_i = int(pred_dur[i].data)
            pred_aln_trg[i, c_frame : c_frame + dur_i] = 1
            c_frame += dur_i

        prefix_sum = pred_aln_trg.int().sum(dim=-1).cumsum(dim=0)
        sums = torch.where(
            starts == 0,
            prefix_sum[ends],
            prefix_sum[ends] - prefix_sum[starts - 1],
        )
        prsinf_p = prsinf[:, 1:].repeat_interleave(sums, dim=0)

        en = d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0)
        asr_new = torch.zeros_like(en)
        asr_new[:, :, 0] = en[:, :, 0]
        asr_new[:, :, 1:] = en[:, :, 0:-1]
        en = asr_new

        f0n_emb = self.embed_f0N(prsinf_p.transpose(0, 1).unsqueeze(0)).transpose(-1, -2)
        f0n_emb = f0n_emb.reshape(f0n_emb.size(0), -1, f0n_emb.size(-1))
        en = en + f0n_emb

        f0n_style = ref_p_f0n if ref_p_f0n is not None else ref_p
        f0_pred, n_pred = self.predictor.f0n_train(en, f0n_style)
        f0_pred[0, -num_sil:] = 0
        n_pred[0, -num_sil:] = -6

        asr = t_en @ pred_aln_trg.unsqueeze(0)
        asr_new = torch.zeros_like(asr)
        asr_new[:, :, 0] = asr[:, :, 0]
        asr_new[:, :, 1:] = asr[:, :, 0:-1]
        asr = asr_new

        out = self.decoder(asr, f0_pred, n_pred, ref_s.squeeze().unsqueeze(0))
        return out.squeeze()[..., :-50]


# ─── vLLM Omni model wrapper ─────────────────────────────────────────────────


class GraniteStyleTTS2Decoder(nn.Module):
    have_multimodal_outputs = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config: GraniteStyleTTS2Config = vllm_config.model_config.hf_config
        self.model = StyleTTS2Model(self.config)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        **kwargs: Any,
    ) -> OmniOutput:
        from vllm_omni.model_executor.stage_input_processors.granite_prosody_lm import (
            unpack_tts_payload,
        )

        if input_ids.numel() < 9:
            return self._warmup_output(input_ids)

        packed = input_ids.cpu().long()
        h = packed[:8].tolist()
        n_ph = int(h[0])
        expected_min = 8 + n_ph + int(h[1]) * int(h[2]) + int(h[3]) * int(h[4]) + int(h[7])
        if n_ph <= 0 or expected_min > packed.numel():
            return self._warmup_output(input_ids)

        device = input_ids.device
        sample_rate = torch.tensor(
            self.config.sr,
            dtype=torch.int32,
            device=device,
        )

        tts_data = unpack_tts_payload(packed)

        phoneme_tokens = tts_data["phoneme_tokens"].to(device)
        prsinf = tts_data["prsinf"].to(device)
        boundaries = tts_data["boundaries"].to(device)
        spk_emb = tts_data["speaker_embedding"].to(device)
        ref_p = torch.zeros_like(spk_emb)

        wav = self.model.inference(
            tokens=phoneme_tokens,
            ref_s=spk_emb,
            ref_p=ref_p,
            prsinf=prsinf,
            boundaries=boundaries,
        )

        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={
                "audio": [wav.to(device)],
                "sample_rate": [sample_rate],
            },
        )

    def _warmup_output(self, input_ids: torch.Tensor) -> OmniOutput:
        """Return empty output during warmup/dummy_run."""
        device = input_ids.device
        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={
                "audio": [torch.zeros(0, dtype=torch.float32, device=device)],
                "sample_rate": [
                    torch.tensor(self.config.sr, dtype=torch.int32, device=device),
                ],
            },
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: Any = None,
    ) -> torch.Tensor | None:
        return None

    def embed_input_ids(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return torch.zeros(
            input_ids.shape[0],
            self.config.hidden_size,
            device=input_ids.device,
            dtype=torch.float32,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params = dict(self.model.named_parameters())
        buffers = dict(self.model.named_buffers())
        loaded: set[str] = set()
        for name, tensor in weights:
            target = params.get(name)
            if target is None:
                target = buffers.get(name)
            if target is not None:
                target.data.copy_(tensor)
                loaded.add(f"model.{name}")
        return loaded
