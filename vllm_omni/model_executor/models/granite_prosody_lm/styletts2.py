# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""StyleTTS2 model — nn.Module wrapper for vLLM Omni.

All component module definitions are ported with __init__ methods matching
the reference exactly so state_dict keys align with exported checkpoints.
Forward methods are stubs — inference is not yet implemented.

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

import numpy as np
import torch
import torch.nn as nn
import torchaudio.functional as audio_F
from torch.nn.utils import spectral_norm, weight_norm
from transformers import AlbertConfig, AlbertModel
from vllm.config import VllmConfig

from vllm_omni.transformers_utils.configs.granite_styletts2 import (
    GraniteStyleTTS2Config,
)

logger = logging.getLogger(__name__)

_PLBERT_HIDDEN_SIZE = 768


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
        raise NotImplementedError


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
        raise NotImplementedError


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
        raise NotImplementedError


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
        raise NotImplementedError


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
        raise NotImplementedError


class MFCC(nn.Module):
    def __init__(self, n_mfcc=40, n_mels=80):
        super().__init__()
        dct_mat = audio_F.create_dct(n_mfcc, n_mels, "ortho")
        self.register_buffer("dct_mat", dct_mat)

    def forward(self, mel_specgram):
        raise NotImplementedError


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
        raise NotImplementedError


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
        raise NotImplementedError


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
        raise NotImplementedError


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
        raise NotImplementedError


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
        raise NotImplementedError


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
        raise NotImplementedError


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

    def forward(self, x):
        raise NotImplementedError


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
        raise NotImplementedError


class LayerNorm1d(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        raise NotImplementedError


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
        raise NotImplementedError


class AdaIN1d(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm1d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features * 2)

    def forward(self, x, s):
        raise NotImplementedError


class UpSample1d(nn.Module):
    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        raise NotImplementedError


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

    def forward(self, x, s):
        raise NotImplementedError


class AdaLayerNorm(nn.Module):
    def __init__(self, style_dim, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.fc = nn.Linear(style_dim, channels * 2)

    def forward(self, x, s):
        raise NotImplementedError


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
        raise NotImplementedError


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

    def forward(self, *args, **kwargs):
        raise NotImplementedError


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

    def forward(self, f0):
        raise NotImplementedError


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
        raise NotImplementedError


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
        raise NotImplementedError


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
        raise NotImplementedError


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
        raise NotImplementedError


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

    def forward(self, *args, **kwargs):
        raise NotImplementedError("StyleTTS2Model.forward() not yet implemented.")


# ─── vLLM Omni model wrapper ─────────────────────────────────────────────────


class GraniteStyleTTS2Decoder(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config: GraniteStyleTTS2Config = vllm_config.model_config.hf_config
        self.model = StyleTTS2Model(config)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, **kwargs):
        raise NotImplementedError("GraniteStyleTTS2Decoder.forward() not yet implemented.")

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        raise NotImplementedError("GraniteStyleTTS2Decoder.compute_logits() not yet implemented.")

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("GraniteStyleTTS2Decoder.embed_input_ids() not yet implemented.")

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
