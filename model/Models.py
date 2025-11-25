"""
Models.py - 时空事件序列建模核心模型模块

本模块实现了基于Transformer架构的时空事件序列建模方法，主要用于处理具有时间和空间属性的事件数据。
该模块是SMASH（Spatio-temporal Multi-Attention Sequence Modeling）项目的核心组件。

主要组件：
===========

1. 工具函数
-----------
- get_non_pad_mask(): 生成非填充位置的掩码，用于区分真实数据和填充数据
- get_attn_key_pad_mask(): 生成注意力机制中key的填充掩码，防止注意力关注到填充位置
- get_subsequent_mask(): 生成后续位置掩码，实现因果注意力（只能看到过去，看不到未来）

2. 编码器类
-----------
Encoder: 
    基础编码器，使用自注意力机制编码事件序列
    - 支持时间编码（正弦-余弦位置编码）
    - 支持空间位置嵌入（通过多层MLP）
    - 支持事件类型嵌入（可选，当loc_dim=3时）
    - 使用掩码自注意力防止信息泄露

Encoder_ST (Spatio-Temporal):
    增强版编码器，分别处理时间、空间和事件标记信息
    - 三个独立的编码器层栈：时间、空间、标记（可选）
    - 允许时间和空间信息独立建模并融合
    - 提供更细粒度的时空特征表示
    - 支持2维（时间+空间）和3维（时间+空间+类型）事件

3. 辅助层
---------
RNN_layers:
    可选的循环层，在Transformer编码器之后添加LSTM层
    - 使用LSTM捕获序列的长期依赖
    - 通过pack_padded_sequence高效处理变长序列
    - 线性投影层将RNN输出映射回模型维度

4. 完整模型
-----------
Transformer:
    基础的序列到序列模型，组合编码器和RNN层
    - 整合Encoder和RNN_layers
    - 包含可学习的alpha和beta参数用于时间建模
    - 输出隐藏表示用于下游预测任务

Transformer_ST:
    时空分离的Transformer模型，是本项目的主要模型
    - 使用Encoder_ST进行时空分离编码
    - 为时间、空间和整体表示分别配备RNN层
    - 将多个表示拼接作为最终输出（维度：3或4倍的d_model）
    - 适用于需要细粒度时空建模的复杂场景

数据流说明：
===========
输入：
    - event_loc: [batch, seq_len, loc_dim]，事件的空间位置（2维或3维）
    - event_time: [batch, seq_len]，事件发生的时间戳

处理流程：
    1. 生成非填充掩码和注意力掩码
    2. 时间编码（正弦-余弦位置编码）
    3. 空间位置嵌入（多层MLP）
    4. 事件类型嵌入（如果有，nn.Embedding）
    5. 多层自注意力编码
    6. RNN后处理
    7. 输出最终的事件序列表示

输出：
    - enc_output: [batch, seq_len, d_model]或拼接后的更高维表示
    - non_pad_mask: [batch, seq_len, 1]，用于后续计算

应用场景：
=========
- 犯罪事件预测（crime dataset）
- 地震序列建模（earthquake dataset）
- 体育赛事分析（football dataset）
- 其他时空点过程建模任务

模型参数说明：
============
- d_model: 模型隐藏层维度（默认256）
- d_inner: FFN内部维度（默认1024）
- n_layers: Transformer层数（默认4）
- n_head: 多头注意力的头数（默认4）
- d_k, d_v: 注意力的key和value维度（默认64）
- dropout: dropout比例（默认0.1）
- loc_dim: 位置维度，2表示(x,y)，3表示(mark,x,y)
- num_types: 事件类型数量（当loc_dim=3时使用）

注意事项：
=========
1. 所有序列必须进行padding对齐到相同长度
2. 时间戳应该是相对时间且归一化后的时间
3. 空间坐标建议归一化到合理范围
4. 使用因果掩码确保模型只能看到历史信息
5. device参数需要与数据所在设备一致

最后更新：2025年11月
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import model.Constants as Constants
from model.Layers import EncoderLayer


def get_non_pad_mask(seq):
    """ Get the non-padding positions. """

    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)


def get_attn_key_pad_mask(seq_k, seq_q):
    """ For masking out the padding part of key sequence. """

    # expand to fit the shape of key query attention matrix
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1, -1)  # b x lq x lk
    return padding_mask


def get_subsequent_mask(seq, dim=2):
    """ For masking out the subsequent info, i.e., masked self-attention. """

    sz_b, len_s = seq.size()[:2]
    subsequent_mask = torch.triu(torch.ones((dim, len_s, len_s), device=seq.device, dtype=torch.uint8),
                                 diagonal=1).permute(1, 2, 0)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1, -1)  # b x ls x ls
    return subsequent_mask


class Encoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(self, d_model, d_inner, n_layers, n_head, d_k, d_v, dropout, device, loc_dim, num_types):
        super().__init__()

        self.d_model = d_model
        self.loc_dim = loc_dim

        # position vector, used for temporal encoding
        self.position_vec = torch.tensor([math.pow(10000.0, 2.0 * (i // 2) / d_model) for i in range(d_model)],
                                         device=device)

        if loc_dim == 3:
            self.event_mark_emb = nn.Embedding(num_types + 1, d_model, padding_idx=0)

        # event loc embedding
        self.event_emb = nn.Sequential(
            nn.Linear(2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, normalize_before=False)
            for _ in range(n_layers)
        ])

        self.layer_stack_temporal = nn.Modulelist([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, normalize_before=False)
            for _ in range(n_layers)
        ])

    def temporal_enc(self, time, non_pad_mask):
        """
        Input: batch*seq_len.
        Output: batch*seq_len*d_model.
        """
        result = time.unsqueeze(-1) / self.position_vec
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        return result * non_pad_mask

    def forward(self, event_loc, event_time, non_pad_mask):
        """ Encode event sequences via masked self-attention. """

        # prepare attention masks
        # slf_attn_mask is where we cannot look, i.e., the future and the padding
        event_loc, event_mark = event_loc[:, :, 1:], event_loc[:, :, :1]

        slf_attn_mask_subseq = get_subsequent_mask(event_loc, dim=2)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=event_loc, seq_q=event_loc)
        slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(slf_attn_mask_subseq)

        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

        tem_enc = self.temporal_enc(event_time, non_pad_mask)
        mark_enc = self.event_type_emb(event_mark)
        enc_output = self.event_emb(event_loc)

        slf_attn_mask = slf_attn_mask[:, :, :, 0]

        for enc_layer in self.layer_stack:
            enc_output += tem_enc + mark_enc
            enc_output, _ = enc_layer(enc_output, non_pad_mask=non_pad_mask, slf_attn_mask=slf_attn_mask)
        return enc_output


class Encoder_ST(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(self,
                 d_model,
                 d_inner,
                 n_layers,
                 n_head,
                 d_k,
                 d_v,
                 dropout,
                 device,
                 loc_dim,
                 CosSin=False,
                 num_types=1):
        super().__init__()

        self.d_model = d_model
        self.loc_dim = loc_dim

        # position vector, used for temporal encoding
        self.position_vec = torch.tensor([math.pow(10000.0, 2.0 * (i // 2) / d_model) for i in range(d_model)],
                                         device=device)

        # event loc embedding
        self.event_emb_temporal = nn.Sequential(
            nn.Linear(1, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        self.event_emb_loc = nn.Sequential(
            nn.Linear(2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, normalize_before=False)
            for _ in range(n_layers)
        ])

        self.layer_stack_loc = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, normalize_before=False)
            for _ in range(n_layers)
        ])

        self.layer_stack_temporal = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, normalize_before=False)
            for _ in range(n_layers)
        ])

        if loc_dim == 3:
            self.event_emb_mark = nn.Embedding(num_types + 1, d_model, padding_idx=0)
            self.layer_stack_mark = nn.ModuleList([
                EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, normalize_before=False)
                for _ in range(n_layers)
            ])

    def temporal_enc(self, time, non_pad_mask):
        """
        Input: batch*seq_len.
        Output: batch*seq_len*d_model.
        """
        self.position_vec = self.position_vec.to(time)
        result = time.unsqueeze(-1) / self.position_vec
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        return result * non_pad_mask

    def forward(self, event_loc, event_time, non_pad_mask):
        """ Encode event sequences via masked self-attention. """

        if self.loc_dim == 3:
            event_loc, event_mark = event_loc[:, :, 1:], event_loc[:, :, 0]

        # prepare attention masks
        # slf_attn_mask is where we cannot look, i.e., the future and the padding
        slf_attn_mask_subseq = get_subsequent_mask(event_loc, dim=2)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=event_loc, seq_q=event_loc)
        slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(slf_attn_mask_subseq)

        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

        enc_output_temporal = self.temporal_enc(event_time, non_pad_mask)

        enc_output_loc = self.event_emb_loc(event_loc)

        if self.loc_dim == 3:
            enc_output_mark = self.event_emb_mark(event_mark.long())
            enc_output = enc_output_temporal + enc_output_loc + enc_output_mark
        else:
            enc_output = enc_output_temporal + enc_output_loc

        slf_attn_mask = slf_attn_mask[:, :, :, 0]
        for index in range(len(self.layer_stack)):

            enc_output_loc, _ = self.layer_stack_loc[index](enc_output_loc,
                                                            non_pad_mask=non_pad_mask,
                                                            slf_attn_mask=slf_attn_mask)

            enc_output_temporal, _ = self.layer_stack_temporal[index](enc_output_temporal,
                                                                      non_pad_mask=non_pad_mask,
                                                                      slf_attn_mask=slf_attn_mask)

            enc_output, _ = self.layer_stack[index](enc_output, non_pad_mask=non_pad_mask, slf_attn_mask=slf_attn_mask)

            if self.loc_dim == 3:
                enc_output_mark, _ = self.layer_stack_mark[index](enc_output_mark,
                                                                  non_pad_mask=non_pad_mask,
                                                                  slf_attn_mask=slf_attn_mask)
            else:
                enc_output_mark = None

        return enc_output, enc_output_temporal, enc_output_loc, enc_output_mark


class RNN_layers(nn.Module):
    """
    Optional recurrent layers. This is inspired by the fact that adding
    recurrent layers on top of the Transformer helps language modeling.
    """

    def __init__(self, d_model, d_rnn):
        super().__init__()

        self.rnn = nn.LSTM(d_model, d_rnn, num_layers=1, batch_first=True)
        self.projection = nn.Linear(d_rnn, d_model)

    def forward(self, data, non_pad_mask):
        lengths = non_pad_mask.squeeze(2).long().sum(1).cpu()
        pack_enc_output = nn.utils.rnn.pack_padded_sequence(data, lengths, batch_first=True, enforce_sorted=False)
        temp = self.rnn(pack_enc_output)[0]
        out = nn.utils.rnn.pad_packed_sequence(temp, batch_first=True)[0]

        out = self.projection(out)
        return out


class Transformer(nn.Module):
    """ A sequence to sequence model with attention mechanism. """

    def __init__(self,
                 d_model=256,
                 d_rnn=128,
                 d_inner=1024,
                 n_layers=4,
                 n_head=4,
                 d_k=64,
                 d_v=64,
                 dropout=0.1,
                 device=None,
                 loc_dim=2):
        super().__init__()

        self.encoder = Encoder(d_model=d_model,
                               d_inner=d_inner,
                               n_layers=n_layers,
                               n_head=n_head,
                               d_k=d_k,
                               d_v=d_v,
                               dropout=dropout,
                               device=device,
                               loc_dim=loc_dim)

        # parameter for the weight of time difference
        self.alpha = nn.Parameter(torch.tensor(-0.1))

        # parameter for the softplus function
        self.beta = nn.Parameter(torch.tensor(1.0))

        # OPTIONAL recurrent layer, this sometimes helps
        self.rnn = RNN_layers(d_model, d_rnn)

    def forward(self, event_loc, event_time):
        """
        Return the hidden representations and predictions.
        For a sequence (l_1, l_2, ..., l_N), we predict (l_2, ..., l_N, l_{N+1}).
        Input: event_loc: batch*seq_len*2;
               event_time: batch*seq_len.
        Output: enc_output: batch*seq_len*model_dim
        """

        non_pad_mask = get_non_pad_mask(event_time)

        enc_output = self.encoder(event_loc, event_time, non_pad_mask)
        enc_output = self.rnn(enc_output, non_pad_mask)

        return enc_output, non_pad_mask


class Transformer_ST(nn.Module):
    """ A sequence to sequence model with attention mechanism. """

    def __init__(self,
                 d_model=256,
                 d_rnn=128,
                 d_inner=1024,
                 n_layers=4,
                 n_head=4,
                 d_k=64,
                 d_v=64,
                 dropout=0.1,
                 device=None,
                 loc_dim=2,
                 CosSin=False,
                 num_types=1):
        super().__init__()

        self.encoder = Encoder_ST(d_model=d_model,
                                  d_inner=d_inner,
                                  n_layers=n_layers,
                                  n_head=n_head,
                                  d_k=d_k,
                                  d_v=d_v,
                                  dropout=dropout,
                                  device=device,
                                  loc_dim=loc_dim,
                                  CosSin=CosSin,
                                  num_types=num_types)

        # parameter for the weight of time difference
        self.alpha = nn.Parameter(torch.tensor(-0.1))

        # parameter for the softplus function
        self.beta = nn.Parameter(torch.tensor(1.0))

        # OPTIONAL recurrent layer, this sometimes helps
        self.rnn = RNN_layers(d_model, d_rnn)
        self.rnn_temporal = RNN_layers(d_model, d_rnn)
        self.rnn_spatial = RNN_layers(d_model, d_rnn)
        if loc_dim == 3:
            self.rnn_mark = RNN_layers(d_model, d_rnn)

    def forward(self, event_loc, event_time):
        """
        Return the hidden representations and predictions.
        For a sequence (l_1, l_2, ..., l_N), we predict (l_2, ..., l_N, l_{N+1}).
        Input: event_loc: batch*seq_len*2;
               event_time: batch*seq_len.
        Output: enc_output: batch*seq_len*model_dim
        """

        non_pad_mask = get_non_pad_mask(event_time)

        enc_output, enc_output_temporal, enc_output_loc, enc_output_mark = self.encoder(
            event_loc, event_time, non_pad_mask)

        assert (enc_output != enc_output_temporal).any() & (enc_output != enc_output_loc).any() & (
            enc_output_loc != enc_output_temporal).any()

        enc_output = self.rnn(enc_output, non_pad_mask)
        enc_output_temporal = self.rnn_temporal(enc_output_temporal, non_pad_mask)
        enc_output_loc = self.rnn_spatial(enc_output_loc, non_pad_mask)
        if enc_output_mark is not None:
            enc_output_mark = self.rnn_mark(enc_output_mark, non_pad_mask)
            enc_output_all = torch.cat((enc_output_temporal, enc_output_loc, enc_output, enc_output_mark), dim=-1)
        else:
            enc_output_all = torch.cat((enc_output_temporal, enc_output_loc, enc_output), dim=-1)

        return enc_output_all, non_pad_mask
