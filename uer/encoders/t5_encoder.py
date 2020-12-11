# -*- encoding:utf-8 -*-
import torch
import torch.nn as nn
from uer.layers.layer_norm import LayerNorm
from uer.layers.position_ffn import PositionwiseFeedForward
from uer.layers.multi_headed_attn import MultiHeadedAttention
from uer.layers.transformer import *

class T5Encoder(nn.Module):
    """
    T5 encoder exploits 12 or 24 gptblock layers to extract features.
    """
    def __init__(self, args):
        super(T5Encoder, self).__init__()
        self.layers_num = args.layers_num
        self.encoder_block = nn.ModuleList([
            GptBlock(args) for _ in range(self.layers_num)
        ])

        self.layer_norm = LayerNorm(args.hidden_size)

    def forward(self, emb, seg):
        """
        Args:
            emb: [batch_size x seq_length x emb_size]
            seg: [batch_size x seq_length]

        Returns:
            hidden: [batch_size x seq_length x hidden_size]
        """

        batch_size, seq_length, _ = emb.size()
        # Generate mask according to segment indicators.
        # mask: [batch_size x 1 x seq_length x seq_length]

        mask = (seg > 0). \
            unsqueeze(1). \
            repeat(1, seq_length, 1). \
            unsqueeze(1)

        mask = mask.float()
        encoder_mask = (1.0 - mask) * -10000.0

        hidden = emb
        for i in range(self.layers_num):
            hidden = self.encoder_block[i](hidden, encoder_mask)

        hidden =  self.layer_norm(hidden)

        return (emb, hidden)