import torch
import torch.nn as nn
from .Model import Model

import numpy as np
from numpy.random import RandomState

class ConvKB(Model):
    def __init__(self, ent_tot, rel_tot, embed_dim=100, n_filters=64, kernel_size=1, drop=0.2):
        super().__init__(ent_tot, rel_tot)
        self.embed_dim = embed_dim
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.drop = drop

        self.ent_embeddings = nn.Embedding(self.ent_tot, self.embed_dim)
        self.rel_embeddings = nn.Embedding(self.rel_tot, self.embed_dim)

        self.conv1_bn = nn.BatchNorm2d(1)
        self.conv_layer = nn.Conv2d(1, self.n_filters, (self.kernel_size, 3))
        self.conv2_bn = nn.BatchNorm2d(self.n_filters)
        self.dropout = nn.Dropout(self.drop)
        self.non_linearity = nn.ReLU()
        self.fc_layer = nn.Linear((self.embed_dim - self.kernel_size + 1) * self.n_filters, 1, bias=False)

        nn.init.xavier_uniform_(self.fc_layer.weight, gain=1.414)
        nn.init.xavier_uniform_(self.conv_layer.weight, gain=1.414)
    
    def _calc(self, h, r, t):
        h = h.unsqueeze(1)
        r = r.unsqueeze(1)
        t = t.unsqueeze(1)
        conv_input = torch.cat([h, r, t], 1)  # bs x 3 x dim
        conv_input = conv_input.transpose(1, 2)
        # To make tensor of size 4, where second dim is for input channels
        conv_input = conv_input.unsqueeze(1)
        conv_input = self.conv1_bn(conv_input)
        out_conv = self.conv_layer(conv_input)
        out_conv = self.conv2_bn(out_conv)
        out_conv = self.non_linearity(out_conv)
        out_conv = out_conv.view(-1, (self.embed_dim - self.kernel_size + 1) * self.n_filters)
        input_fc = self.dropout(out_conv)
        score = self.fc_layer(input_fc).view(-1)
        return -score

    def regularization(self, data):
        # (h, r, t) embedding
        h = self.ent_embeddings(data['batch_h'])
        r = self.rel_embeddings(data['batch_r'])
        t = self.ent_embeddings(data['batch_t'])
        l2_reg = torch.mean(h ** 2) + torch.mean(t ** 2) + torch.mean(r ** 2)
        for W in self.conv_layer.parameters():
            l2_reg = l2_reg + W.norm(2)
        for W in self.fc_layer.parameters():
            l2_reg = l2_reg + W.norm(2)
        return l2_reg

    def forward(self, data):
        # (h, r, t) embedding
        h = self.ent_embeddings(data['batch_h'])
        r = self.rel_embeddings(data['batch_r'])
        t = self.ent_embeddings(data['batch_t'])
        score = self._calc(h, r, t)

        return score

    def predict(self, data):
        # (h, r, t) embedding
        h = self.ent_embeddings(data['batch_h'])
        r = self.rel_embeddings(data['batch_r'])
        t = self.ent_embeddings(data['batch_t'])
        score = self._calc(h, r, t)

        return score.cpu().data.numpy()

