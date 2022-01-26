import torch
import torch.nn as nn
from .Model import Model

class LineaRE(Model):
    def __init__(self, ent_tot, rel_tot, ent_dim=100, rel_dim=100, p_norm = 1, norm_flag = True, margin = None, epsilon = None):
        super().__init__(ent_tot, rel_tot)
        self.ent_dim = ent_dim
        self.rel_dim = rel_dim
        self.margin = margin
        self.epsilon = epsilon
        self.norm_flag = norm_flag
        self.p_norm = p_norm

        self.ent_embeddings = nn.Embedding(self.ent_tot, self.ent_dim)
        self.rel_embeddings = nn.Embedding(self.rel_tot, self.rel_dim)

        self.wrh = nn.Embedding(self.rel_tot, self.rel_dim, max_norm=self.rel_dim, norm_type=1)
        self.wrt = nn.Embedding(self.rel_tot, self.rel_dim, max_norm=self.rel_dim, norm_type=1)

        nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_embeddings.weight.data)

        nn.init.zeros_(self.wrh.weight)
        nn.init.zeros_(self.wrt.weight)

    def forward(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        mode = data['mode']

        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        wh = self.wrh(batch_r)
        wt = self.wrt(batch_r)

        if mode == "head_batch":
            score = wh * h + (r - wt * t)
        elif mode == "tail_batch":
            score = (wh * h + r) - wt * t 
        else:
            raise ValueError("mode %s not supported" % mode)

        score = torch.norm(score, self.p_norm, -1).flatten()
        if self.margin_flag:
            return self.margin - score
        else:
            return score

    def regularization(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        wh = self.wrh(batch_r)
        wt = self.wrt(batch_r)

        regul = (torch.mean(h ** 2) +
                 torch.mean(t ** 2) +
                 torch.mean(r ** 2) +
                 torch.mean(wh ** 2) +
                 torch.mean(wt ** 2)) / 5
        
        return regul

    def predict(self, data):
        score = self.forward(data)
        if self.margin_flag:
            score = self.margin - score
            return score.cpu().data.numpy()
        else:
            return score.cpu().data.numpy()

