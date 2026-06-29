import torch.nn as nn
import torch
from mask3d_spconv.position_embedding import PositionEmbeddingCoordsSine
from torch.cuda.amp import autocast

class PPONet(nn.Module):
    def __init__(self, L=3, K=1+10):
        super(PPONet, self).__init__()
        self.point_features_head = nn.Linear(256, 256)
        self.attention, self.FFN = nn.ModuleList([]), nn.ModuleList([])
        for _ in range(L):
            self.attention.append(SelfAttentionLayer(d_model=256, nhead=8, normalize_before=True))
            self.FFN.append(FFNLayer(d_model=256, dim_feedforward=256, normalize_before=True))
        self.proj_pos = nn.Sequential(nn.Linear(256, 256), nn.LeakyReLU(), nn.Linear(256, 256))
        self.pos_enc = PositionEmbeddingCoordsSine(pos_type="sine", d_pos=256, gauss_scale=1.0, normalize=True).cuda()
        self.seed_indicator = nn.Embedding(2, 256)
        self.action_head = nn.Sequential(nn.Linear(256, 256), nn.LeakyReLU(), nn.Linear(256, 1))

        self.fc = nn.Sequential(nn.Linear(in_features=256, out_features=32), nn.LeakyReLU())
        self.value_head = nn.Sequential(nn.Linear(in_features=32*K, out_features=32*K), nn.BatchNorm1d(32*K), nn.LeakyReLU(),
                                  nn.Linear(in_features=32*K, out_features=1))

    def forward(self, sampled_env_xyz, sampled_env_feats, history):
        ## the 1st is cur_sp, rest are nbr_sp, shape: [B, N, C]
        rl_batch_size = sampled_env_xyz.shape[0]
        mins, maxs = sampled_env_xyz.min(dim=1)[0], sampled_env_xyz.max(dim=1)[0] ##[K, 3], [K, 3]
        maxs[maxs==mins]+=0.1
        ## sometimes, some max value will equals to their min value, like moved to a large ground, causing the later normalization be nan
        with autocast(enabled=False):
            point_pos = self.pos_enc(sampled_env_xyz.float(), input_range=[mins, maxs]).permute((0, 2, 1))
        point_feats = self.point_features_head(sampled_env_feats)
        point_pos = self.proj_pos(point_pos)
        ###
        point_feats[:, 0, :]  += self.seed_indicator.weight[0]
        point_feats[:, 1:, :]  += self.seed_indicator.weight[1]
        ###
        for i in range(len(self.attention)):
            output = self.attention[i](tgt = point_feats, query_pos = point_pos)### [bs, 1+5, 256]
            output = self.FFN[i](output)### [bs, 1+5, 256]

        x = self.fc(output)
        return self.action_head(output).squeeze(2), self.value_head(x.reshape(x.shape[0], -1)).squeeze(1)


class SelfAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0, normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.normalize_before = normalize_before
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, tgt_mask=None, tgt_key_padding_mask=None, query_pos=None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt, tgt_mask=None, tgt_key_padding_mask=None, query_pos=None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt, tgt_mask=None, tgt_key_padding_mask=None, query_pos=None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask, tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask, tgt_key_padding_mask, query_pos)


class FFNLayer(nn.Module):
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0, normalize_before=False):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)
        self.activation = nn.GELU(approximate='tanh')
        self.normalize_before = normalize_before
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


