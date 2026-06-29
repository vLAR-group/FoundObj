import torch.nn as nn
import torch
from mask3d_spconv.position_embedding import PositionEmbeddingCoordsSine
from torch.cuda.amp import autocast

class PPONet(nn.Module):
    def __init__(self, L=3):
        super(PPONet, self).__init__()
        self.point_features_head = nn.Linear(128, 128)
        self.attention, self.FFN = nn.ModuleList([]), nn.ModuleList([])
        for _ in range(L):
            self.attention.append(SelfAttentionLayer(d_model=128, nhead=8, normalize_before=True))
            self.FFN.append(FFNLayer(d_model=128, dim_feedforward=128, normalize_before=True))
        self.proj_pos = nn.Sequential(nn.Linear(128, 128), nn.LeakyReLU(), nn.Linear(128, 128))
        self.pos_enc = PositionEmbeddingCoordsSine(pos_type="sine", d_pos=128, gauss_scale=1.0, normalize=True).cuda()
        self.seed_indicator = nn.Embedding(2, 128)
        self.action_head = nn.Sequential(nn.Linear(128, 128), nn.LeakyReLU(), nn.Linear(128, 1))
        self.value_token = nn.Embedding(2, 128) ## feats and pos
        self.value_head = nn.Linear(in_features=128, out_features=1)

        self.step0_action_head = nn.Sequential(nn.Linear(128, 128), nn.LayerNorm(128), nn.LeakyReLU(), nn.Linear(128, 1))
        self.step0_attention = SelfAttentionLayer(d_model=128, nhead=8, normalize_before=True)
        self.step0_ffn = FFNLayer(d_model=128, dim_feedforward=128, normalize_before=True)
        self.step0_value_head = nn.Linear(in_features=128, out_features=1)
        self.step0_value_token = nn.Embedding(1, 128) ## feats and pos

    def forward(self, sampled_env_xyz, sampled_env_feats, history):
        ## the 1st is cur_sp, rest are nbr_sp, shape: [B, N, C]
        point_feats = self.point_features_head(sampled_env_feats)
        ###
        point_feats[:, 0, :]  += self.seed_indicator.weight[0]
        point_feats[:, 1:, :]  += self.seed_indicator.weight[1]
        value_token = self.value_token.weight[0][None, ...] # [1, 256]
        point_feats = torch.cat((value_token.repeat(point_feats.shape[0], 1, 1), point_feats), dim=1)
        ###
        mask = torch.zeros(point_feats.shape[1], point_feats.shape[1], dtype=torch.bool, device=point_feats.device)
        mask[1:, 0] = True  # [1:, 0] = True 表示屏蔽
        for i in range(len(self.attention)):
            output = self.attention[i](tgt = point_feats, tgt_mask=mask)### [bs, 1+5, 256]
            output = self.FFN[i](output)### [bs, 1+5, 256]
            point_feats = output

        return self.action_head(output[:, 1:, :]).squeeze(2), self.value_head(output[:, 0, :].squeeze(1)).squeeze(1)

    def step0(self, sampled_env_feats, history):
        point_feats = self.point_features_head(sampled_env_feats)
        value_token = self.step0_value_token.weight[None, ...]
        point_feats = torch.cat((value_token.repeat(point_feats.shape[0], 1, 1), point_feats), dim=1)# [bs, 1+100, C]
        mask = torch.zeros(point_feats.shape[1], point_feats.shape[1], dtype=torch.bool, device=point_feats.device)
        mask[1:, 0] = True  # [1:, 0] = True 表示屏蔽
        output = self.step0_attention(point_feats, tgt_mask=mask)
        action = self.step0_action_head(output[:, 1:, :]).squeeze(2)
        if torch.isinf(action).sum()>0:
            print('inf action')
        return action, self.step0_value_head(output[:, 0, :].squeeze(1)).squeeze(1)


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


class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0, normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

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

    def forward_post(self, tgt, memory, memory_mask=None, memory_key_padding_mask=None, pos=None, query_pos=None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos), key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt, memory, memory_mask=None, memory_key_padding_mask=None, pos=None, query_pos=None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos), key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt, memory, memory_mask=None, memory_key_padding_mask=None, pos=None, query_pos=None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask, memory_key_padding_mask, pos, query_pos)


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


