# coding: utf-8
# author: wcc
# create date: 2021-01-10 20:35
import logging

import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F


class MetaLearner(torch.nn.Module):
    def __init__(self, config):
        super(MetaLearner, self).__init__()
        self.embedding_dim = config['embedding_dim']
        self.fc1_in_dim = self.embedding_dim * 2
        self.fc2_in_dim = self.embedding_dim
        self.fc2_out_dim = 1
        # self.layer_norm = torch.nn.LayerNorm(dim1 + dim2)
        # self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
        # self.fc2 = torch.nn.Linear(dim3, dim4)
        # self.act = torch.nn.ReLU()

        # prediction parameters
        self.vars = torch.nn.ParameterDict()

        w1 = torch.nn.Parameter(torch.ones([self.fc2_in_dim, self.fc1_in_dim]))  # 172, 172*2
        torch.nn.init.xavier_normal_(w1)
        self.vars['ml_fc_w1'] = w1
        self.vars['ml_fc_b1'] = torch.nn.Parameter(torch.zeros(self.fc2_in_dim))

        w2 = torch.nn.Parameter(torch.ones([self.fc2_out_dim, self.fc2_in_dim]))
        torch.nn.init.xavier_normal_(w2)
        self.vars['ml_fc_w2'] = w2
        self.vars['ml_fc_b2'] = torch.nn.Parameter(torch.zeros(self.fc2_out_dim))

    def forward(self, x1, x2, vars_dict=None):
        # print('x1.shape', x1.shape)
        # print('x2.shape', x2.shape)
        if vars_dict is None:
            vars_dict = self.vars

        x = torch.cat([x1, x2], dim=1)
        # x = self.layer_norm(x)
        x = F.relu(F.linear(x,  vars_dict['ml_fc_w1'], vars_dict['ml_fc_b1']))
        return F.linear(x, vars_dict['ml_fc_w2'], vars_dict['ml_fc_b2'])

    def zero_grad(self, vars_dict=None):
        with torch.no_grad():
            if vars_dict is None:
                for p in self.vars.values():
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars_dict.values():
                    if p.grad is not None:
                        p.grad.zero_()

    def update_parameters(self):
        return self.vars


class TGAT_Encoder(torch.nn.Module):
    def __init__(self, config, ngh_finder, n_feat, e_feat):
        super(TGAT_Encoder, self).__init__()

        self.num_layers = config['num_layers']
        self.ngh_finder = ngh_finder
        self.attn_mode = config['attn_mode']
        self.use_time = config['use_time']
        self.agg_method = config['agg_method']
        self.num_layers = config['num_layers']
        self.n_head = config['n_head']
        self.drop_out = config['drop_out']

        self.logger = logging.getLogger(__name__)
        self.n_feat_th = torch.nn.Parameter(torch.from_numpy(n_feat.astype(np.float32)))
        self.e_feat_th = torch.nn.Parameter(torch.from_numpy(e_feat.astype(np.float32)))
        self.edge_raw_embed = torch.nn.Embedding.from_pretrained(self.e_feat_th, padding_idx=0, freeze=True)
        self.node_raw_embed = torch.nn.Embedding.from_pretrained(self.n_feat_th, padding_idx=0, freeze=True)

        self.feat_dim = self.n_feat_th.shape[1]

        self.n_feat_dim = self.feat_dim
        self.e_feat_dim = self.feat_dim
        self.time_dim = self.feat_dim
        self.model_dim = (self.n_feat_dim + self.e_feat_dim + self.time_dim)

        self.vars = torch.nn.ParameterDict()

        # aggregate module output map
        w1 = torch.nn.Parameter(torch.zeros(self.feat_dim, self.model_dim + self.feat_dim))
        torch.nn.init.xavier_normal_(w1)
        self.vars['agg_fc_w1'] = w1

        w2 = torch.nn.Parameter(torch.zeros(self.feat_dim, self.feat_dim))
        torch.nn.init.xavier_normal_(w2)
        self.vars['agg_fc_w2'] = w2

        if self.agg_method == 'attn':
            self.logger.info('Aggregation uses attention model')

            # multi_head att weight d_k = d_v = model_dim // n_head
            d_k = d_v = self.model_dim // self.n_head

            w_qs = torch.nn.Parameter(torch.zeros(self.n_head * d_k, self.model_dim))
            nn.init.normal_(w_qs, mean=0, std=np.sqrt(2.0 / (self.model_dim + d_k)))
            self.vars['w_qs'] = w_qs

            w_ks = torch.nn.Parameter(torch.zeros(self.n_head * d_k, self.model_dim))
            nn.init.normal_(w_ks, mean=0, std=np.sqrt(2.0 / (self.model_dim + d_k)))
            self.vars['w_ks'] = w_ks

            w_vs = torch.nn.Parameter(torch.zeros(self.n_head * d_v, self.model_dim))
            nn.init.normal_(w_vs, mean=0, std=np.sqrt(2.0 / (self.model_dim + d_v)))
            self.vars['w_vs'] = w_vs

            fc_w = torch.nn.Parameter(torch.zeros(self.model_dim, self.n_head * d_v))
            nn.init.xavier_normal_(fc_w)
            self.vars['fc_w'] = fc_w

            self.attn_model_list = torch.nn.ModuleList([AttnModel(self.feat_dim,
                                                                  self.feat_dim,
                                                                  self.feat_dim,
                                                                  attn_mode=self.attn_mode,
                                                                  n_head=self.n_head,
                                                                  drop_out=self.drop_out) for _ in range(self.num_layers)])

        #     if attn_mode = 'map':
        #         weight_map = torch.nn.Parameter(torch.zeros(1, 2 * d_k))
        #         nn.init.xavier_normal_(weight_map)
        #         self.vars['weight_map'] = weight_map
        #
        #     self.attn_model_list = torch.nn.ModuleList([AttnModel(self.feat_dim,
        #                                                           self.feat_dim,
        #                                                           self.feat_dim,
        #                                                           attn_mode=attn_mode,
        #                                                           n_head=n_head,
        #                                                           drop_out=drop_out) for _ in range(num_layers)])
        #
        # elif agg_method == 'mean':
        #     self.logger.info('Aggregation uses constant mean model')
        #     self.attn_model_list = torch.nn.ModuleList([MeanPool(self.feat_dim,
        #                                                          self.feat_dim) for _ in range(num_layers)])

        else:

            raise ValueError('invalid agg_method value, use attn or lstm')

        if self.use_time == 'time':
            self.logger.info('Using time encoding')
            self.time_encoder = TimeEncode(expand_dim=self.n_feat_th.shape[1])
        # elif use_time == 'empty':
        #     self.logger.info('Using empty encoding')
        #     self.time_encoder = EmptyEncode(expand_dim=self.n_feat_th.shape[1])
        else:
            raise ValueError('invalid time option!')

    def forward(self, src_idx_l, target_idx_l, cut_time_l, num_neighbors, vars_dict=None):

        if vars_dict is None:
            vars_dict = self.vars

        src_embed = self.tem_conv(src_idx_l, cut_time_l, self.num_layers, vars_dict, num_neighbors)
        target_embed = self.tem_conv(target_idx_l, cut_time_l, self.num_layers, vars_dict, num_neighbors)

        return src_embed, target_embed

    def tem_conv(self, src_idx_l, cut_time_l, curr_layers, vars_dict, num_neighbors):
        assert (curr_layers >= 0)
        # print('curr_layers ', curr_layers)

        device = self.n_feat_th.device
        # print('device ', device)

        batch_size = len(src_idx_l)

        src_node_batch_th = torch.from_numpy(src_idx_l).long().to(device)
        # print('src_node_batch_th ', src_node_batch_th)

        cut_time_l_th = torch.from_numpy(cut_time_l).float().to(device)
        # print('cut_time_l_th.shape ', cut_time_l_th.shape)

        cut_time_l_th = torch.unsqueeze(cut_time_l_th, dim=1)
        # print('cut_time_l_th.unsqueeze ', cut_time_l_th.shape)

        # query node always has the start time -> time span == 0
        src_node_t_embed = self.time_encoder(torch.zeros_like(cut_time_l_th))
        # print('src_node_t_embed ', src_node_t_embed)

        src_node_feat = self.node_raw_embed(src_node_batch_th)
        # print('src_node_feat ', src_node_feat)

        if curr_layers == 0:
            # print('curr_layers ', curr_layers)
            # print('return src_node_feat')
            return src_node_feat
        else:
            src_node_conv_feat = self.tem_conv(src_idx_l, cut_time_l, curr_layers=curr_layers - 1,
                                               vars_dict=vars_dict, num_neighbors=num_neighbors)
            # print('curr_layers ', curr_layers)
            # print(src_node_conv_feat.shape)
            src_ngh_node_batch, src_ngh_eidx_batch, src_ngh_t_batch \
                = self.ngh_finder.get_temporal_neighbor(src_idx_l, cut_time_l, num_neighbors=num_neighbors)

            src_ngh_node_batch_th = torch.from_numpy(src_ngh_node_batch).long().to(device)
            # print('src_ngh_node_batch_th ', src_ngh_node_batch_th)
            src_ngh_eidx_batch = torch.from_numpy(src_ngh_eidx_batch).long().to(device)
            # print('src_ngh_eidx_batch', src_ngh_eidx_batch)
            src_ngh_t_batch_delta = cut_time_l[:, np.newaxis] - src_ngh_t_batch
            # print('src_ngh_t_batch_delta', src_ngh_t_batch_delta)
            src_ngh_t_batch_th = torch.from_numpy(src_ngh_t_batch_delta).float().to(device)
            # print('src_ngh_t_batch_th', src_ngh_t_batch_th)

            # get previous layer's node features
            src_ngh_node_batch_flat = src_ngh_node_batch.flatten()  # reshape(batch_size, -1)
            # print('src_ngh_node_batch_flat ', src_ngh_node_batch_flat)
            src_ngh_t_batch_flat = src_ngh_t_batch.flatten()  # reshape(batch_size, -1)
            # print('src_ngh_t_batch_flat ', src_ngh_t_batch_flat)

            src_ngh_node_conv_feat = self.tem_conv(src_ngh_node_batch_flat,
                                                   src_ngh_t_batch_flat,
                                                   curr_layers=curr_layers - 1,
                                                   num_neighbors=num_neighbors, vars_dict=vars_dict)

            src_ngh_feat = src_ngh_node_conv_feat.view(batch_size, num_neighbors, 172)
            # print("src_ngh_feat.shape", src_ngh_feat.shape)

            # get edge time features and node features
            src_ngh_t_embed = self.time_encoder(src_ngh_t_batch_th)
            # print('true src_ngh_t_embed ', src_ngh_t_embed)
            src_ngn_edge_feat = self.edge_raw_embed(src_ngh_eidx_batch)

            # attention aggregation
            mask = src_ngh_node_batch_th == 0
            # print('mask', mask)
            attn_m = self.attn_model_list[curr_layers - 1]

            local, weight = attn_m(src_node_conv_feat,
                                   src_node_t_embed,
                                   src_ngh_feat,
                                   src_ngh_t_embed,
                                   src_ngn_edge_feat,
                                   mask,
                                   vars_dict)
            return local

    def zero_grad(self, vars_dict=None):
        with torch.no_grad():
            if vars_dict is None:
                for p in self.vars.values():
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars_dict.values():
                    if p.grad is not None:
                        p.grad.zero_()

    def update_parameters(self):
        return self.vars


class TimeEncode(torch.nn.Module):
    def __init__(self, expand_dim, factor=5):
        super(TimeEncode, self).__init__()
        # init_len = np.array([1e8**(i/(time_dim-1)) for i in range(time_dim)])

        time_dim = expand_dim
        self.factor = factor
        self.basis_freq = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim))).float())
        self.phase = torch.nn.Parameter(torch.zeros(time_dim).float())

        # self.dense = torch.nn.Linear(time_dim, expand_dim, bias=False)

        # torch.nn.init.xavier_normal_(self.dense.weight)

    def forward(self, ts):
        # ts: [N, L]
        batch_size = ts.size(0)
        seq_len = ts.size(1)

        ts = ts.view(batch_size, seq_len, 1)  # [N, L, 1]
        map_ts = ts * self.basis_freq.view(1, 1, -1)  # [N, L, time_dim]
        map_ts += self.phase.view(1, 1, -1)

        harmonic = torch.cos(map_ts)

        return harmonic  # self.dense(harmonic)


class AttnModel(torch.nn.Module):
    """Attention based temporal layers
    """

    def __init__(self, feat_dim, edge_dim, time_dim,
                 attn_mode='prod', n_head=2, drop_out=0.1):
        """
        args:
          feat_dim: dim for the node features
          edge_dim: dim for the temporal edge features
          time_dim: dim for the time encoding
          attn_mode: choose from 'prod' and 'map'
          n_head: number of heads in attention
          drop_out: probability of dropping a neural.
        """
        super(AttnModel, self).__init__()

        self.feat_dim = feat_dim
        self.time_dim = time_dim

        self.edge_in_dim = (feat_dim + edge_dim + time_dim)
        self.model_dim = self.edge_in_dim

        assert (self.model_dim % n_head == 0)
        self.logger = logging.getLogger(__name__)
        self.attn_mode = attn_mode

        if attn_mode == 'prod':
            self.multi_head_target = MultiHeadAttention(n_head,
                                                        d_model=self.model_dim,
                                                        d_k=self.model_dim // n_head,
                                                        d_v=self.model_dim // n_head,
                                                        dropout=drop_out)
            self.logger.info('Using scaled prod attention')

        # elif attn_mode == 'map':
        #     self.multi_head_target = MapBasedMultiHeadAttention(n_head,
        #                                                         d_model=self.model_dim,
        #                                                         d_k=self.model_dim // n_head,
        #                                                         d_v=self.model_dim // n_head,
        #                                                         dropout=drop_out)
        #     self.logger.info('Using map based attention')
        else:
            raise ValueError('attn_mode can only be prod or map')

    def forward(self, src, src_t, seq, seq_t, seq_e, mask, vars_dict):
        """"Attention based temporal attention forward pass
        args:
          src: float Tensor of shape [B, D]
          src_t: float Tensor of shape [B, Dt], Dt == D
          seq: float Tensor of shape [B, N, D]
          seq_t: float Tensor of shape [B, N, Dt]
          seq_e: float Tensor of shape [B, N, De], De == D
          mask: boolean Tensor of shape [B, N], where the true value indicate a null value in the sequence.

        returns:
          output, weight

          output: float Tensor of shape [B, D]
          weight: float Tensor of shape [B, N]
        """

        src_ext = torch.unsqueeze(src, dim=1)  # src [B, 1, D]
        src_e_ph = torch.zeros_like(src_ext)
        q = torch.cat([src_ext, src_e_ph, src_t], dim=2)  # [B, 1, D + De + Dt] -> [B, 1, D]
        k = torch.cat([seq, seq_e, seq_t], dim=2)  # [B, 1, D + De + Dt] -> [B, 1, D]

        mask = torch.unsqueeze(mask, dim=2)  # mask [B, N, 1]
        mask = mask.permute([0, 2, 1])  # mask [B, 1, N]

        # # target-attention
        output, attn = self.multi_head_target(q=q, k=k, v=k, vars_dict=vars_dict, mask=mask)  # output: [B, 1, D + Dt], attn: [B, 1, N]
        # print('src.shape', src.shape)
        output = output.squeeze(1)
        # print('output.shape', output.shape)
        # print('output', output)
        # print('output.squeeze().shape', output.shape)
        attn = attn.squeeze()

        # output = self.merger(output, src)
        x = torch.cat([output, src], dim=1)
        # x = self.layer_norm(x)
        x = F.relu(F.linear(x, vars_dict['agg_fc_w1']))
        output = F.linear(x, vars_dict['agg_fc_w2'])

        return output, attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        # replaced by vars_dict
        # self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        # self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        # self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        # nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        # nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        # nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5), attn_dropout=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        # self.fc = nn.Linear(n_head * d_v, d_model)

        # nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, vars_dict, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        # print('q.shape',  q.shape)
        # q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        # print('q.shape',  q.shape)
        # k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        # v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = F.linear(q, vars_dict['w_qs']).view(sz_b, len_q, n_head, d_k)
        k = F.linear(k, vars_dict['w_ks']).view(sz_b, len_k, n_head, d_k)
        v = F.linear(v, vars_dict['w_vs']).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)

        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, 516)  # b x lq x (n*dv)
        # print('output.shape', outpshape)

        output = self.dropout(F.linear(output, vars_dict['fc_w']))
        output = self.layer_norm(output + residual)
        # output = self.layer_norm(output)

        return output, attn


class ScaledDotProductAttention(torch.nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = torch.nn.Dropout(attn_dropout)
        self.softmax = torch.nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        # print('q.shape', q.shape)
        # print('k.shape', k.shape)
        attn = torch.bmm(q, k.transpose(1, 2))  # calculate attention
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -1e10)

        attn = self.softmax(attn)  # [n * b, l_q, l_k]
        attn = self.dropout(attn)  # [n * b, l_v, d]

        output = torch.bmm(attn, v)

        return output, attn