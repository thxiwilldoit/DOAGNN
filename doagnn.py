import time

import torch
from torch import nn

class GraphSAGE(nn.Module):
    def __init__(self, W_sizes_ope, hidden_size_ope, out_size_ope, num_head, dropout, negative_slope=0.2,):
        super(GraphSAGE, self).__init__()

        self.in_sizes_ope = W_sizes_ope
        self.hidden_size_ope = hidden_size_ope
        self.out_size_ope = out_size_ope
        self.num_head = num_head
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        self.agg_z_j = nn.Sequential(
            nn.Linear(self.out_size_ope, self.hidden_size_ope),
            nn.ELU(),
            nn.Linear(self.hidden_size_ope, self.out_size_ope),
        )
        self.project = nn.Sequential(
            nn.Linear(self.out_size_ope, self.hidden_size_ope),
            nn.ELU(),
            nn.Linear(self.hidden_size_ope, self.out_size_ope),
        )
        self.agg_f_w = nn.Sequential(
            nn.Linear(self.out_size_ope, self.hidden_size_ope),
            nn.ELU(),
            nn.Linear(self.hidden_size_ope, self.out_size_ope),
        )

        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                nn.init.kaiming_normal_(m.weight.detach())
                m.bias.detach().zero_()

    def aggregate_dif(self, feat_wait: torch.Tensor, r: float, valid_mask: torch.Tensor):

        vm = valid_mask.unsqueeze(-1)
        feat_valid = feat_wait * vm
        sum_all = feat_valid.sum(dim=1, keepdim=True)
        K = valid_mask.sum(dim=1, keepdim=True).clamp(min=1).to(feat_wait.dtype)

        mixed = (feat_wait + (sum_all - feat_wait) * (feat_wait.new_tensor(r))) / K.unsqueeze(-1)
        mixed = self.agg_z_j(mixed)
        return mixed * vm + feat_wait * (~vm)

    def forward(self, ope_step_batch, end_ope_biases_batch, opes_appertain_batch, discount_r, batch_idxes, agg_func, features):

        x = features[0].clone()
        device = x.device
        dtype  = x.dtype

        batch_idxes = torch.as_tensor(batch_idxes, device=device, dtype=torch.long)
        x = x.index_select(dim=0, index=batch_idxes)

        s_all = torch.as_tensor(ope_step_batch, device=device, dtype=torch.long).index_select(0, batch_idxes)
        e_all = torch.as_tensor(end_ope_biases_batch, device=device, dtype=torch.long).index_select(0, batch_idxes)
        B, N, D = x.shape
        J = s_all.shape[1]

        t_idx = torch.arange(N, device=device).view(1, 1, N)
        s_bj  = s_all.unsqueeze(-1)
        e_bj  = e_all.unsqueeze(-1)
        valid_job = (s_all <= e_all)

        if (~valid_job.any(dim=1)).any():
            raise Exception("None")

        M = (t_idx > s_bj) & (t_idx <= e_bj) & valid_job.unsqueeze(-1)

        dist = (t_idx - s_bj).clamp_min(0)
        rpow = torch.pow(x.new_tensor(discount_r), dist.to(x.dtype))

        x_t  = x.unsqueeze(1)
        x_tm1 = torch.roll(x, shifts=1, dims=1).unsqueeze(1)

        agg_pre = self.agg_f_w(x_t).expand(-1, J, -1, -1)
        agg_sub = self.agg_f_w(x_tm1 * rpow.unsqueeze(-1))
        new_bjt = (agg_pre + agg_sub) * M.unsqueeze(-1)

        new_bt  = new_bjt.sum(dim=1)
        tgt_mask = M.any(dim=1)
        x = torch.where(tgt_mask.unsqueeze(-1), new_bt, x)

        idx_s = s_all.clamp(min=0, max=N-1).unsqueeze(-1).expand(B, J, D)
        feat_at_s = x.gather(dim=1, index=idx_s)

        feat_job_new = self.aggregate_dif(feat_at_s, discount_r, valid_job)

        job_slot = x[:, :J, :]
        job_slot = torch.where(valid_job.unsqueeze(-1), feat_job_new, job_slot)
        x = torch.cat([job_slot, x[:, J:, :]], dim=1)

        feat_opt = self.project(x)
        feat_opt = self.leaky_relu(feat_opt)
        return feat_opt


class RGCN(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_head,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 num_types=3,
                 num_bases=3):
        super(RGCN, self).__init__()
        self._num_heads = num_head
        self._in_opt_feats = in_feats[0]
        self._in_mas_feats = in_feats[1]
        self._out_feats = out_feats
        self.feat_drop = nn.Dropout(feat_drop)
        self.leaky_relu_1 = nn.LeakyReLU(negative_slope)
        self.hidden_dim = 128
        self.num_bases = num_bases
        self.num_types = num_types
        self.fc_w0 = nn.Sequential(
            nn.Linear(self._in_opt_feats, self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, self._out_feats * num_head),
        )
        self.v = nn.Parameter(torch.rand(size=(self.num_bases, self._in_mas_feats, self._out_feats)))
        self.coeff = nn.Parameter(torch.rand(size=(num_types, self.num_bases)))
        self.fc_edge = nn.Sequential(
            nn.Linear(1, self.hidden_dim),
            nn.Linear(self.hidden_dim, 1),
        )
        self.fc_ope_ma = nn.Sequential(
            nn.Linear(num_types*self._out_feats, self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, 1),
        )

        self.activation = activation

        self._num_heads = num_head
        self._out_feats = out_feats
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                nn.init.kaiming_normal_(m.weight.detach())
                m.bias.detach().zero_()

    def forward(self, ope_ma_adj_batch, batch_idxes, feat):
        device = feat[0].device
        idx = torch.as_tensor(batch_idxes, device=device, dtype=torch.long)

        h_opt = self.feat_drop(feat[0].index_select(0, idx))
        h_mas = self.feat_drop(feat[1].index_select(0, idx))
        e_raw = feat[2].index_select(0, idx)
        adj = ope_ma_adj_batch.index_select(0, idx)

        h_edg = self.fc_edge(e_raw.unsqueeze(-1)).squeeze(-1)
        feat_opt = self.fc_w0(h_opt)
        feat_opt_re = feat_opt.clone()

        v_b = self.v.view(self.num_bases, self._in_mas_feats * self._out_feats)
        w_r = (self.coeff @ v_b).view(self.num_types, self._in_mas_feats, self._out_feats)

        mask = (adj == 1).to(h_edg.dtype)
        weights = h_edg * mask

        agg_in = (weights.unsqueeze(-1) * h_mas.unsqueeze(1)).sum(dim=2)

        typed = torch.einsum('bof,tfe->bote', agg_in, w_r)

        typed_flat = typed.reshape(typed.shape[0] * typed.shape[1], -1)
        mas_w = self.fc_ope_ma(typed_flat)
        mas_w = mas_w.view(typed.shape[0], typed.shape[1], 1)

        feat_opt_re = feat_opt_re + mas_w
        feat_opt_ma = self.leaky_relu_1(feat_opt_re)
        return feat_opt_ma

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc_w0.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_edge.weight, gain=gain)
        nn.init.xavier_normal_(self.v, gain=gain)
        nn.init.xavier_normal_(self.coeff, gain=gain)


if __name__ == "__main__":
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    start = time.time()
    t = RGCN(in_feats=[6, 3, 4], out_feats=12, num_head=1).to(device)
    f = GraphSAGE([12, 12, 12, 12], 128, 12, 1, 0).to(device)
    end = time.time()
    print(end - start)