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
        self.agg_z_j = nn.Sequential(  # Learning the node features among different jobs
            nn.Linear(self.out_size_ope, self.hidden_size_ope),
            nn.ELU(),
            nn.Linear(self.hidden_size_ope, self.out_size_ope),
        )
        self.project = nn.Sequential(
            nn.Linear(self.out_size_ope, self.hidden_size_ope),
            nn.ELU(),
            nn.Linear(self.hidden_size_ope, self.out_size_ope),
        )
        self.agg_f_w = nn.Sequential(  # Learning the node features among same jobs
            nn.Linear(self.out_size_ope, self.hidden_size_ope),
            nn.ELU(),
            nn.Linear(self.hidden_size_ope, self.out_size_ope),
        )
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                nn.init.kaiming_normal_(m.weight.detach())
                m.bias.detach().zero_()

    def aggregate(self, feat_pre, feat_sub, r, agg_func='MEAN'):
        if agg_func == 'MEAN':
            feat_ag = torch.cat((feat_pre.unsqueeze(0).clone(), (feat_sub * r).unsqueeze(0).clone()), dim=0)
            feat_ag = self.agg_f_w(feat_ag)
            feat_ag = torch.sum(feat_ag, dim=0)
            return feat_ag
        elif agg_func == 'MAX':
            return None
        else:
            return None


    def aggregate_dif(self, feat_wait, r):
        feat_age = feat_wait.clone()
        for i_j in range(len(feat_wait)):
            for i_j_2 in range(len(feat_wait)):
                if i_j != i_j_2:
                    feat_wait[i_j] += r*feat_age[i_j_2]
            feat_wait[i_j] /= len(feat_wait)
            feat_wait[i_j] = self.agg_z_j(feat_wait[i_j].clone())
        feat_age = feat_wait.clone()
        return feat_age


    def forward(self, ope_step_batch, end_ope_biases_batch, opes_appertain_batch, discount_r, batch_idxes, agg_func, features):
        feat_opt = features[0].clone()
        for i_batch in range(len(batch_idxes.detach().tolist())):
            first_deal_index = 0
            no_deal = True
            for i_sum_job in range(len(ope_step_batch[batch_idxes[i_batch]])):
                if ope_step_batch[batch_idxes[i_batch]][i_sum_job] <= end_ope_biases_batch[batch_idxes[i_batch]][i_sum_job]:
                    no_deal = False
                    first_deal_index = i_sum_job
                    break
            if no_deal:
                raise Exception("None")
            features_all = torch.unsqueeze(feat_opt[i_batch][ope_step_batch[batch_idxes[i_batch]][first_deal_index]], dim=0)
            save_index = [first_deal_index]
            for i_sum_job in range(len(ope_step_batch[batch_idxes[i_batch]])):
                count = 0
                for i_num_job in range(end_ope_biases_batch[batch_idxes[i_batch]][i_sum_job], ope_step_batch[batch_idxes[i_batch]][i_sum_job], -1):
                    count += 1
                    discount_r_now = pow(discount_r, i_num_job - ope_step_batch[batch_idxes[i_batch]][i_sum_job])
                    feat_opt[i_batch][i_num_job] = self.aggregate(feat_opt[i_batch][i_num_job],
                                                                  feat_opt[i_batch][i_num_job - 1], discount_r_now, agg_func)
                if i_sum_job != first_deal_index:
                    if ope_step_batch[batch_idxes[i_batch]][i_sum_job] <= end_ope_biases_batch[batch_idxes[i_batch]][i_sum_job]:
                        save_index.append(i_sum_job)
                        features_all = torch.cat((features_all, torch.unsqueeze(feat_opt[i_batch][ope_step_batch[batch_idxes[i_batch]][i_sum_job]], dim=0)), dim=0)
            features_all = self.aggregate_dif(features_all, discount_r)
            for i_sum_job in range(len(save_index)):
                feat_opt[i_batch][save_index[i_sum_job]] = features_all[i_sum_job].clone()
        feat_opt = self.project(feat_opt)
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
        h_opt = self.feat_drop(feat[0])
        h_mas = self.feat_drop(feat[1])
        h_edg = self.fc_edge(feat[2].unsqueeze(-1)).squeeze(-1)
        feat_opt = self.fc_w0(h_opt)
        feat_opt_re = torch.clone(feat_opt)

        v_b = self.v.view(self.num_bases, self._in_mas_feats*self._out_feats)
        w_r = (self.coeff @ v_b).view(self.num_types, self._in_mas_feats, self._out_feats)

        for i_batch in range(len(batch_idxes.detach().tolist())):

            for i_opt_num in range(len(feat_opt[0])):
                # try:
                #     feat_edj_t = h_edg[i_batch][i_opt_num]
                # except IndexError:
                #     print("wrong")
                feat_edj_t = h_edg[i_batch][i_opt_num]

                mask = (ope_ma_adj_batch[i_batch][i_opt_num] == 1).unsqueeze(1)
                feat_mas_lip = h_mas[i_batch] * mask
                feat_mas_w = feat_edj_t @ feat_mas_lip @ w_r
                mas_w = feat_mas_w.clone().view(-1)
                mas_w = self.fc_ope_ma(mas_w)
                feat_opt_re[i_batch][i_opt_num] += mas_w
        feat_opt_ma = self.leaky_relu_1(feat_opt_re)
        return feat_opt_ma

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc_w0.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_edge.weight, gain=gain)
        nn.init.xavier_normal_(self.v, gain=gain)
        nn.init.xavier_normal_(self.coeff, gain=gain)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t = RGCN(in_feats=[6, 3, 4], out_feats=12, num_head=1).to(device)
    print(t)
    f = GraphSAGE([12, 12, 12, 12], 128, 12, 1, 0).to(device)
    print(f)