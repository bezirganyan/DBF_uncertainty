import numpy as np
import torch
import torch.nn as nn


class RCML(nn.Module):
    def __init__(self, num_views, dims, num_classes, aggregation='average'):
        super(RCML, self).__init__()
        self.aggregation = aggregation
        self.num_views = num_views
        self.num_classes = num_classes
        self.EvidenceCollectors = nn.ModuleList([EvidenceCollector(dims[i], self.num_classes) for i in range(self.num_views)])
        # self.W_K = torch.nn.Linear(self.num_classes, self.num_classes, bias=False)
        # self.W_Q = torch.nn.Linear(self.num_classes, self.num_classes, bias=False)
        # # self.W_V = torch.nn.Linear(self.num_classes, self.num_classes, bias=False)
        self.multihead_attention = torch.nn.MultiheadAttention(embed_dim=self.num_classes, num_heads=1, batch_first=True)
        self.attention_dropout = torch.nn.Dropout(0.2)

    def forward(self, X):
        # get evidence
        evidences = dict()
        for v in range(self.num_views):
            evidences[v] = self.EvidenceCollectors[v](X[v])
        # average belief fusion
        indices = list(range(self.num_views))
        np.random.shuffle(indices)
        if self.aggregation == 'conf_agg_random':
            evidence_a = evidences[indices[0]]
        else:
            evidence_a = evidences[0]
        if self.aggregation == 'weighted_belief':
            div_a, evidence_a = self.get_weighted_belief_fusion(self.num_views,0, evidences, 0)
        if self.aggregation == 'doc':
            return self.get_doc_belief_fusion(self.num_views, evidences)
        elif self.aggregation == 'tmc':
            return self.DS_Combin(evidences)

        for i in range(1, self.num_views):
            if self.aggregation == 'average':
                evidence_a = (evidences[i] + evidence_a)
            elif self.aggregation == 'conf_agg':
                evidence_a = (evidences[i] + evidence_a) / 2
            elif self.aggregation == 'conf_agg_random':
                evidence_a = (evidences[indices[i]] + evidence_a) / 2
            elif self.aggregation == 'sum':
                evidence_a = evidence_a + evidences[i]
            elif self.aggregation == 'conf_agg_rev':
                evidence_a = (evidences[self.num_views-i] + evidence_a) / 2
            elif self.aggregation == 'weighted_belief':
                div_a, evidence_a = self.get_weighted_belief_fusion(div_a, evidence_a, evidences, i)
            else:
                raise ValueError(f"Invalid aggregation method: {self.aggregation}")

        if self.aggregation == 'average':
            evidence_a = evidence_a / self.num_views
        elif self.aggregation == 'weighted_belief':
            evidence_a = evidence_a / div_a
        return evidences, evidence_a

    def get_weighted_belief_fusion(self, div_a, evidence_a, evidences, i):
        uncertainty = self.num_classes / (evidences[i] + 1).sum(dim=-1).unsqueeze(-1)
        alphas = evidences[i] + 1
        alpha_0 = alphas.sum(dim=-1, keepdim=True)
        probs = alphas / alpha_0
        # aleatoric = -torch.sum(probs * (torch.digamma(alphas + 1) - torch.digamma(alpha_0 + 1)), dim=-1).unsqueeze(-1)
        aleatoric = -torch.sum(probs * torch.log(probs), dim=-1).unsqueeze(-1)
        max_entropy = np.log(self.num_classes)
        aleatoric_to_1 = aleatoric / max_entropy
        evidence_a = evidence_a + evidences[i] * (1 - uncertainty) * (1 - aleatoric_to_1)
        # evidence_a = evidence_a + evidences[i]  * (1 - aleatoric_to_1)
        div_a = div_a - aleatoric_to_1
        return div_a, evidence_a


    def get_doc_belief_fusion(self, div_a, evidences):
        evidence_tensor = torch.stack(list(evidences.values()), dim=1)
        belief_tensor = evidence_tensor / (evidence_tensor + 1).sum(dim=-1).unsqueeze(-1)
        prob_tensor = (evidence_tensor + 1) / (evidence_tensor + 1).sum(dim=-1).unsqueeze(-1)
        uncertainty = self.num_classes / (evidence_tensor + 1).sum(dim=-1).unsqueeze(-1)
        discount = torch.ones(belief_tensor.shape[:-1]).to(belief_tensor.device)
        for i in range(self.num_views):
            cp = torch.abs(prob_tensor[:, i].unsqueeze(1) - prob_tensor).sum(-1) / 2
            cc = ((1 - uncertainty[:, i].unsqueeze(1)) * (1 - uncertainty)).squeeze(1)
            dc = cp * cc.squeeze(-1)
            agreement = torch.prod(1 - dc, dim=1)
            discount[:, i] *= agreement
        discount = discount.unsqueeze(-1)
        belief_tensor = belief_tensor * discount
        uncertainty =  uncertainty * discount + 1 - discount
        assert torch.allclose(belief_tensor.sum(dim=-1) + uncertainty.squeeze(-1), torch.ones_like(belief_tensor.sum(dim=-1)))
        evidences_a = self.num_classes * belief_tensor / uncertainty
        combined_evidence = evidences_a.mean(dim=1)

        return evidences, combined_evidence

    def DS_Combin(self, evidences):
        """
        :param alpha: All Dirichlet distribution parameters.
        :return: Combined Dirichlet distribution parameters.
        """
        alphas = dict()
        for v in range(self.num_views):
            alphas[v] = evidences[v] + 1
        def DS_Combin_two(alpha1, alpha2):
            """
            :param alpha1: Dirichlet distribution parameters of view 1
            :param alpha2: Dirichlet distribution parameters of view 2
            :return: Combined Dirichlet distribution parameters
            """
            alpha = dict()
            alpha[0], alpha[1] = alpha1, alpha2
            b, S, E, u = dict(), dict(), dict(), dict()
            for v in range(2):
                S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
                E[v] = alpha[v]-1
                b[v] = E[v]/(S[v].expand(E[v].shape))
                u[v] = self.num_classes/S[v]

            # b^0 @ b^(0+1)
            bb = torch.bmm(b[0].view(-1, self.num_classes, 1), b[1].view(-1, 1, self.num_classes))
            # b^0 * u^1
            uv1_expand = u[1].expand(b[0].shape)
            bu = torch.mul(b[0], uv1_expand)
            # b^1 * u^0
            uv_expand = u[0].expand(b[0].shape)
            ub = torch.mul(b[1], uv_expand)
            # calculate C
            bb_sum = torch.sum(bb, dim=(1, 2), out=None)
            bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
            C = bb_sum - bb_diag

            # calculate b^a
            b_a = (torch.mul(b[0], b[1]) + bu + ub)/((1-C).view(-1, 1).expand(b[0].shape))
            # calculate u^a
            u_a = torch.mul(u[0], u[1])/((1-C).view(-1, 1).expand(u[0].shape))

            # calculate new S
            S_a = self.num_classes / u_a
            # calculate new e_k
            e_a = torch.mul(b_a, S_a.expand(b_a.shape))
            alpha_a = e_a + 1
            return alpha_a

        for v in range(len(alphas)-1):
            if v==0:
                alpha_a = DS_Combin_two(alphas[0], alphas[1])
            else:
                alpha_a = DS_Combin_two(alpha_a, alphas[v+1])
        evidences_a = alpha_a - 1
        return evidences, evidences_a

class EvidenceCollector(nn.Module):
    def __init__(self, dims, num_classes):
        super(EvidenceCollector, self).__init__()
        self.num_layers = len(dims)
        self.net = nn.ModuleList()
        for i in range(self.num_layers - 1):
            self.net.append(nn.Linear(dims[i], dims[i + 1]))
            self.net.append(nn.ReLU())
            self.net.append(nn.Dropout(0.1))
        self.net.append(nn.Linear(dims[self.num_layers - 1], num_classes))
        self.net.append(nn.Softplus())

    def forward(self, x):
        h = self.net[0](x)
        for i in range(1, len(self.net)):
            h = self.net[i](h)
        return h
