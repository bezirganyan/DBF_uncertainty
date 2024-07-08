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
        self.W_K = torch.nn.Linear(self.num_classes, self.num_classes, bias=False)
        self.W_Q = torch.nn.Linear(self.num_classes, self.num_classes, bias=False)
        # self.W_V = torch.nn.Linear(self.num_classes, self.num_classes, bias=False)
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

        evidences_c = torch.stack([evidences[i] for i in range(self.num_views)], dim=1)
        keys = self.W_K(evidences_c)
        if self.aggregation == 'weighted':
            q = evidences[0]
            queries = self.W_Q(q)

            attention = torch.matmul(queries.unsqueeze(1), keys.transpose(1, 2))
            attention = self.attention_dropout(attention)
            attention = torch.nn.functional.softmax(attention, dim=-1)
            evidence_a = torch.matmul(attention, evidences_c).squeeze(1)
        for i in range(1, self.num_views):
            if self.aggregation == 'weighted':
                # use attention mechanism to weight the views, where the query is the average of the views
                q = evidences[i]
                queries = self.W_Q(q)

                attention = torch.matmul(queries.unsqueeze(1), keys.transpose(1, 2))
                attention = self.attention_dropout(attention)
                attention = torch.nn.functional.softmax(attention, dim=-1)
                evidence_a += torch.matmul(attention, evidences_c).squeeze(1)
            elif self.aggregation == 'average':
                evidence_a = (evidences[i] + evidence_a)
            elif self.aggregation == 'conf_agg':
                evidence_a = (evidences[i] + evidence_a) / 2
            elif self.aggregation == 'conf_agg_random':
                evidence_a = (evidences[indices[i]] + evidence_a) / 2
            elif self.aggregation == 'prod':
                evidence_a = torch.mul(evidences[i], evidence_a) / (evidences[i] + evidence_a)
            elif self.aggregation == 'conf_agg_rev':
                evidence_a = (evidences[self.num_views-i] + evidence_a) / 2
            else:
                raise ValueError(f"Invalid aggregation method: {self.aggregation}")

        if self.aggregation == 'average' or self.aggregation == 'weighted':
            evidence_a = evidence_a / self.num_views
        return evidences, evidence_a


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
