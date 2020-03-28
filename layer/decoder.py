import torch
from torch import nn


class DedicomDecoder(torch.nn.Module):
    """
    Dedicom decoder
    Return drug_disease prediction metrics (m*n)
    """

    def __init__(self, input_dim):
        super(DedicomDecoder, self).__init__()
        self.global_interaction = nn.Parameter(nn.init.xavier_uniform_(torch.empty(input_dim, input_dim)))
        self.local_interaction_drug = nn.Parameter(nn.init.xavier_uniform_(torch.empty(input_dim, 1)).flatten())
        self.local_interaction_disease = nn.Parameter(nn.init.xavier_uniform_(torch.empty(input_dim, 1)).flatten())
        self.act = nn.Sigmoid()

    def forward(self, drug_feature, disease_feature):
        """
        :param drug_feature:
        :param disease_feature:
        :return: metric predictions
        """
        inputs_row = nn.Dropout2d(p=0.5)(drug_feature)
        inputs_col = nn.Dropout2d(p=0.5)(disease_feature)
        relation_drug = torch.diag(self.local_interaction_drug)
        relation_disease = torch.diag(self.local_interaction_disease)

        product1 = torch.mm(inputs_row, relation_drug)
        product2 = torch.mm(product1, self.global_interaction)
        product3 = torch.mm(product2, relation_disease)
        rec = torch.mm(product3, torch.transpose(inputs_col, 0, 1))
        outputs = self.act(rec)

        return outputs


class MlpDecoder(torch.nn.Module):
    """
    MLP decoder
    return drug-disease pair predictions
    """

    def __init__(self, input_dim):
        super(MlpDecoder, self).__init__()
        self.mlp_1 = nn.Sequential(nn.Dropout2d(p=0.2),
                                   nn.Linear(int(input_dim * 2), int(input_dim)),
                                   nn.ReLU())
        self.mlp_2 = nn.Sequential(nn.Dropout2d(p=0.0),
                                   nn.Linear(int(input_dim), int(input_dim // 2)),
                                   nn.ReLU())
        self.mlp_3 = nn.Sequential(nn.Dropout2d(p=0.0),
                                   nn.Linear(int(input_dim // 2), 1),
                                   nn.Sigmoid())

    def forward(self, drug_feature, disease_feature):
        """
        :param drug_feature:
        :param disease_feature:
        :return:
        """
        pair_feature = torch.cat([drug_feature, disease_feature], dim=1)
        embedding_1 = self.mlp_1(pair_feature)
        embedding_2 = self.mlp_2(embedding_1)
        outputs = self.mlp_3(embedding_2)
        return outputs
