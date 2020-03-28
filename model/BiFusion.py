import sys
import torch
import numpy as np

sys.path.append('../../BiFusion')
from layer.decoder import *
from layer.encoder import *
from dataloader.data_loader import BiFusionDataset
from utils.evaluation_metrics import auroc, auprc

hidden_dim_1 = 256
hidden_dim_2 = 128

batch_num = 512

global_protein_num = 13460
global_drug_num = 1012
global_disease_num = 592

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BiFusionNet(torch.nn.Module):
    def __init__(self, protein_num, drug_num, disease_num, protein_feature_num, drug_feature_num, disease_feature_num):
        super(BiFusionNet, self).__init__()
        self.encoder_1 = BiFusionLayer(protein_num, drug_num, disease_num, protein_feature_num, drug_feature_num,
                                       disease_feature_num, hidden_dim_1)
        self.encoder_2 = BiFusionLayer(protein_num, drug_num, disease_num, hidden_dim_1, hidden_dim_1, hidden_dim_1,
                                       hidden_dim_2)
        self.decoder = MlpDecoder(hidden_dim_2)

    def forward(self, ppi, drug_protein, disease_protein, drug_feature, disease_feature, protein_feature, pair):
        drug_feature, disease_feature, protein_feature = self.encoder_1(ppi, drug_protein, disease_protein,
                                                                        drug_feature, disease_feature, protein_feature)

        drug_feature, disease_feature, protein_feature = self.encoder_2(ppi, drug_protein, disease_protein,
                                                                        drug_feature, disease_feature, protein_feature)

        row, col = pair
        pred = self.decoder(drug_feature[row, :], disease_feature[col, :]).flatten()
        return pred


def model_train(epoch, batch, hyper_net, ppi, drug_protein, disease_protein, drug_feature,
                disease_feature, protein_feature, pair, gt):
    hyper_net.train()
    optimizer.zero_grad()
    logging_info = {}
    logging_info.update({'Train Epoch': epoch, 'batch': batch})
    prob = hyper_net(ppi, drug_protein, disease_protein, drug_feature, disease_feature, protein_feature, pair)
    gt = gt.float().to(device)

    weight = class_weight[gt.long()].to(device)
    loss_func = torch.nn.BCELoss(weight=weight, reduction='mean').to(device)

    loss = loss_func(prob, gt)
    loss.backward()
    optimizer.step()

    logging_info.update({'loss': '%.04f' % loss.data.item()})
    logging_info.update({'auroc': '%.04f' % auroc(prob, gt)})
    logging_info.update({'auprc': '%.04f' % auprc(prob, gt)})
    return logging_info


def model_test(epoch, batch, model, ppi, drug_protein, disease_protein, drug_feature, disease_feature,
               protein_feature, pair, gt):
    model.eval()
    logging_info = {}
    logging_info.update({'Test Epoch': epoch, 'batch': batch})
    with torch.no_grad():
        prob = model(ppi, drug_protein, disease_protein, drug_feature, disease_feature, protein_feature, pair)
        gt = gt.float().to(device)

        weight = class_weight[gt.long()].to(device)
        loss_func = torch.nn.BCELoss(weight=weight, reduction='mean').to(device)
        loss = loss_func(prob, gt)

        logging_info.update({'loss': '%.04f' % loss.data.item()})
        logging_info.update({'auroc': '%.04f' % auroc(prob, gt)})
        logging_info.update({'auprc': '%.04f' % auprc(prob, gt)})
    return logging_info


if __name__ == '__main__':

    class_weight = torch.Tensor([1, 1])
    database = BiFusionDataset()

    # ppi
    ppi = database.protein_protein
    # drug_protein interactions
    drug_protein = database.drug_protein
    # disease_protein interactions
    disease_protein = database.disease_protein
    # drug_disease interactions
    drug_disease = database.drug_disease
    pair, label = drug_disease.edge_index, drug_disease.edge_label

    train_index, test_index = drug_disease.train_index, drug_disease.test_index
    pair_train, pair_test = pair[:, train_index], pair[:, test_index]
    label_train, label_test = label[train_index], label[test_index]

    # we only use drug/disease in training set to construct similarity features
    selected_drug = np.unique(pair_train[0])
    selected_disease = np.unique(pair_train[1])

    # define features of drug and disease
    drug_feature = np.load('../data/drug_drug_similarity.npy')[:, selected_drug]
    drug_feature = torch.from_numpy(drug_feature).float().to(device)

    disease_feature = np.load('../data/disease_disease_similarity.npy')[:, selected_disease]
    disease_feature = torch.from_numpy(disease_feature).float().to(device)

    protein_feature = torch.zeros(global_protein_num, hidden_dim_1).float().to(device)

    # load data and model to GPU
    model = BiFusionNet(global_protein_num, global_drug_num, global_disease_num,
                        protein_feature_num=hidden_dim_1, drug_feature_num=len(selected_drug),
                        disease_feature_num=len(selected_disease)).to(device)

    ppi, drug_protein, disease_protein = ppi.to(device), drug_protein.to(device), disease_protein.to(device)

    # optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200], gamma=0.5)

    for epoch in range(500):
        # shuffle batches
        for batch, idx in enumerate(torch.split(torch.randperm(len(train_index)), batch_num)):
            train_logging = model_train(epoch, batch, model, ppi, drug_protein, disease_protein,
                                        drug_feature, disease_feature, protein_feature,
                                        pair=pair_train[:, idx],
                                        gt=label_train[idx])
            print(train_logging)

        for batch, idx in enumerate(torch.split(torch.randperm(len(test_index)), batch_num)):
            test_logging = model_test(epoch, batch, model, ppi, drug_protein, disease_protein,
                                      drug_feature, disease_feature, protein_feature,
                                      pair=pair_test[:, idx],
                                      gt=label_test[idx])
            print(test_logging)

        scheduler.step()
