import dgl
import csv
import torch
import numpy as np
import pandas as pd

from scipy import sparse
from sklearn import metrics
from scipy.linalg import expm
from collections import OrderedDict

import random

class MetricLogger(object):
    def __init__(self, attr_names, parse_formats, save_path):
        self._attr_format_dict = OrderedDict(zip(attr_names, parse_formats))
        self._file = open(save_path, 'w')
        self._csv = csv.writer(self._file)
        self._csv.writerow(attr_names)
        self._file.flush()

    def log(self, **kwargs):
        self._csv.writerow([parse_format % kwargs[attr_name]
                            for attr_name, parse_format in self._attr_format_dict.items()])
        self._file.flush()

    def close(self):
        self._file.close()



def load_Zdataset(path):
    # load heterogeneous data
    herb_herb = pd.read_csv(path + 'herb_herb.csv', index_col=0).values
    herb_herb = sparse.csr_matrix(herb_herb)
    herb_num = herb_herb.shape[0]

    herb_disease = pd.read_csv(path + 'herb_disease.csv', index_col=0).values
    herb_disease = sparse.csr_matrix(herb_disease)
    disease_herb = herb_disease.T

    herb_TCMsymptom = pd.read_csv(path + 'herb_TCMsymptom.csv', index_col=0).values
    herb_TCMsymptom = sparse.csr_matrix(herb_TCMsymptom)
    TCMsymptom_herb = herb_TCMsymptom.T

    herb_symptom = pd.read_csv(path + 'herb_symptom.csv', index_col=0).values
    herb_symptom = sparse.csr_matrix(herb_symptom)
    symptom_herb = herb_symptom.T

    # target graph
    target_drug = pd.read_csv(path + 'target_drug.csv', index_col=0).values
    target_drug = sparse.csr_matrix(target_drug)
    drug_target = target_drug.T
    target_num = target_drug.shape[0]

    target_disease = pd.read_csv(path + 'target_disease.csv', index_col=0).values
    target_disease = sparse.csr_matrix(target_disease)
    disese_target = target_disease.T

    target_symptom = pd.read_csv(path + 'target_symptom.csv', index_col=0).values
    target_symptom = sparse.csr_matrix(target_symptom)
    symptom_target = target_symptom.T

    # construct graph
    herb_graph = dgl.heterograph({
        ('herb', 'similarity', 'herb'): herb_herb.nonzero(),
        ('herb', 'hdi', 'diease'): herb_disease.nonzero(),
        ('diease', 'dih', 'herb'): disease_herb.nonzero(),
        ('herb', 'hTC', 'TCMsymptom'): herb_TCMsymptom.nonzero(),
        ('TCMsymptom', 'TCh', 'herb'): TCMsymptom_herb.nonzero(),
        ('herb','hsy','symptom'): herb_symptom.nonzero(),
        ('symptom','syh','herb'): symptom_herb.nonzero()
    })

    target_graph = dgl.heterograph({
        ('target', 'tdr', 'drug'): target_drug.nonzero(),
        ('drug', 'drt', 'target'): drug_target.nonzero(),
        ('target', 'tdi', 'disease'): target_disease.nonzero(),
        ('disease', 'dit', 'target'): disese_target.nonzero(),
        ('target', 'tsy', 'symptom'): target_symptom.nonzero(),
        ('symptom', 'syt', 'target'): symptom_target.nonzero()
    })

    graph = [herb_graph, target_graph]
    ground_truth = pd.read_csv(path + 'target_herb.csv', index_col=0).values.T

    neg_row, neg_col = np.where(ground_truth == 0)
    negative_index = [list(item) for item in zip(neg_row, neg_col)]

    pos_row, pos_col = np.where(ground_truth == 1)
    positive_index = [list(item) for item in zip(pos_row, pos_col)]
    
    # Find the class with fewer samples and determine the number of samples to keep
    num_samples = min(len(negative_index), len(positive_index))

    # Randomly sample the same number of negative samples
    selected_negative_index = random.sample(negative_index, num_samples)

    # Combine positive and negative samples to create the balanced dataset
    data_set = np.zeros((num_samples * 2, 3), dtype=int)

    count = 0
    for i in positive_index:
        data_set[count][0] = i[0]
        data_set[count][1] = i[1]
        data_set[count][2] = 1
        count += 1
    for j in selected_negative_index:
        data_set[count][0] = j[0]
        data_set[count][1] = j[1]
        data_set[count][2] = 0
        count += 1

    print('Zdataset loading finished')
    return data_set, graph, herb_num, target_num


def construct_Zdataset(path, include_herb_target=True):
    herb_herb = pd.read_csv(path + 'herb_herb.csv', index_col=0).values
    herb_herb = sparse.csr_matrix(herb_herb)
    herb_num = herb_herb.shape[0]

    herb_disease = pd.read_csv(path + 'herb_disease.csv', index_col=0).values
    herb_disease = sparse.csr_matrix(herb_disease)
    disease_herb = herb_disease.T

    herb_TCMsymptom = pd.read_csv(path + 'herb_TCMsymptom.csv', index_col=0).values
    herb_TCMsymptom = sparse.csr_matrix(herb_TCMsymptom)
    TCMsymptom_herb = herb_TCMsymptom.T

    herb_symptom = pd.read_csv(path + 'herb_symptom.csv', index_col=0).values
    herb_symptom = sparse.csr_matrix(herb_symptom)
    symptom_herb = herb_symptom.T

    # construct graph
    if include_herb_target:
        herb_target = pd.read_csv(path + 'target_herb.csv', index_col=0).values.T
        herb_target = sparse.csr_matrix(herb_target)
        target_herb = herb_target.T

        target_drug = pd.read_csv(path + 'target_drug.csv', index_col=0).values
        target_drug = sparse.csr_matrix(target_drug)
        drug_target = target_drug.T

        target_disease = pd.read_csv(path + 'target_disease.csv', index_col=0).values
        target_disease = sparse.csr_matrix(target_disease)
        disese_target = target_disease.T

        target_symptom = pd.read_csv(path + 'target_symptom.csv', index_col=0).values
        target_symptom = sparse.csr_matrix(target_symptom)
        symptom_target = target_symptom.T

        herb_graph = dgl.heterograph({
            ('herb', 'similarity', 'herb'): herb_herb.nonzero(),
            ('herb', 'hdi', 'disease'): herb_disease.nonzero(),
            ('disease', 'dih', 'herb'): disease_herb.nonzero(),
            ('herb', 'hTC', 'TCMsymptom'): herb_TCMsymptom.nonzero(),
            ('TCMsymptom', 'TCh', 'herb'): TCMsymptom_herb.nonzero(),
            ('herb', 'hsy', 'symptom'): herb_symptom.nonzero(),
            ('symptom', 'syh', 'herb'): symptom_herb.nonzero(),
            ('herb', 'htg', 'target'): herb_target.nonzero(),
            ('target', 'gth', 'herb'): target_herb.nonzero()
        })

        target_graph = dgl.heterograph({
            ('target', 'tdr', 'drug'): target_drug.nonzero(),
            ('drug', 'drt', 'target'): drug_target.nonzero(),
            ('target', 'tdi', 'disease'): target_disease.nonzero(),
            ('disease', 'dit', 'target'): disese_target.nonzero(),
            ('target', 'tsy', 'symptom'): target_symptom.nonzero(),
            ('symptom', 'syt', 'target'): symptom_target.nonzero(),
            ('target', 'gth', 'herb'): target_herb.nonzero(),
            ('herb', 'htg', 'target'): herb_target.nonzero()
        })

        graph = [herb_graph, target_graph]
    else:
        herb_graph = dgl.heterograph({
            ('herb', 'similarity', 'herb'): herb_herb.nonzero(),
            ('herb', 'hdi', 'disease'): herb_disease.nonzero(),
            ('disease', 'dih', 'herb'): disease_herb.nonzero(),
            ('herb', 'hTC', 'TCMsymptom'): herb_TCMsymptom.nonzero(),
            ('TCMsymptom', 'TCh', 'herb'): TCMsymptom_herb.nonzero(),
            ('herb', 'hsy', 'symptom'): herb_symptom.nonzero(),
            ('symptom', 'syh', 'herb'): symptom_herb.nonzero()
        })

        target_graph = dgl.heterograph({
            ('target', 'tdr', 'drug'): target_drug.nonzero(),
            ('drug', 'drt', 'target'): drug_target.nonzero(),
            ('target', 'tdi', 'disease'): target_disease.nonzero(),
            ('disease', 'dit', 'target'): disese_target.nonzero(),
            ('target', 'tsy', 'symptom'): target_symptom.nonzero(),
            ('symptom', 'syt', 'target'): symptom_target.nonzero()
        })

        graph = [herb_graph, target_graph]

    print('Zgraph construct finished')
    return graph


def load_data(args):
    if args.dataset == 'Zdataset':
        return load_Zdataset(args.path)
    else:
        print("None dataset")


def load_denovo_Zdataset(path):
    # load heterogeneous data
    herb_herb = pd.read_csv(path + 'herb_herb.csv', index_col=0).values
    herb_herb = sparse.csr_matrix(herb_herb)
    herb_num = herb_herb.shape[0]

    herb_disease = pd.read_csv(path + 'herb_disease.csv', index_col=0).values
    herb_disease = sparse.csr_matrix(herb_disease)
    disease_herb = herb_disease.T

    herb_TCMsymptom = pd.read_csv(path + 'herb_TCMsymptom.csv', index_col=0).values
    herb_TCMsymptom = sparse.csr_matrix(herb_TCMsymptom)
    TCMsymptom_herb = herb_TCMsymptom.T

    herb_symptom = pd.read_csv(path + 'herb_symptom.csv', index_col=0).values
    herb_symptom = sparse.csr_matrix(herb_symptom)
    symptom_herb = herb_symptom.T

    # target graph
    target_drug = pd.read_csv(path + 'target_drug.csv', index_col=0).values
    target_drug = sparse.csr_matrix(target_drug)
    drug_target = target_drug.T
    target_num = target_drug.shape[0]

    target_disease = pd.read_csv(path + 'target_disease.csv', index_col=0).values
    target_disease = sparse.csr_matrix(target_disease)
    disese_target = target_disease.T

    target_symptom = pd.read_csv(path + 'target_symptom.csv', index_col=0).values
    target_symptom = sparse.csr_matrix(target_symptom)
    symptom_target = target_symptom.T

    # construct graph
    herb_graph = dgl.heterograph({
        ('herb', 'similarity', 'herb'): herb_herb.nonzero(),
        ('herb', 'hdi', 'diease'): herb_disease.nonzero(),
        ('diease', 'dih', 'herb'): disease_herb.nonzero(),
        ('herb', 'hTC', 'TCMsymptom'): herb_TCMsymptom.nonzero(),
        ('TCMsymptom', 'TCh', 'herb'): TCMsymptom_herb.nonzero(),
        ('herb','hsy','symptom'): herb_symptom.nonzero(),
        ('symptom','syh','herb'): symptom_herb.nonzero()
    })

    target_graph = dgl.heterograph({
        ('target', 'tdr', 'drug'): target_drug.nonzero(),
        ('drug', 'drt', 'target'): drug_target.nonzero(),
        ('target', 'tdi', 'disease'): target_disease.nonzero(),
        ('disease', 'dit', 'target'): disese_target.nonzero(),
        ('target', 'tsy', 'symptom'): target_symptom.nonzero(),
        ('symptom', 'syt', 'target'): symptom_target.nonzero()
    })

    graph = [herb_graph, target_graph]
    ground_truth = pd.read_csv(path + 'target_herb.csv', index_col=0).values.T

    print('Zdataset loading finished')
    return ground_truth, graph, herb_num, target_num

def load_denovo(args):
    if args.dataset == 'Zdataset':
        return load_denovo_Zdataset(args.path)
    else:
        print("None dataset")


def evaluate(model, g, test_data, args):
    model.eval()
    with torch.no_grad():
        drug_embedding, disease_embedding, score_mat = model(g, args)
        score_mat = score_mat.cpu().numpy()

    test_data = test_data.cpu().numpy()
    test_score = score_mat[tuple(np.array(test_data[:, 0:2]).T)].tolist()
    test_true = list(test_data[:, 2])

    # Calculate AUROC and AUPR
    auroc = metrics.roc_auc_score(test_true, test_score)
    aupr = metrics.average_precision_score(test_true, test_score)

    # Convert scores to binary predictions using a threshold (default: 0.5)
    threshold = 0.5
    test_pred = [1 if score >= threshold else 0 for score in test_score]

    # Calculate accuracy, precision, recall, and F1-score
    accuracy = metrics.accuracy_score(test_true, test_pred)
    precision = metrics.precision_score(test_true, test_pred)
    recall = metrics.recall_score(test_true, test_pred)
    f1 = metrics.f1_score(test_true, test_pred)

    return auroc, aupr, accuracy, precision, recall, f1, test_score, test_true


def get_ppr_matrix(
        adj_matrix: np.ndarray,
        alpha: float = 0.1) -> np.ndarray:
    num_nodes = adj_matrix.shape[0]
    A_tilde = adj_matrix + np.eye(num_nodes)
    D_tilde = np.diag(1 / np.sqrt(A_tilde.sum(axis=1)))
    H = D_tilde @ A_tilde @ D_tilde
    return alpha * np.linalg.inv(np.eye(num_nodes) - (1 - alpha) * H)


def get_heat_matrix(
        adj_matrix: np.ndarray,
        t: float = 5.0) -> np.ndarray:
    num_nodes = adj_matrix.shape[0]
    A_tilde = adj_matrix + np.eye(num_nodes)
    D_tilde = np.diag(1 / np.sqrt(A_tilde.sum(axis=1)))
    H = D_tilde @ A_tilde @ D_tilde
    return expm(-t * (np.eye(num_nodes) - H))
