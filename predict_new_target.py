import os
import torch
import argparse
import random
import time
import numpy as np
import pandas as pd
import torch.nn as nn

from utils import load_denovo, evaluate, MetricLogger, construct_Zdataset
from sklearn.model_selection import StratifiedKFold
from model import IMCHAN


def set_random_seed(seed=0):
    random.seed(seed)


class model_loss(nn.Module):
    def __init__(self):
        super(model_loss, self).__init__()

    def forward(self, score_mat, train_data, train_mat, args):
        alpha = args.alpha
        pos_idx = torch.tensor(train_data[0], dtype=torch.long).to(args.device).t().tolist()
        neg_idx = torch.tensor(train_data[1], dtype=torch.long).to(args.device).t().tolist()

        loss_fn = torch.nn.MSELoss(reduction='none')
        loss_mat = loss_fn(score_mat, train_mat)
        loss = (loss_mat[pos_idx].sum() * ((1-alpha) / 2) + loss_mat[neg_idx].sum() * (alpha / 2))
        # lamda_u =
        # lamda_v = 1
        # reg = lamda_u * (torch.trace(torch.mm(x_m.t(), x_m))) + lamda_v * (torch.trace(torch.mm(x_d.t(), x_d)))
        # loss = loss + reg
        return loss


def train(test_data, graph, train_data, train_mat, include_herb_target, args):
    if args.dataset == 'Zdataset':
        if not include_herb_target:
            model_meta_paths = [
                # For herb_graph
                [['similarity'], ['hdi', 'dih'], ['hTC', 'TCh'], ['hsy', 'syh']],
                # For target_graph
                [['tdr', 'drt'], ['tdi', 'dit'], ['tsy', 'syt']]
            ]
        else:
            model_meta_paths = [
                # For herb_graph
                [['similarity'], ['hdi', 'dih'], ['hTC', 'TCh'], ['hsy', 'syh'], ['htg', 'gth']],
                # For target_graph
                [['tdr', 'drt'], ['tdi', 'dit'], ['tsy', 'syt'], ['gth', 'htg']]
            ]

        model = IMCHAN(
            meta_paths=model_meta_paths,
            in_size=args.in_size,
            herb_fd=args.herb_num,
            target_fd=args.target_num,
            hidden_size=args.hidden_units,
            out_size=args.out_size,
            num_heads=args.num_heads,
            dropout=args.dropout,
            GAT_Layers=args.Gat_layers,
            W_size=args.W_size).to(args.device)

    loss_object = model_loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)

    test_data = torch.from_numpy(test_data).long()
    train_mat = torch.from_numpy(train_mat).float()
    test_data = test_data.to(args.device)
    train_mat = train_mat.to(args.device)

    best_auroc, best_aupr, best_score, best_true, best_epoch = 0, 0, 0, 0, 0
    # test_loss_logger = MetricLogger(['epoch', 'loss', 'auroc', 'aupr'], ['%d', '%.4f', '%.4f', '%.4f'],
    #                                 os.path.join(args.save_dir, 'test_metric%d.csv' % args.save_id))

    for epoch in range(args.epochs):
        model.train()
        d, p, score_mat = model(graph, args)
        loss = loss_object(score_mat, train_data, train_mat, args)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        auroc, aupr, score, true = evaluate(model, graph, test_data, args)
        logging_str = "Iter={}, loss={:.4f}, AUROC={:.4f}, AUPR={:.4f}".format(
            epoch, loss.item()/(len(train_data[0])), auroc, aupr)

        # test_loss_logger.log(epoch=epoch, loss=loss.item()/(len(train_data[0])), auroc=auroc, aupr=aupr)

        if aupr > best_aupr:
            best_auroc, best_aupr, best_score, best_true, best_epoch = auroc, aupr, score, true, epoch
            # torch.save(model.state_dict(), args.save_dir + 'NIMCHAN.pt')

        if epoch % args.train_interval == 0:
            print("test-logging_str", logging_str)

    result = {
        "y_score": best_score,
        "y_true": best_true
    }
    values = {
        "auroc": best_auroc,
        "aupr": best_aupr,
    }
    data_result = pd.DataFrame(result)
    values_result = pd.DataFrame(values, index=[0])
    data_result.to_csv(os.path.join(args.save_dir, '%d_result.csv' % int(args.save_id)), index=False)
    values_result.to_csv(os.path.join(args.save_dir, '%d_values.csv' % int(args.save_id)), index=False)
    # test_loss_logger.close()
    return best_auroc, best_aupr, best_epoch


def main(args):
    true_matrix, graph, herb_num, target_num = load_denovo(args)
    train_graph = construct_Zdataset(args.path, include_herb_target=True)
    args.herb_num = herb_num
    args.target_num = target_num

    # for times in range(args.repeat_times):
    # print('sample round', times + 1)
    # np.random.seed(args.seed)
    # rs = np.random.randint(0, 1000, 1)[0]
    # kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=rs)

    # args.save_dir = args.dataset + '_' + ''.join(str(times+1))
    args.save_dir = args.dataset

    args.save_dir = os.path.join("log", args.save_dir)
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
    auc_list, aupr_list = [], []

    for ith in range(herb_num):
        args.save_id = ith + 1
        train_matrix = true_matrix.copy()

        test_data = [[ith, cols] for cols in range(target_num)]
        for t in range(len(test_data)):
            test_data[t].append(int(train_matrix[ith, :][t]))
        test_data = np.array(test_data)
        train_matrix[ith, :] = 0
        start = time.perf_counter()
        print("#############%d th" % ith + "#############")

        pos_row, pos_col = np.nonzero(train_matrix)
        pos_data = [list(item) for item in zip(pos_row, pos_col)]
        neg_row, neg_col = np.nonzero(1 - train_matrix)
        neg_data = [list(item) for item in zip(neg_row, neg_col)]
        train_data = [pos_data, neg_data]

        auroc, aupr, epoch = train(test_data, train_graph, train_data, train_matrix, True,args)
        auc_list.append(auroc)
        aupr_list.append(aupr)
        print('best epoch{} | best auroc {:.4f} | best aupr {:.4f}'.format(epoch, auroc, aupr))
        end = time.perf_counter()
        print("running time", time.strftime("%H:%M:%S", time.gmtime(round(end - start))))

    print('mean auroc{:.4f} | mean aupr{:.4f}'.format(np.mean(auc_list), np.mean(aupr_list)))
    mean_auroc = np.mean(auc_list)
    mean_aupr = np.mean(aupr_list)
    print("Mean AUROC: %.4f, Mean AUPRC: %.4f" % (mean_auroc, mean_aupr))
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser('IMCHAN')
    # optimization setting
    parser.add_argument('--epochs', type=int, default=200, help='epoch number') # 64->100
    parser.add_argument('--lr', type=int, default=0.001, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout rate')
    parser.add_argument('--alpha', type=float, default=0.2, help='alpha rate')
    parser.add_argument('--weight_decay', type=float, default=0.00005, help='weight_decay rate0.001')
    # model setting
    parser.add_argument('--hidden_units', type=int, default=8, help='units number')
    parser.add_argument('--in_size', type=int, default=512, help='input size')
    parser.add_argument('--out_size', type=int, default=128, help='output size')
    parser.add_argument('--W_size', type=int, default=256, help='weight size')
    parser.add_argument('--Gat_layers', type=int, default=1, help='layer number')
    parser.add_argument('--num_heads', type=list, default=[8], help='attention head number')
    # general setting
    parser.add_argument('--dataset', '--data', type=str, default='Zdataset',
                        help='Sdataset, Gdataset, Cdataset, LRSSL, Ydataset')

    parser.add_argument('--path', type=str, default='dataset/Zdataset/',
                        help='Sdataset, Gdataset, Cdataset, LRSSL, Ydataset')

    parser.add_argument('--repeat_times', type=int, default=1, help='model repeat times')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--device', default='0', type=int,
                        help='Running device. E.g `--device 0`, if using cpu, set `--device -1`')
    parser.add_argument('--train_interval', type=int, default=1)
    parser.add_argument('--save_dir', type=str, help='The saving directory')
    parser.add_argument('--save_id', type=int, help='The saving log id')
    args = parser.parse_args()

    args.device = torch.device(args.device) if args.device >= 0 else torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    set_random_seed(args.seed)
    print(args)
    main(args)