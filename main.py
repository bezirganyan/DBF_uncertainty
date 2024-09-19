import numpy as np
import torch
import torch.optim as optim
from labml_nn.normalization.batch_norm.cifar10 import model
from torch.utils.data import DataLoader, Subset

from data import CUB, CalTech, Scene, HandWritten, PIE
from loss_function import get_bm_loss, get_loss
from model import RCML
import sys

gettrace = getattr(sys, 'gettrace', None)

if gettrace is None:
    print('No sys.gettrace')
elif gettrace():
    old_repr = torch.Tensor.__repr__
    def tensor_info(tensor):
        return repr(tensor.shape)[6:] + ' ' + repr(tensor.dtype)[6:] + '@' + str(tensor.device) + '\n' + old_repr(
            tensor)
    torch.Tensor.__repr__ = tensor_info

np.set_printoptions(precision=4, suppress=True)


def do_hyperparameter_search(args, dataset, agg):
    print('Hyperparameter search')
    print('=' * 20)
    num_samples = len(dataset)
    num_classes = dataset.num_classes
    num_views = dataset.num_views
    dims = dataset.dims
    index = np.arange(num_samples)
    np.random.shuffle(index)
    train_index, test_index = index[:int(0.8 * num_samples)], index[int(0.8 * num_samples):]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # cv_folds = 5
    # lr_values = [0.0003, 0.001, 0.003, 0.01]
    # annealing_values = [1, 10, 30, 50]
    # gamma_values = [0.5, 0.7, 1]
    # weight_decays = [1e-5, 1e-4, 1e-3]

    cv_folds = 2
    lr_values = [0.0003]
    annealing_values = [1]
    gamma_values = [0.5, 0.7]
    weight_decays = [1e-5]
    parameter_groups = [(lr, annealing, gamma, weight_decay) for lr in lr_values for annealing in annealing_values for gamma in gamma_values for weight_decay in weight_decays]
    cv_results = {}
    for lr, annealing, gamma, weight_decay in parameter_groups:
        print(f'====> Trying parameters: lr: {lr} annealing: {annealing} gamma: {gamma} weight_decay: {weight_decay}')
        cv_results[(lr, annealing, gamma, weight_decay)] = []
        for fold in range(cv_folds):
            print(f'====> fold: {fold}')
            val_set = train_index[fold * len(train_index) // cv_folds: (fold + 1) * len(train_index) // cv_folds]
            train_set = np.setdiff1d(train_index, val_set)
            train_loader = DataLoader(Subset(dataset, train_set), batch_size=args.batch_size, shuffle=True)
            val_loader = DataLoader(Subset(dataset, val_set), batch_size=args.batch_size, shuffle=False)

            model, acc = train(agg, num_classes, num_views, train_loader, val_loader, args, device, dims, annealing, gamma, lr, weight_decay)
            cv_results[(lr, annealing, gamma, weight_decay)].append(acc)
    best_params = max(cv_results, key=lambda x: np.mean(cv_results[x]))
    best_params = dict(zip(['lr', 'annealing', 'gamma', 'weight_decay'], best_params))
    return best_params


def normal(args, dataset, agg, best_params):
    num_samples = len(dataset)
    num_classes = dataset.num_classes
    num_views = dataset.num_views
    dims = dataset.dims
    index = np.arange(num_samples)
    np.random.shuffle(index)
    train_index, test_index = index[:int(0.8 * num_samples)], index[int(0.8 * num_samples):]
    train_loader = DataLoader(Subset(dataset, train_index), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(Subset(dataset, test_index), batch_size=args.batch_size, shuffle=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, acc = train(agg, num_classes, num_views, train_loader, None, args, device, dims, **best_params)

    model.eval()
    model.to(device)
    num_correct, num_sample = 0, 0
    for X, Y, indexes in test_loader:
        for v in range(num_views):
            X[v] = X[v].to(device)
        Y = Y.to(device)
        with torch.no_grad():
            evidences, evidence_a = model(X)
            _, Y_pre = torch.max(evidence_a, dim=1)
            num_correct += (Y_pre == Y).sum().item()
            num_sample += Y.shape[0]
    print('====> acc: {:.4f}'.format(num_correct / num_sample))
    return model, num_correct / num_sample


def train(agg, num_classes, num_views, train_loader, val_loader, args, device, dims, annealing, gamma, lr, weight_decay=1e-5):
    model = RCML(num_views, dims, num_classes, agg)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.to(device)
    model.train()
    for epoch in range(1, args.epochs + 1):
        for X, Y, indexes in train_loader:
            for v in range(num_views):
                X[v] = X[v].to(device)
            Y = Y.to(device)
            evidences, evidence_a = model(X)
            loss = get_loss(evidences, evidence_a, Y, epoch, num_classes, annealing, gamma, device)
            # loss = get_bm_loss(evidences, evidence_a, Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    val_acc = 0
    if val_loader is None:
        return model, val_acc
    model.eval()
    num_correct, num_sample = 0, 0
    for X, Y, indexes in val_loader:
        for v in range(num_views):
            X[v] = X[v].to(device)
        Y = Y.to(device)
        with torch.no_grad():
            evidences, evidence_a = model(X)
            _, Y_pre = torch.max(evidence_a, dim=1)
            num_correct += (Y_pre == Y).sum().item()
            num_sample += Y.shape[0]
    val_acc = num_correct / num_sample
    print(f'====> validation acc: {val_acc:0.3f}')
    return model, val_acc


def conflict(args, dataset, agg, model):
    num_samples = len(dataset)
    num_classes = dataset.num_classes
    num_views = dataset.num_views
    dims = dataset.dims
    index = np.arange(num_samples)
    np.random.shuffle(index)
    train_index, test_index = index[:int(0.8 * num_samples)], index[int(0.8 * num_samples):]

    # create a test set with conflict instances
    dataset.postprocessing(test_index, addNoise=True, sigma=0.5, ratio_noise=0.1, addConflict=True, ratio_conflict=0.4)

    train_loader = DataLoader(Subset(dataset, train_index), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(Subset(dataset, test_index), batch_size=args.batch_size, shuffle=False)

    retrain_conflict = model is None
    model = model if model else RCML(num_views, dims, num_classes, agg)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gamma = 1

    if retrain_conflict:
        print('====> Retraining the model with conflict instances')
        model.to(device)
        model.train()
        for epoch in range(1, args.epochs + 1):
            # print(f'====> {epoch}')
            for X, Y, indexes in train_loader:
                for v in range(num_views):
                    X[v] = X[v].to(device)
                Y = Y.to(device)
                evidences, evidence_a = model(X)
                loss = get_loss(evidences, evidence_a, Y, epoch, num_classes, args.annealing_step, gamma, device)
                # loss = get_bm_loss(evidences, evidence_a, Y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    model.eval()
    model.to(device)
    num_correct, num_sample = 0, 0
    for X, Y, indexes in test_loader:
        for v in range(num_views):
            X[v] = X[v].to(device)
        Y = Y.to(device)
        with torch.no_grad():
            evidences, evidence_a = model(X)
            _, Y_pre = torch.max(evidence_a, dim=1)
            num_correct += (Y_pre == Y).sum().item()
            num_sample += Y.shape[0]
    print('====> Conflict acc: {:.4f}'.format(num_correct / num_sample))
    return num_correct / num_sample


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=200, metavar='N',
                        help='input batch size for training [default: 100]')
    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                        help='number of epochs to train [default: 500]')
    parser.add_argument('--annealing_step', type=int, default=50, metavar='N',
                        help='gradually increase the value of lambda from 0 to 1')
    parser.add_argument('--lr', type=float, default=0.003, metavar='LR',
                        help='learning rate')
    parser.add_argument('--agg', type=str, default='conf_agg')
    parser.add_argument('--runs', type=int, default=1)
    args = parser.parse_args()

    datasets = [HandWritten, PIE]
    # datasets = [CUB, CalTech, Scene, HandWritten, PIE]

    results = dict()
    runs = args.runs
    for dset in datasets:
        dataset = dset()
        print(f'====> {dataset.data_name}')
        acc_normal = []
        acc_conflict = []
        best_params = do_hyperparameter_search(args, dataset, args.agg)
        for i in range(runs):
            print(f'====> {dataset.data_name} {i:02d}')
            model, acc = normal(args, dset(), args.agg, best_params)
            acc_normal.append(acc)
            conflict_acc = conflict(args, dset(), args.agg, model)
            acc_conflict.append(conflict_acc)
        results[f'{dataset.data_name}_normal_mean'] = np.mean(acc_normal) * 100
        results[f'{dataset.data_name}_normal_std'] = np.std(acc_normal) * 100
        results[f'{dataset.data_name}_conflict_mean'] = np.mean(acc_conflict) * 100
        results[f'{dataset.data_name}_conflict_std'] = np.std(acc_conflict) * 100

    print('====> Results')
    with open(f'{args.agg}_{args.runs}_{args.epochs}_hs.txt', 'w+') as f:
        for key, value in results.items():
            if key.endswith('mean'):
                print(f'{key}: {value:0.3f}% ± {results[key.replace("_mean", "_std")]:0.3f}\n')
                f.write(f'{key}: {value:0.3f}% ± {results[key.replace("_mean", "_std")]:0.3f}\n')