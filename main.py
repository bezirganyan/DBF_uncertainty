import sys

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from data import CUB, CalTech, HandWritten, PIE, Scene
from loss_function import get_loss
from model import RCML
torch.autograd.set_detect_anomaly(True)
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


def normal(args, dataset, agg, best_params):
    num_samples = len(dataset)
    num_classes = dataset.num_classes
    num_views = dataset.num_views
    dims = dataset.dims
    index = np.arange(num_samples)
    np.random.shuffle(index)
    train_index, test_index = index[:int(0.8 * num_samples)], index[int(0.8 * num_samples):]

    # dataset.postprocessing(train_index, addNoise=True, sigma=0.5, ratio_noise=1.0, addConflict=False, ratio_conflict=1.0)
    # dataset.postprocessing(test_index, addNoise=True, sigma=0.5, ratio_noise=1.0, addConflict=False, ratio_conflict=1.0)

    train_loader = DataLoader(Subset(dataset, train_index), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(Subset(dataset, test_index), batch_size=args.batch_size, shuffle=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, acc = train(agg, num_classes, num_views, train_loader, test_loader, args, device, dims, **best_params)

    model.eval()
    model.to(device)
    num_correct, num_sample = 0, 0
    uncertainty_values = []
    modality_uncertainties_values = []
    for X, Y, indexes in test_loader:
        for v in range(num_views):
            X[v] = X[v].to(device)
        Y = Y.to(device)
        with torch.no_grad():
            evidences, evidence_a = model(X)
            _, Y_pre = torch.max(evidence_a, dim=1)
            num_correct += (Y_pre == Y).sum().item()
            num_sample += Y.shape[0]
            uncertainties = num_classes / (evidence_a + 1).sum(dim=-1).unsqueeze(-1)
            uncertainty_values.append(uncertainties)
            modality_uncertainties = []
            for e in evidences:
                modality_uncertainties.append(num_classes / (evidences[e] + 1).sum(dim=-1).unsqueeze(-1))
        modality_uncertainties_values.append(torch.stack(modality_uncertainties, dim=1))
    uncertainty_values = torch.cat(uncertainty_values)
    print(f'====> {agg}_{dataset.data_name} Mean uncertainty: {uncertainty_values.mean().item()}')
    torch.save(uncertainty_values, f'exp_results/{agg}_{dataset.data_name}_uncertainty_values_normal_{args.activation}_{args.flambda}.pth')
    modality_uncertainties_values = torch.cat(modality_uncertainties_values)
    torch.save(modality_uncertainties_values, f'exp_results/{agg}_{dataset.data_name}_modality_uncertainty_values_normal_{args.activation}_{args.flambda}.pth')
    print('====> acc: {:.4f}'.format(num_correct / num_sample))
    return model, num_correct / num_sample


def train(agg, num_classes, num_views, train_loader, val_loader, args, device, dims, annealing, gamma, lr,
          weight_decay=1e-5):
    model = RCML(num_views, dims, num_classes, agg, flambda=args.flambda, activation=args.activation)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.to(device)
    model.train()
    max_val_acc = 0
    for epoch in range(1, args.epochs + 1):
        for X, Y, indexes in train_loader:
            for v in range(num_views):
                X[v] = X[v].to(device)
            Y = Y.to(device)
            evidences, evidence_a = model(X)
            assert not torch.isnan(evidence_a).any() and not torch.isinf(evidence_a).any()
            for e in evidences:
                assert not torch.isnan(evidences[e]).any() and not torch.isinf(evidences[e]).any()
            loss = get_loss(evidences, evidence_a, Y, epoch, num_classes, annealing, gamma, device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model, 0


def conflict(args, dataset, agg, model):
    num_samples = len(dataset)
    num_classes = dataset.num_classes
    num_views = dataset.num_views
    dims = dataset.dims
    index = np.arange(num_samples)
    np.random.shuffle(index)
    train_index, test_index = index[:int(0.8 * num_samples)], index[int(0.8 * num_samples):]

    # create a test set with conflict instances
    dataset.postprocessing(test_index, addNoise=False, sigma=0.5, ratio_noise=0.0, addConflict=True, ratio_conflict=1.0)

    train_loader = DataLoader(Subset(dataset, train_index), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(Subset(dataset, test_index), batch_size=args.batch_size, shuffle=False)

    retrain_conflict = model is None
    model = model if model else RCML(num_views, dims, num_classes, agg, flambda=args.flambda, activation=args.activation)
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
    uncertainty_values = []
    modality_uncertainties_values = []
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
            uncertainties = num_classes / (evidence_a + 1).sum(dim=-1).unsqueeze(-1)
            uncertainty_values.append(uncertainties)
            modality_uncertainties = []
            for e in evidences:
                modality_uncertainties.append(num_classes / (evidences[e] + 1).sum(dim=-1).unsqueeze(-1))
        modality_uncertainties_values.append(torch.stack(modality_uncertainties, dim=1))
    uncertainty_values = torch.cat(uncertainty_values)
    modality_uncertainties_values = torch.cat(modality_uncertainties_values)
    print(f'====> {agg}_{dataset.data_name} Mean uncertainty: {uncertainty_values.mean().item()}')
    torch.save(uncertainty_values, f'exp_results/{agg}_{dataset.data_name}_uncertainty_values_conflict_{args.activation}_{args.flambda}.pth')
    torch.save(modality_uncertainties_values, f'exp_results/{agg}_{dataset.data_name}_modality_uncertainty_values_conflict_{args.activation}_{args.flambda}.pth')
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
    parser.add_argument('--flambda', type=float, default=1)
    parser.add_argument('--activation', type=str, default='softplus')
    args = parser.parse_args()

    # datasets = [HandWritten, PIE]
    datasets = [Scene, CUB,  PIE, HandWritten, CalTech]

    best_params = {'CUB': {'lr': 0.003, 'annealing': 10, 'gamma': 1, 'weight_decay': 0.0001},
                   'HandWritten': {'lr': 0.003, 'annealing': 50, 'gamma': 0.7, 'weight_decay': 1e-5},
                   'PIE': {'lr': 0.003, 'annealing': 50, 'gamma': 0.5, 'weight_decay': 1e-5},
                   'CalTech': {'lr': 0.0003, 'annealing': 30, 'gamma': 0.5, 'weight_decay': 1e-4},
                   'Scene': {'lr': 0.01, 'annealing': 30, 'gamma': 0.5, 'weight_decay': 1e-5}}

    results = dict()
    runs = args.runs
    import pandas as pd
    columns = ['dataset', 'aggregation', 'activation', 'flambda',  'n_mean', 'n_std', 'c_mean', 'c_std']
    data = []
    for dset in datasets:
        dataset = dset()
        print(f'====> {dataset.data_name}')
        acc_normal = []
        acc_conflict = []

        for i in range(runs):
            print(f'====> {dataset.data_name} {i:02d}')
            model, acc = normal(args, dset(), args.agg, best_params[dataset.data_name])
            acc_normal.append(acc)
            conflict_acc = conflict(args, dset(), args.agg, model)
            acc_conflict.append(conflict_acc)
        print(f'====> {dataset.data_name} Normal: {np.mean(acc_normal) * 100:0.3f}% ± {np.std(acc_normal) * 100:0.3f}')
        print(f'====> {dataset.data_name} Conflict: {np.mean(acc_conflict) * 100:0.3f}% ± {np.std(acc_conflict) * 100:0.3f}')
        results[f'{dataset.data_name}_normal_mean'] = np.mean(acc_normal) * 100
        results[f'{dataset.data_name}_normal_std'] = np.std(acc_normal) * 100
        results[f'{dataset.data_name}_conflict_mean'] = np.mean(acc_conflict) * 100
        results[f'{dataset.data_name}_conflict_std'] = np.std(acc_conflict) * 100
        data.append([dataset.data_name, args.agg, args.activation, args.flambda, np.mean(acc_normal) * 100, np.std(acc_normal) * 100, np.mean(acc_conflict) * 100, np.std(acc_conflict) * 100])
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(f'exp_results/{args.agg}_{args.runs}_{args.epochs}_{args.activation}.csv')

    print('====> Results')
    with open(f'exp_results/{args.agg}_{args.runs}_{args.epochs}_{args.activation}.txt', 'w+') as f:
        for key, value in results.items():
            if key.endswith('mean'):
                print(f'{key}: {value:0.3f}% ± {results[key.replace("_mean", "_std")]:0.3f}')
                f.write(f'{key}: {value:0.3f}% ± {results[key.replace("_mean", "_std")]:0.3f}\n')
