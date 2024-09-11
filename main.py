import numpy as np
import torch
import torch.optim as optim
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


def normal(args, dataset, agg):
    num_samples = len(dataset)
    num_classes = dataset.num_classes
    num_views = dataset.num_views
    dims = dataset.dims
    index = np.arange(num_samples)
    np.random.shuffle(index)
    train_index, test_index = index[:int(0.8 * num_samples)], index[int(0.8 * num_samples):]
    train_loader = DataLoader(Subset(dataset, train_index), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(Subset(dataset, test_index), batch_size=args.batch_size, shuffle=False)

    model = RCML(num_views, dims, num_classes, agg)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gamma = 1

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
        lr_scheduler.step()

    model.eval()
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
    return num_correct / num_sample


def conflict(args, dataset, agg):
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

    model = RCML(num_views, dims, num_classes, agg)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gamma = 1

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
    return num_correct / num_sample


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=200, metavar='N',
                        help='input batch size for training [default: 100]')
    parser.add_argument('--epochs', type=int, default=500, metavar='N',
                        help='number of epochs to train [default: 500]')
    parser.add_argument('--annealing_step', type=int, default=50, metavar='N',
                        help='gradually increase the value of lambda from 0 to 1')
    parser.add_argument('--lr', type=float, default=0.003, metavar='LR',
                        help='learning rate')
    parser.add_argument('--agg', type=str, default='conf_agg')
    parser.add_argument('--runs', type=int, default=1)
    args = parser.parse_args()

    datasets = [CUB(), CalTech(), Scene(), HandWritten(), PIE()]

    results = dict()
    runs = args.runs
    for dataset in datasets:
        print(f'====> {dataset.data_name}')
        acc_normal = []
        for i in range(runs):
            print(f'====> {dataset.data_name} Normal {i:02d}')
            acc = normal(args, dataset, args.agg)
            acc_normal.append(acc)
        results[f'{dataset.data_name}_normal_mean'] = np.mean(acc_normal) * 100
        results[f'{dataset.data_name}_normal_std'] = np.std(acc_normal) * 100

        acc_conflict = []
        for i in range(runs):
            print(f'====> {dataset.data_name} Conflict {i:02d}')
            acc = conflict(args, dataset, args.agg)
            acc_conflict.append(acc)
        results[f'{dataset.data_name}_conflict_mean'] = np.mean(acc_conflict) * 100
        results[f'{dataset.data_name}_conflict_std'] = np.std(acc_conflict) * 100

    print('====> Results')
    for key, value in results.items():
        print(f'{key}: {value:0.3f}%')

    with open(f'{args.agg}_{args.runs}_{args.epochs}.txt', 'w+') as f:
        for key, value in results.items():
            f.write(f'{key}: {value:0.3f}%\n')