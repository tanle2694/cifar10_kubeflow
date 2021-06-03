import torch


def accuracy(outputs, targets):
    with torch.no_grad():
        _, pred = torch.max(outputs, dim=1)
        assert pred.shape[0] == len(targets)
        correct = 0
        correct += torch.sum(pred == targets).item()
    return correct * 1.0 / len(targets)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)