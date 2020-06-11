import torch


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.sigmoid(output) > 0.5
        assert len(pred) == len(target)
        correct = 0
        correct += torch.sum(pred.float() == target).item()
    return correct / len(target)


def accuracy_oneshot(output, target):
    with torch.no_grad():
        # TODO support batch
        pred = output.max(0)[1]
        assert len(pred) == len(target)
        correct = torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=0)[1]
        # assert pred.shape[0] == len(target)
        assert pred.shape[1] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[i] == target).item()
    return correct / len(target)
