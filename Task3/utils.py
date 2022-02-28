import torch
import torch.nn.functional as F
import numpy as np

def accuracy(output, target, topk=(1,)):
    """
    Function taken from pytorch examples:
    https://github.com/pytorch/examples/blob/master/imagenet/main.py
    Computes the accuracy over the k top predictions for the specified
    values of k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        
        return res


def L2_norm(x):
    """
    Function taken from: 
    https://github.com/lyakaap/VAT-pytorch
    Normalize the incoming tensor x, 
    such that ||L2Norm(x)||_2 = 1
    """
    reshaped_x = x.view(x.shape[0], -1, *(1 for _ in range(x.dim() - 2)))
    x /= torch.norm(reshaped_x, dim=1, keepdim=True) + 1e-8
    
    return x


def entropy_loss(p_ul):
    p = F.softmax(p_ul, dim=1)
    return - torch.sum(p * F.log_softmax(p_ul)) / p_ul.shape[0]


def alpha_weight(step, T1 = 30, T2 = 100, af = 3):
    """
    TODO
    """

    if step < T1:
        return 0.0
    elif step > T2:
        return af
    else:
        return ((step-T1) / (T2-T1)) * af


def create_triplet(X, Y):
    X = X.cpu()
    Y = Y.cpu()
    anchors = torch.tensor([])
    positives = torch.tensor([])
    negatives = torch.tensor([])

    for i in range(X.size(0)):

        random_idx = np.random.randint(0, X.size(0))
        anchor_x = X[random_idx]
        anchor_y = Y[random_idx]

        indices_for_pos = np.squeeze(np.where(Y == anchor_y))
        indices_for_neg = np.squeeze(np.where(Y != anchor_y))

        if indices_for_pos.size == 0 or indices_for_pos.size == 1:
            continue

        positive_x = X[indices_for_pos[np.random.randint(0, indices_for_pos.size)]]
        negative_x = X[indices_for_neg[np.random.randint(0, indices_for_neg.size)]]

        anchors = torch.cat((anchors, anchor_x), dim=0)
        positives = torch.cat((positives, positive_x), dim=0)
        negatives = torch.cat((negatives, negative_x), dim=0)

    return anchors, positives, negatives
