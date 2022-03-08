import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

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


def create_triplet(X, Y):
    X = X.cpu()
    Y = Y.cpu()
    anchors, positives, negatives = torch.tensor([]), torch.tensor([]), torch.tensor([])
    
    for i in range(math.floor(X.size(0)/2)):
        random_idx = np.random.randint(0, X.size(0))
        anchor_x = X[random_idx]
        anchor_y = Y[random_idx]

        indices_for_pos = np.squeeze(np.where(Y == anchor_y))
        indices_for_neg = np.squeeze(np.where(Y != anchor_y))

        if indices_for_pos.size == 0 or indices_for_pos.size == 1 or indices_for_neg.size == 0 or indices_for_neg.size == 1:
            continue

        negative_x = X[indices_for_neg]
        anchor_x = anchor_x.unsqueeze(0)

        first_run = True
        for pos_idx in indices_for_pos:
            positive_x = X[pos_idx]
            positive_x = positive_x.unsqueeze(0)

            if first_run:
                anchor_x = anchor_x.repeat(negative_x.size(0), 1, 1, 1)
                first_run = False

            positive_x = torch.cat(negative_x.size(0) * [positive_x], dim=0)
            
            anchors = torch.cat((anchors, anchor_x), dim=0)
            positives = torch.cat((positives, positive_x), dim=0)
            negatives = torch.cat((negatives, negative_x), dim=0)

    return anchors, positives, negatives
