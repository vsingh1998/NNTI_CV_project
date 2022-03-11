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
    """
    Creates triplets given an input batch
    Returns anchors, positives and negatives, each of same size
    """

    # move args to cpu for numpy
    X = X.cpu()
    Y = Y.cpu()
    # initialize empty tensors
    anchors, positives, negatives = torch.tensor([]), torch.tensor([]), torch.tensor([])
    
    for i in range(X.size(0)):

        # select a randon image from batch to be anchor
        random_idx = np.random.randint(0, X.size(0))
        anchor_x = X[random_idx]
        anchor_y = Y[random_idx]

        # get indices of positives (having same label as anchor)
        indices_for_pos = np.squeeze(np.where(Y == anchor_y))
        # get indices of negatives (having different label than anchor)
        indices_for_neg = np.squeeze(np.where(Y != anchor_y))

        if indices_for_pos.size == 0 or indices_for_pos.size == 1 or indices_for_neg.size == 0 or indices_for_neg.size == 1:
            continue

        # get all negatives
        negative_x = X[indices_for_neg]
        anchor_x = anchor_x.unsqueeze(0)

        # flag to be used later to create a batch of same anchors given a positive
        first_run = True

        # run loop for every positive
        for pos_idx in indices_for_pos:
            positive_x = X[pos_idx]
            positive_x = positive_x.unsqueeze(0)

            if first_run:
                # create batch of same achors
                # batch size equal to number of negatives
                anchor_x = anchor_x.repeat(negative_x.size(0), 1, 1, 1)
                first_run = False

            # create batch of same positives
            positive_x = torch.cat(negative_x.size(0) * [positive_x], dim=0)
            
            # concatenate latest batch of (same) anchors, (same) positives and negatives
            anchors = torch.cat((anchors, anchor_x), dim=0)
            positives = torch.cat((positives, positive_x), dim=0)
            negatives = torch.cat((negatives, negative_x), dim=0)

    return anchors, positives, negatives
