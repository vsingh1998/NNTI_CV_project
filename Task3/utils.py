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
    anchors, positives, negatives = torch.tensor([]), torch.tensor([]), torch.tensor([])
    anchors_y, positives_y, negatives_y = torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long) 

    for i in range(math.floor(X.size(0)/2)):
        random_idx = np.random.randint(0, X.size(0))
        anchor_x = X[random_idx]
        anchor_y = Y[random_idx]

        indices_for_pos = np.squeeze(np.where(Y == anchor_y))
        # print(indices_for_pos)
        indices_for_neg = np.squeeze(np.where(Y != anchor_y))
        # print(indices_for_neg)

        if indices_for_pos.size == 0 or indices_for_pos.size == 1:
            continue

        negative_x, negative_y = X[indices_for_neg], Y[indices_for_neg]
        anchor_x = anchor_x.unsqueeze(0)
        # print('initial anchor_x shape', anchor_x.size())
        anchor_y = anchor_y.unsqueeze(0)
        # print('negative_x.size()', negative_x.size())

        # pos_indices = indices_for_pos[np.random.randint(0, indices_for_pos.size)]
        # neg_indices = indices_for_neg[np.random.randint(0, indices_for_neg.size)]
        # positive_x, positive_y = X[pos_indices], Y[pos_indices]
        # negative_x, negative_y = X[neg_indices], Y[neg_indices]
        first_run = True
        for pos_idx in indices_for_pos:
            positive_x, positive_y = X[pos_idx], Y[pos_idx]
            positive_x = positive_x.unsqueeze(0)
            positive_y = positive_y.unsqueeze(0)

            if first_run:
                anchor_x = anchor_x.repeat(negative_x.size(0), 1, 1, 1)
                # print('anchor_x.shape', anchor_x.shape)
                anchor_y = torch.cat(negative_x.size(0) * [anchor_y], dim=0)
                first_run = False

            positive_x = torch.cat(negative_x.size(0) * [positive_x], dim=0)
            # print('positive_x.shape', positive_x.shape)
            positive_y = torch.cat(negative_x.size(0) * [positive_y], dim=0)
            # print('positive_y.shape', positive_y.shape)
            # anchor_x = torch.cat(negative_x.size(0) * [anchor_x], dim=0)
            
            # print('anchor_x.size()', anchor_x.size())
            anchors = torch.cat((anchors, anchor_x), dim=0)
            print('anchors.size()', anchors.size())
            positives = torch.cat((positives, positive_x), dim=0)
            negatives = torch.cat((negatives, negative_x), dim=0)
            # print('negatives.size()', negatives.size())

            anchors_y = torch.cat((anchors_y, anchor_y), dim=0)
            positives_y = torch.cat((positives_y, positive_y), dim=0)
            negatives_y = torch.cat((negatives_y, negative_y), dim=0)

    return anchors, positives, negatives, anchors_y, positives_y, negatives_y


def pairwise_distance_torch(embeddings, device):

    # pairwise distance matrix with precise embeddings
    precise_embeddings = embeddings.to(dtype=torch.float32)

    c1 = torch.pow(precise_embeddings, 2).sum(axis=-1)
    c2 = torch.pow(precise_embeddings.transpose(0, 1), 2).sum(axis=0)
    c3 = precise_embeddings @ precise_embeddings.transpose(0, 1)

    c1 = c1.reshape((c1.shape[0], 1))
    c2 = c2.reshape((1, c2.shape[0]))
    c12 = c1 + c2
    pairwise_distances_squared = c12 - 2.0 * c3

    # Deal with numerical inaccuracies. Set small negatives to zero.
    pairwise_distances_squared = torch.max(pairwise_distances_squared, torch.tensor([0.]).to(device))
    # Get the mask where the zero distances are at.
    error_mask = pairwise_distances_squared.clone()
    error_mask[error_mask > 0.0] = 1.
    error_mask[error_mask <= 0.0] = 0.

    pairwise_distances = torch.mul(pairwise_distances_squared, error_mask)

    # Explicitly set diagonals to zero.
    mask_offdiagonals = torch.ones((pairwise_distances.shape[0], pairwise_distances.shape[1])) - torch.diag(torch.ones(pairwise_distances.shape[0]))
    pairwise_distances = torch.mul(pairwise_distances.to(device), mask_offdiagonals.to(device))
    return pairwise_distances


def TripletSemiHardLoss(y_true, y_pred, device, margin=1.0):

    labels, embeddings = y_true, y_pred

    # Reshape label tensor to [batch_size, 1].
    lshape = labels.shape
    labels = torch.reshape(labels, [lshape[0], 1])

    pdist_matrix = pairwise_distance_torch(embeddings, device)

    # Build pairwise binary adjacency matrix.
    adjacency = torch.eq(labels, labels.transpose(0, 1))
    # Invert so we can select negatives only.
    adjacency_not = adjacency.logical_not()

    batch_size = labels.shape[0]

    # Compute the mask.
    pdist_matrix_tile = pdist_matrix.repeat(batch_size, 1)
    adjacency_not_tile = adjacency_not.repeat(batch_size, 1)

    transpose_reshape = pdist_matrix.transpose(0, 1).reshape(-1, 1)
    greater = pdist_matrix_tile > transpose_reshape

    mask = adjacency_not_tile & greater

    # final mask
    mask_step = mask.to(dtype=torch.float32)
    mask_step = mask_step.sum(axis=1)
    mask_step = mask_step > 0.0
    mask_final = mask_step.reshape(batch_size, batch_size)
    mask_final = mask_final.transpose(0, 1)

    adjacency_not = adjacency_not.to(dtype=torch.float32)
    mask = mask.to(dtype=torch.float32)

    # negatives_outside: smallest D_an where D_an > D_ap.
    axis_maximums = torch.max(pdist_matrix_tile, dim=1, keepdim=True)
    masked_minimums = torch.min(torch.mul(pdist_matrix_tile - axis_maximums[0], mask), dim=1, keepdim=True)[0] + axis_maximums[0]
    negatives_outside = masked_minimums.reshape([batch_size, batch_size])
    negatives_outside = negatives_outside.transpose(0, 1)

    # negatives_inside: largest D_an.
    axis_minimums = torch.min(pdist_matrix, dim=1, keepdim=True)
    masked_maximums = torch.max(torch.mul(pdist_matrix - axis_minimums[0], adjacency_not), dim=1, keepdim=True)[0] + axis_minimums[0]
    negatives_inside = masked_maximums.repeat(1, batch_size)

    semi_hard_negatives = torch.where(mask_final, negatives_outside, negatives_inside)

    loss_mat = margin + pdist_matrix - semi_hard_negatives

    mask_positives = adjacency.to(dtype=torch.float32) - torch.diag(torch.ones(batch_size)).to(device)
    num_positives = mask_positives.sum()

    triplet_loss = (torch.max(torch.mul(loss_mat, mask_positives), torch.tensor([0.]).to(device))).sum() / num_positives
    triplet_loss = triplet_loss.to(dtype=embeddings.dtype)
    return triplet_loss


class TripletLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, input, target, **kwargs):
        return TripletSemiHardLoss(target, input, self.device)
