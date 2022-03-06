import torch
import torch.nn as nn
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
    anchors_y, positives_y, negatives_y = torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long) 

    for i in range(X.size(0)):

        random_idx = np.random.randint(0, X.size(0))
        anchor_x = X[random_idx]
        anchor_y = Y[random_idx]

        indices_for_pos = np.squeeze(np.where(Y == anchor_y))
        indices_for_neg = np.squeeze(np.where(Y != anchor_y))

        if indices_for_pos.size == 0 or indices_for_pos.size == 1:
            continue

        pos_indices = indices_for_pos[np.random.randint(0, indices_for_pos.size)]
        neg_indices = indices_for_neg[np.random.randint(0, indices_for_neg.size)]
        positive_x, positive_y = X[pos_indices], Y[pos_indices]
        negative_x, negative_y = X[neg_indices], Y[neg_indices]

        # if torch.numel(anchors) == 0:
        #     anchors = anchor_x.unsqueeze(0)
        #     positives = positive_x.unsqueeze(0)
        #     negatives = negative_x.unsqueeze(0)
        # else:
        anchors = torch.cat((anchors, anchor_x.unsqueeze(0)), dim=0)
        positives = torch.cat((positives, positive_x.unsqueeze(0)), dim=0)
        negatives = torch.cat((negatives, negative_x.unsqueeze(0)), dim=0)

        anchors_y = torch.cat((anchors_y, anchor_y.unsqueeze(0)), dim=0)
        positives_y = torch.cat((positives_y, positive_y.unsqueeze(0)), dim=0)
        negatives_y = torch.cat((negatives_y, negative_y.unsqueeze(0)), dim=0)

    return anchors, positives, negatives, anchors_y, positives_y, negatives_y


def pairwise_distance_torch(embeddings, device):
    """Computes the pairwise distance matrix with numerical stability.
    output[i, j] = || feature[i, :] - feature[j, :] ||_2
    Args:
      embeddings: 2-D Tensor of size [number of data, feature dimension].
    Returns:
      pairwise_distances: 2-D Tensor of size [number of data, number of data].
    """

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
    """Computes the triplet loss_functions with semi-hard negative mining.
       The loss_functions encourages the positive distances (between a pair of embeddings
       with the same labels) to be smaller than the minimum negative distance
       among which are at least greater than the positive distance plus the
       margin constant (called semi-hard negative) in the mini-batch.
       If no such negative exists, uses the largest negative distance instead.
       See: https://arxiv.org/abs/1503.03832.
       We expect labels `y_true` to be provided as 1-D integer `Tensor` with shape
       [batch_size] of multi-class integer labels. And embeddings `y_pred` must be
       2-D float `Tensor` of l2 normalized embedding vectors.
       Args:
         margin: Float, margin term in the loss_functions definition. Default value is 1.0.
         name: Optional name for the op.
       """

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
