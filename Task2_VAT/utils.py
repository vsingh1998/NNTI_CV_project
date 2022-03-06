import torch
import torch.nn.functional as F

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
    return -(p*F.log_softmax(p_ul, dim=1)).sum(dim=1).mean(dim=0)
