import torch

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

def _l2_normalize(x):
    """
    TODO
    reference: https://github.com/lyakaap/VAT-pytorch
    """
    reshaped_x = x.view(x.shape[0], -1, *(1 for _ in range(x.dim() - 2)))
    x /= torch.norm(reshaped_x, dim=1, keepdim=True) + 1e-8
    
    return x
