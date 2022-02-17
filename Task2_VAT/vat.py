
import torch
import torch.nn as nn
import torch.nn.functional as F

class VATLoss(nn.Module):

    def __init__(self, args):
        super(VATLoss, self).__init__()
        self.xi = args.vat_xi
        self.eps = args.vat_eps
        self.vat_iter = args.vat_iter

    def forward(self, model, x):
        # prepare a random tensor r 
        r = torch.Tensor(x.size()).normal_().to(x.device)
        r = r.norm(dim=1, p=2)

        with torch.no_grad():
            pred = F.softmax(model(x), dim=1)

        for i in range(self.vat_iter):
            r.requires_grad_()
            adv_examples = x + self.xi * r
            adv_pred = F.softmax(model(adv_examples), dim=1)
            adv_distance = F.kl_div(pred, adv_pred)
            adv_distance.backward()
            r = r.grad
            r = r.norm(dim=1, p=2)
            model.zero_grad()

        r_adv = r * self.eps
        adv_pred = model(x + r_adv)
        loss = F.kl_div(pred, adv_pred)

        return loss
