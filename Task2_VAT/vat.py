from torchvision.utils import save_image
from utils import _l2_normalize

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
        r = _l2_normalize(r)

        with torch.no_grad():
            pred = F.softmax(model(x), dim=1)

        for i in range(self.vat_iter):
            r.requires_grad_()
            adv_examples = x + self.xi * r
            adv_pred = F.log_softmax(model(adv_examples), dim=1)
            adv_distance = F.kl_div(pred, adv_pred, reduction='batchmean')
            adv_distance.backward()
            r = r.grad
            r = _l2_normalize(r)
            model.zero_grad()

        r_adv = r * self.eps
        # test = x + r_adv #rm after debugging
        
        adv_pred = F.log_softmax(model(x + r_adv), dim=1)
        loss = F.kl_div(pred, adv_pred)

        # print('x_l pred', torch.argmax(pred[0]), 'adv_pred', torch.argmax(adv_pred[0]))
        # save_image(x[0], 'x1.png')
        # save_image(test[0], 'x1_adv.png')

        return loss
