from torchvision.utils import save_image
from utils import L2_norm, kl_div_with_logit

import torch
import torch.nn as nn
import torch.nn.functional as F

class VATLoss(nn.Module):

    def __init__(self, args):
        super(VATLoss, self).__init__()
        self.xi = args.vat_xi
        self.eps = args.vat_eps
        self.vat_iter = args.vat_iter
        # flag to get adversarial example for report
        # set to True to get image
        self.save_img = False 

    def forward(self, model, x):
        # prepare a random tensor r 
        r = torch.Tensor(x.size()).normal_().to(x.device)
        r = L2_norm(r)

        with torch.no_grad():
            # remove softmax
            pred = model(x)

        for i in range(self.vat_iter):
            r.requires_grad_()
            adv_examples = x + self.xi * r
            adv_pred = model(adv_examples)
            adv_distance = kl_div_with_logit(pred, adv_pred)
            adv_distance.backward()
            r = r.grad.data
            model.zero_grad()

        # for i in range(self.vat_iter):
        #     r.requires_grad_()
        #     adv_examples = x + self.xi * r
        #     adv_pred = F.log_softmax(model(adv_examples), dim=1)
        #     kldiv_loss = nn.KLDivLoss(reduction='batchmean')
        #     adv_distance = kldiv_loss(adv_pred, pred)
        #     adv_distance.backward()
        #     r = r.grad
        #     model.zero_grad()

        r = L2_norm(r)
        r_adv = r * self.eps
        adv_pred = model(x + r_adv)
        # adv_pred = F.log_softmax(model(x + r_adv), dim=1)

        loss = kl_div_with_logit(pred, adv_pred)
        # kldiv_loss = nn.KLDivLoss(reduction='batchmean')
        # loss = kldiv_loss(adv_pred, pred)

        # print('x_l pred', torch.argmax(pred[0]), 'adv_pred', torch.argmax(adv_pred[0]))
        if self.save_img:
            test = x + r_adv
            save_image(x[0], 'x1.png')
            save_image(test[0], 'x1_adv.png')
            self.save_img = False

        return loss
