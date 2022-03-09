from torchvision.utils import save_image
from utils import L2_norm

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
        self.save_img = True 

    def forward(self, model, x):
        # prepare a random tensor r from Gaussian distribution
        r = torch.Tensor(x.size()).normal_().to(x.device)

        # L2 normalize r
        r = L2_norm(r)

        # generate predictions on unlabeled data
        with torch.no_grad():
            pred = F.softmax(model(x), dim=1)

        for i in range(self.vat_iter):
            # turn on grad for r
            r.requires_grad_()

            # generate adversarial examples
            adv_examples = x + self.xi * r

            # prediction on adversarial examples
            adv_pred = F.log_softmax(model(adv_examples), dim=1)

            # initialize KL divergence 
            kldiv_loss = nn.KLDivLoss(reduction='batchmean')

            # calculate KL divergence between adversarial and original image
            adv_distance = kldiv_loss(adv_pred, pred)

            # backpropagate
            adv_distance.backward()
            r = r.grad
            model.zero_grad()

        r = L2_norm(r)
        # calculate r in the adversarial direction
        r_adv = r * self.eps

        # generate prediction on adversarial example
        adv_pred = F.log_softmax(model(x + r_adv), dim=1)

        # calculate loss
        kldiv_loss = nn.KLDivLoss(reduction='batchmean')
        loss = kldiv_loss(adv_pred, pred)

        # save adversarial image
        if self.save_img:
            test = x + r_adv
            save_image(x[0], 'x1.png')
            save_image(test[0], 'x1_adv.png')
            self.save_img = False

        return loss
