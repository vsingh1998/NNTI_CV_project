#!/usr/bin/env python3

import argparse
import math

from dataloader import get_cifar10, get_cifar100
from vat        import VATLoss
from utils      import accuracy, entropy_loss
from model.wrn  import WideResNet

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data   import DataLoader

import logging

#now we will create and configure logger
logger = logging.getLogger()
fhandler = logging.FileHandler(filename='out.log', mode='a')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fhandler.setFormatter(formatter)
logger.addHandler(fhandler)
logger.setLevel(logging.INFO)

def main(args):
    if args.dataset == "cifar10":
        args.num_classes = 10
        labeled_dataset, unlabeled_dataset, test_dataset = get_cifar10(args, 
                                                                args.datapath)
    if args.dataset == "cifar100":
        args.num_classes = 100
        labeled_dataset, unlabeled_dataset, test_dataset = get_cifar100(args, 
                                                                args.datapath)
    args.epoch = math.ceil(args.total_iter / args.iter_per_epoch)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    labeled_loader      = iter(DataLoader(labeled_dataset, 
                                    batch_size = args.train_batch, 
                                    shuffle = True, 
                                    num_workers=args.num_workers))
    unlabeled_loader    = iter(DataLoader(unlabeled_dataset, 
                                    batch_size=args.train_batch,
                                    shuffle = True, 
                                    num_workers=args.num_workers))
    test_loader         = DataLoader(test_dataset,
                                    batch_size = args.test_batch,
                                    shuffle = False, 
                                    num_workers=args.num_workers)
    
    model       = WideResNet(args.model_depth, 
                                args.num_classes, widen_factor=args.model_width,
                                dropRate= args.dropout)
    model       = model.to(device)

    ############################################################################
    # TODO: SUPPLY your code
    # define loss, optimizer and lr scheduler
    criterion = nn.CrossEntropyLoss().to(device)
    vat_loss = VATLoss(args)
    optimizer = optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum, weight_decay=args.wd)

    loss_log = []
    train_acc_log = []
    ############################################################################
    
    for epoch in range(args.epoch):
        running_loss = 0.0
        running_train_acc = 0.0
        model.train()
        for i in range(args.iter_per_epoch):
            try:
                x_l, y_l    = next(labeled_loader)
            except StopIteration:
                labeled_loader      = iter(DataLoader(labeled_dataset, 
                                            batch_size = args.train_batch, 
                                            shuffle = True, 
                                            num_workers=args.num_workers))
                x_l, y_l    = next(labeled_loader)
            
            try:
                x_ul, _     = next(unlabeled_loader)
            except StopIteration:
                unlabeled_loader    = iter(DataLoader(unlabeled_dataset, 
                                            batch_size=args.train_batch,
                                            shuffle = True, 
                                            num_workers=args.num_workers))
                x_ul, _     = next(unlabeled_loader)
            
            x_l, y_l    = x_l.to(device), y_l.to(device)
            x_ul        = x_ul.to(device)
            ####################################################################
            # TODO: SUPPLY you code
            v_loss = vat_loss.forward(model.eval(), x_ul)
            model.train()
            pred = model(x_l)
            classification_loss = criterion(pred, y_l)

            # new loss
            ul_pred = model(x_ul)
            entropy_loss_ul = entropy_loss(ul_pred)

            total_loss = classification_loss + args.alpha * v_loss + entropy_loss_ul # new
            print("For {} in epoch {}: classification_loss= {} v_loss= {} total_loss= {}".format(i, epoch, classification_loss, v_loss, total_loss))
            acc = accuracy(pred.data, y_l, topk=(1,))[0] 
            
            running_loss += total_loss.item()
            running_train_acc += acc

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        acc_per_epoch = running_train_acc / args.iter_per_epoch
        train_acc_log.append(acc_per_epoch)
        loss_per_epoch = running_loss / args.iter_per_epoch
        loss_log.append(loss_per_epoch)
        print('Epoch: ', epoch, 'Loss: ', loss_per_epoch, 'Accuracy: ', acc_per_epoch.item())
        logger.info(f'==>>> epoch: {epoch}, train loss: {loss_per_epoch}, train accuracy: {acc_per_epoch}')
        running_loss, running_train_acc = 0.0, 0.0

    with open('loss_log.txt', 'w') as f:
        for item in loss_log:
            f.write("%s\n" % item)

    with open('acc_log.txt', 'w') as f:
        for item in train_acc_log:
            f.write("%s\n" % item)

    torch.save(model.state_dict(), 'task2_cifar100_2500.pth')

    ### Test
    running_acc = 0.0
    acc_log = []
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            pred = model(inputs)
            acc = accuracy(pred.data, labels, topk=(1,))[0]
            running_acc += acc

        test_accuracy = running_acc.item() / batch_idx
        print('Accuracy: ', test_accuracy)
        acc_log.append(test_accuracy)
        logger.info(f'==>>> test accuracy: {test_accuracy}')
        running_acc = 0.0
        
            ####################################################################



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Virtual adversarial training \
                                        of CIFAR10/100 using with pytorch")
    parser.add_argument("--dataset", default="cifar100", 
                        type=str, choices=["cifar10", "cifar100"])
    parser.add_argument("--datapath", default="./data/", 
                        type=str, help="Path to the CIFAR-10/100 dataset")
    parser.add_argument('--num-labeled', type=int, 
                        default=2500, help='Total number of labeled samples')
    parser.add_argument("--lr", default=0.001, type=float, 
                        help="The initial learning rate") 
    parser.add_argument("--momentum", default=0.9, type=float,
                        help="Optimizer momentum")
    parser.add_argument("--wd", default=0.00005, type=float,
                        help="Weight decay")
    parser.add_argument("--expand-labels", action="store_true", 
                        help="expand labels to fit eval steps")
    parser.add_argument('--train-batch', default=64, type=int,
                        help='train batchsize')
    parser.add_argument('--test-batch', default=64, type=int,
                        help='train batchsize')
    parser.add_argument('--total-iter', default=256*100, type=int,
                        help='total number of iterations to run')
    parser.add_argument('--iter-per-epoch', default=256, type=int,
                        help="Number of iterations to run per epoch")
    parser.add_argument('--num-workers', default=8, type=int,
                        help="Number of workers to launch during training")                        
    parser.add_argument('--alpha', type=float, default=1.0, metavar='ALPHA',
                        help='regularization coefficient (default: 0.01)')
    parser.add_argument("--dataout", type=str, default="./path/to/output/",
                        help="Path to save log files")
    parser.add_argument("--model-depth", type=int, default=16,
                        help="model depth for wide resnet") 
    parser.add_argument("--model-width", type=int, default=8,
                        help="model width for wide resnet")
    parser.add_argument("--vat-xi", default=1e-6, type=float, 
                        help="VAT xi parameter")
    parser.add_argument("--vat-eps", default=6.0, type=float, 
                        help="VAT epsilon parameter") 
    parser.add_argument("--vat-iter", default=1, type=int, 
                        help="VAT iteration parameter")

    # added arguments
    parser.add_argument('--milestones', action='append', type=int, default=[], 
                        help="Milestones for the LR scheduler")# see if useful, else rm
    parser.add_argument("--modelpath", default="./model/task2.pth", 
                        type=str, help="Path to save model")
    parser.add_argument("--dropout", default=0.0, type=float, 
                        help="Dropout rate for model") 

    # Add more arguments if you need them
    # Describe them in help
    # You can (and should) change the default values of the arguments
    
    args = parser.parse_args()

    main(args)
