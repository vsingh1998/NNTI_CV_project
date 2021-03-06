import argparse
import math
from sys import prefix
from tokenize import PseudoExtras

from dataloader import get_cifar10, get_cifar100
from utils      import accuracy, alpha_weight

from model.wrn  import WideResNet

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data   import DataLoader
import numpy as np


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

    # define loss, optimizer and learning rate scheduler
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.1)
    
    # pseudo dataset initialization
    pseudo_dataset????_x = torch.tensor([]).to(device)
    pseudo_dataset????_y = torch.tensor([]).long().to(device)

    supervised_epochs = 20 # T1 in paper
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
            # counting number of elements in pseudo dataset
            pseudo_elements = torch.numel(pseudo_datase?????_x)

            model.train()
            
            # training for CIFAR10 (250 labeled) and CIFAR100 (2500 labeled)
            if args.num_labeled == 250 or args.num_labeled == 2500:

                # prediction on labeled data
                pred = model(x_l)

                # loss for labeled data
                total_loss = criterion(pred, y_l)

                # accuracy for labeled data
                acc = accuracy(pred.data, y_l, topk=(1,))[0]

                running_loss += total_loss.item()
                running_train_acc += acc

                # compute gradient and do SGD step
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                # train on pseudo labeled data
                if pseudo_elements != 0:
                    
                    # prediction on pseudo labeled data
                    pred_unlabeled = model(pseudo_datase?????_x)
                    
                    # loss for pseudo labeled data
                    total_loss = criterion(pred_unlabeled, pseudo_datase?????_y)

                    running_loss += total_loss.item()
                    running_train_acc += acc
                    
                    # compute gradient and do SGD step
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()

            else:
                # training for CIFAR10 (4000 labeled) and CIFAR100 (10000 labeled)
                # train on labeled data for specified epochs (T1 in paper)
                if epoch < supervised_epochs:

                    # prediction on labeled data
                    pred = model(x_l)

                    # loss for labeled data
                    total_loss = criterion(pred, y_l)

                    # accuracy for labeled data
                    acc = accuracy(pred.data, y_l, topk=(1,))[0]

                    running_loss += total_loss.item()
                    running_train_acc += acc

                    # compute gradient and do SGD step
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()

                else:
                    # labeled + pseudo combined training data
                    X_train = torch.cat((x_l, pseudo_dataset????_x), dim=0)
                    Y_train = torch.cat((y_l, pseudo_dataset????_y), dim=0)

                    # train model on combined data
                    model.train()
                    pred = model(X_train)

                    if pseudo_elements == 0:
                        # loss for labeled data
                        total_loss = criterion(pred, Y_train)
                    else:
                        # loss for labeled data
                        main_loss   = criterion(pred[:-pseudo_datase?????_x.shape[0]], Y_train[:-pseudo_datase?????_x.shape[0]])
                        # loss for pseudo labeled data
                        pseudo_loss = criterion(pred[-pseudo_datase?????_x.shape[0]:], Y_train[-pseudo_datase?????_x.shape[0]:])
                        # total loss
                        total_loss  = main_loss + alpha_weight(epoch, T1= supervised_epochs) * pseudo_loss

                    acc = accuracy(pred.data, Y_train, topk=(1,))[0]

                    running_loss += total_loss.item()
                    running_train_acc += acc

                    # compute gradient and do SGD step
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()

            # make prediction on unlabeled data
            model.eval()
            pred_ul = model(x_ul)
            # get class probabilities
            pred_prob = F.softmax(pred_ul, dim=1)
            
            # reinitialize empty pseudo dataset for current iteration
            pseudo_dataset????_x = torch.tensor([]).to(device)
            pseudo_dataset????_y = torch.tensor([]).long().to(device)

            # pseudo labeling
            for idx_ul, pred in enumerate(pred_prob):
                max_prob, max_prob_class = torch.max(pred, dim=-1)
                if max_prob > args.threshold:
                    pseudo_dataset????_x = torch.cat((pseudo_dataset????_x, x_ul[idx_ul].unsqueeze(0)), dim=0)
                    pseudo_dataset????_y = torch.cat((pseudo_dataset????_y, max_prob_class.unsqueeze(0)), dim=0)
        
        # calculate accuracy per epoch
        acc_per_epoch = running_train_acc / args.iter_per_epoch
        train_acc_log.append(acc_per_epoch)

        # calculate loss per epoch
        loss_per_epoch = running_loss / args.iter_per_epoch
        loss_log.append(loss_per_epoch)

        print('Epoch: ', epoch, 'Loss: ', loss_per_epoch, 'Accuracy: ', acc_per_epoch)

        running_loss, running_train_acc = 0.0, 0.0
        scheduler.step()

    # save model
    torch.save(model.state_dict(), args.modelpath)

    ## Test
    model.eval()
    running_acc = 0.0

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # prediction on test data
            pred = model(inputs)

            # accuracy on test data
            acc = accuracy(pred.data, labels, topk=(1,))[0]
            running_acc += acc

        test_accuracy = running_acc.item() / batch_idx
        print('Accuracy: ', test_accuracy)

            ####################################################################



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pseudo labeling \
                                        of CIFAR10/100 with pytorch")
    parser.add_argument("--dataset", default="cifar10", 
                        type=str, choices=["cifar10", "cifar100"])
    parser.add_argument("--datapath", default="./data/", 
                        type=str, help="Path to the CIFAR-10/100 dataset")
    parser.add_argument('--num-labeled', type=int, 
                        default=4000, help='Total number of labeled samples')
    parser.add_argument("--lr", default=0.03, type=float, 
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
                        help='test batchsize')
    parser.add_argument('--total-iter', default=1024*512, type=int,
                        help='total number of iterations to run')
    parser.add_argument('--iter-per-epoch', default=1024, type=int,
                        help="Number of iterations to run per epoch")
    parser.add_argument('--num-workers', default=8, type=int,
                        help="Number of workers to launch during training")
    parser.add_argument('--threshold', type=float, default=0.95,
                        help='Confidence Threshold for pseudo labeling')
    parser.add_argument("--dataout", type=str, default="./path/to/output/",
                        help="Path to save log files")
    parser.add_argument("--model-depth", type=int, default=16,
                        help="model depth for wide resnet") 
    parser.add_argument("--model-width", type=int, default=8,
                        help="model width for wide resnet")
    # added arguments
    parser.add_argument('--milestones', action='append', type=int, default=[50, 80], 
                        help="Milestones for the LR scheduler")
    parser.add_argument("--modelpath", default="./model/task1.pth", 
                        type=str, help="Path to save model")
    parser.add_argument("--dropout", default=0.3, type=float, 
                        help="Dropout rate for model")                    
    
    args = parser.parse_args()

    main(args)