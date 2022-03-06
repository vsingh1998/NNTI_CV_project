#!/usr/bin/env python3

import argparse
import math
import copy
import warnings
warnings.filterwarnings("ignore")
from collections import OrderedDict

from dataloader import get_cifar10, get_cifar100
from model.wrn import WideResNet
from utils import accuracy, alpha_weight, create_triplet, TripletLoss

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import logging

# create and configure logger
logger = logging.getLogger()
fhandler = logging.FileHandler(filename='out.log', mode='a')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fhandler.setFormatter(formatter)
logger.addHandler(fhandler)
logger.setLevel(logging.INFO)


def main(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.dataset == "cifar10":
        args.num_classes = 10
        labeled_dataset, unlabeled_dataset, test_dataset = get_cifar10(args, 
                                                                args.datapath)
    if args.dataset == "cifar100":
        args.num_classes = 100
        labeled_dataset, unlabeled_dataset, test_dataset = get_cifar100(args, 
                                                                args.datapath)
    
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
    
    model       = WideResNet(depth= args.model_depth, num_classes= args.num_classes, 
                                widen_factor= args.model_width)
    model       = model.to(device)

    # define loss, optimizer and lr scheduler
    criterion = nn.CrossEntropyLoss().to(device)
    # triplet_loss = nn.TripletMarginLoss().to(device) # for siamese network
    triplet_loss = TripletLoss(device)
    optimizer = optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.1)

    # Pseudo dataset initialization
    pseudo_dataset_x = torch.tensor([]).to(device)
    pseudo_dataset_y = torch.tensor([]).long().to(device)

    args.epoch = math.ceil(args.total_iter / args.iter_per_epoch)
    # epochs of supervised training
    supervised_epochs = 100
    # epochs of supervised + pseudo labels learning, outermost loop
    combined_epochs = supervised_epochs + 100

    loss_log = []
    train_acc_log = []

    for epoch in range(combined_epochs):
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

            pseudo_elements = torch.numel(pseudo_dataset_x)
            model.train()
            
            if args.num_labeled == 250 or args.num_labeled == 2500:
                pred = model(x_l)

                total_loss = criterion(pred, y_l)
                acc = accuracy(pred.data, y_l, topk=(1,))[0]

                running_loss += total_loss.item()
                running_train_acc += acc

                # compute gradient and do SGD step
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
            

                if pseudo_elements != 0:
                    pred_unlabeled = model(pseudo_dataset_x)

                    total_loss = criterion(pred_unlabeled, pseudo_dataset_y)

                    running_loss += total_loss.item()
                    running_train_acc += acc

                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()
                
                scheduler.step()

            else:
                # train on labeled data for specified epochs (T1 in paper)
                if epoch < supervised_epochs:
                    pred = model(x_l)
                    acc = accuracy(pred.data, y_l, topk=(1,))[0] 
                    total_loss = criterion(pred, y_l)

                    running_loss += total_loss.item()
                    running_train_acc += acc

                    # compute gradient and do SGD step
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()
                    scheduler.step()

                else:

                    # labeled + pseudo combined training data
                    X_train = torch.cat((x_l, pseudo_dataset_x), dim=0)
                    Y_train = torch.cat((y_l, pseudo_dataset_y), dim=0)

                    # train model on combined data
                    model.train()
                    pred = model(X_train)

                    if pseudo_elements == 0:
                        total_loss = criterion(pred, Y_train)
                    else:
                        main_loss   = criterion(pred[:-pseudo_dataset_x.shape[0]], Y_train[:-pseudo_dataset_x.shape[0]])
                        pseudo_loss = criterion(pred[-pseudo_dataset_x.shape[0]:], Y_train[-pseudo_dataset_x.shape[0]:])
                        total_loss  = main_loss + alpha_weight(epoch, T1= supervised_epochs) * pseudo_loss

                    acc = accuracy(pred.data, Y_train, topk=(1,))[0]

                    running_loss += total_loss.item()
                    running_train_acc += acc

                    # compute gradient and do SGD step
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()
                    scheduler.step()

            # make prediction on unlabeled data
            model.eval()
            pred_ul = model(x_ul)
            # get class probabilities
            pred_prob = F.softmax(pred_ul, dim=1)
            
            # reinitialize empty pseudo dataset for current iteration
            pseudo_dataset_x = torch.tensor([]).to(device)
            pseudo_dataset_y = torch.tensor([]).long().to(device)

            # pseudo labeling
            for idx_ul, pred in enumerate(pred_prob):
                max_prob, max_prob_class = torch.max(pred, dim=-1)
                if max_prob > args.threshold:
                    pseudo_dataset_x = torch.cat((pseudo_dataset_x, x_ul[idx_ul].unsqueeze(0)), dim=0)
                    pseudo_dataset_y = torch.cat((pseudo_dataset_y, max_prob_class.unsqueeze(0)), dim=0)
        
        acc_per_epoch = running_train_acc / args.iter_per_epoch
        train_acc_log.append(acc_per_epoch)
        loss_per_epoch = running_loss / args.iter_per_epoch
        loss_log.append(loss_per_epoch)
        print('Epoch: ', epoch, 'Loss: ', loss_per_epoch, 'Accuracy: ', acc_per_epoch)
        logger.info(f'==>>> epoch: {epoch}, train loss: {loss_per_epoch}, train accuracy: {acc_per_epoch}')
        running_loss, running_train_acc = 0.0, 0.0 


    # fine-tune the model to learn embeddings in feature space
    # reinitialize empty pseudo dataset for current iteration
    pseudo_dataset_x = torch.tensor([]).to(device)
    pseudo_dataset_y = torch.tensor([]).long().to(device)
    
    snn_base = copy.deepcopy(model)
    snn_base.fc = nn.Linear(model.fc.in_features, 256)
    fc_relu = nn.ReLU(inplace=True)
    fc2 = nn.Linear(snn_base.fc.out_features, 128)
    fc_last = nn.Linear(128, 64)
    siamese_nn = nn.Sequential(snn_base, fc_relu, fc2, fc_relu, fc_last)

    optimizer = optim.SGD(siamese_nn.parameters(), args.lr,
                                momentum=args.momentum, weight_decay=args.wd)

    for param in list(siamese_nn.children())[:-3]:
        param.requires_grad = False

    siamese_nn = siamese_nn.to(device)
    siamese_nn.train()

    add_classifier = True 
    similarity_loss_log = []
 
    # run for 300 epochs starting from 200th
    # generate labels with 90% confidence and learn embeddings for first 70 epochs
    # in last 30 epochs, learns weights for last layer of classifier for final label prediction
    for epoch in range(combined_epochs, args.epoch):
        running_similarity_loss = 0.0

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

            model.eval()
            pred_ul = model(x_ul)
            # get class probabilities
            pred_prob = F.softmax(pred_ul, dim=1)
            
            # reinitialize empty pseudo dataset for current iteration
            pseudo_dataset_x = torch.tensor([]).to(device)
            pseudo_dataset_y = torch.tensor([]).long().to(device)

            # pseudo labeling
            for idx_ul, pred in enumerate(pred_prob):
                max_prob, max_prob_class = torch.max(pred, dim=-1)
                if max_prob > args.threshold:
                    pseudo_dataset_x = torch.cat((pseudo_dataset_x, x_ul[idx_ul].unsqueeze(0)), dim=0)
                    pseudo_dataset_y = torch.cat((pseudo_dataset_y, max_prob_class.unsqueeze(0)), dim=0)
            
            # labeled + pseudo combined training data
            X_train = torch.cat((x_l, pseudo_dataset_x), dim=0)
            Y_train = torch.cat((y_l, pseudo_dataset_y), dim=0)

            if epoch < (args.epoch - 30):
                anchors, positives, negatives, anchors_y, positives_y, negatives_y = create_triplet(X_train, Y_train)
                embed_A = siamese_nn(anchors.to(device))
                embed_P = siamese_nn(positives.to(device))
                embed_N = siamese_nn(negatives.to(device))

                # print(embed_A)
                # print('size embed_A', embed_A.size())

                embeddings = torch.cat((F.normalize(embed_A, p=2, dim=1), F.normalize(embed_P, p=2, dim=1), F.normalize(embed_N, p=2, dim=1)), dim=0)
                labels = torch.cat((anchors_y, positives_y, negatives_y), dim=0)

                # print(embeddings.size(), labels.size())

                # similarity_loss = triplet_loss(embed_A, embed_P, embed_N)
                similarity_loss = triplet_loss(embeddings, labels)
                running_similarity_loss += similarity_loss.item()

                print('similarity loss', similarity_loss.item())
                similarity_loss.backward()
                optimizer.step()

            elif add_classifier:
                last_layer = nn.Linear(64, args.num_classes)
                snn_classifier = nn.Sequential(siamese_nn, fc_relu, last_layer)
                optimizer = optim.SGD(snn_classifier.parameters(), args.lr,
                                momentum=args.momentum, weight_decay=args.wd)

                for param in list(snn_classifier.children())[:-1]:
                    param.requires_grad = False

                add_classifier = False

            else:
                snn_classifier = snn_classifier.to(device)
                snn_classifier.train()
                pred = snn_classifier(X_train)
                classifier_loss = criterion(pred, Y_train)

                optimizer.zero_grad()
                classifier_loss.backward()
                optimizer.step()

        similarity_loss_per_epoch = running_similarity_loss / args.iter_per_epoch
        similarity_loss_log.append(similarity_loss_per_epoch)
        logger.info(f'==>>> similarity loss: {similarity_loss}')
        scheduler.step()

    torch.save(snn_classifier.state_dict(), 'task3.pth')

    ## Testing
    running_acc = 0.0
    acc_log = []
    
    snn_classifier.eval()
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            pred = snn_classifier(inputs)
            acc = accuracy(pred.data, labels, topk=(1,))[0]
            running_acc += acc

        test_accuracy = running_acc.item() / batch_idx
        print('Accuracy: ', test_accuracy)
        acc_log.append(test_accuracy)
        logger.info(f'==>>> test accuracy: {test_accuracy}')

        running_acc = 0.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pseudo labeling \
                                        of CIFAR10/100 with pytorch")
    parser.add_argument("--dataset", default="cifar10", 
                        type=str, choices=["cifar10", "cifar100"])
    parser.add_argument("--datapath", default="./data/", 
                        type=str, help="Path to the CIFAR-10/100 dataset")
    parser.add_argument('--num-labeled', type=int, 
                        default=4000, help='Total number of labeled samples')
    parser.add_argument("--lr", default=0.003, type=float, 
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
    parser.add_argument('--total-iter', default=800*300, type=int,
                        help='total number of iterations to run')
    parser.add_argument('--iter-per-epoch', default=800, type=int,
                        help="Number of iterations to run per epoch")
    parser.add_argument('--num-workers', default=8, type=int,
                        help="Number of workers to launch during training")
    parser.add_argument('--threshold', type=float, default=0.9,
                        help='Confidence Threshold for pseudo labeling')
    parser.add_argument("--model-depth", type=int, default=16,
                        help="model depth for wide resnet") 
    parser.add_argument("--model-width", type=int, default=8,
                        help="model width for wide resnet")
    parser.add_argument('--milestones', action='append', type=int, default=[40, 80], 
                        help="Milestones for the LR scheduler") # see if useful, else rm
    parser.add_argument("--modelpath", default="./model/task3.pth", 
                        type=str, help="Path to save model")
    parser.add_argument("--dropout", default=0.0, type=float, 
                        help="Dropout rate for model")   

    args = parser.parse_args()

    main(args)