#!/usr/bin/env python3

import argparse
import math
import copy
import logging
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
    
    args.epoch = math.ceil(args.total_iter / args.iter_per_epoch)

    model       = WideResNet(depth= args.model_depth, num_classes= args.num_classes, 
                                widen_factor= args.model_width)
    model       = model.to(device)

    # load trained model from Task 1
    model.load_state_dict(torch.load(args.modelpath, map_location=device))
        
    # fine-tune the model to learn embeddings in feature space
    # snn_base = copy.deepcopy(model)
    model.fc = nn.Linear(model.fc.in_features, 256)
    fc_relu = nn.ReLU(inplace=True)
    fc2 = nn.Linear(model.fc.out_features, 128)
    fc_last = nn.Linear(128, 64)

    siamese_nn = nn.Sequential(model, fc_relu, fc2, fc_relu, fc_last)
    siamese_nn = siamese_nn.to(device)
    siamese_nn.train()

    # define loss, optimizer and lr scheduler
    # triplet_loss = TripletLoss(device=device)
    triplet_loss = nn.TripletMarginLoss().to(device) # for siamese network
    optimizer = optim.SGD(siamese_nn.parameters(), args.lr,
                                momentum=args.momentum, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.1)

    # for param in list(siamese_nn.children())[:-3]:
    #     param.requires_grad = False

    add_classifier = True 
    similarity_loss_log = []
 
    # generate labels with 90% confidence and learn embeddings for first 70 epochs
    # in last 50 epochs, learns weights for last layer of classifier for final label prediction
    for epoch in range(args.epoch):
        print('epoch', epoch)
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
                    pseudo_dataset_x = torch.cat((pseudo_dataset_x, 
                                                    x_ul[idx_ul].unsqueeze(0)), dim=0)
                    pseudo_dataset_y = torch.cat((pseudo_dataset_y, 
                                                    max_prob_class.unsqueeze(0)), dim=0)
            
            # labeled + pseudo combined training data
            X_train = torch.cat((x_l, pseudo_dataset_x), dim=0)
            Y_train = torch.cat((y_l, pseudo_dataset_y), dim=0)

            if epoch < (args.epoch - 50): # - 50
                anchors, positives, negatives = create_triplet(X_train, Y_train) # , anchors_y, positives_y, negatives_y
                if anchors.size(0) == 0 or positives.size(0) == 0 or negatives.size(0) == 0:
                    continue
                embed_A = siamese_nn(anchors.to(device))
                embed_P = siamese_nn(positives.to(device))
                embed_N = siamese_nn(negatives.to(device))

                print('size embed_A', embed_A.size())

                # embeddings = torch.cat((F.normalize(embed_A, p=2, dim=1), 
                #                         F.normalize(embed_P, p=2, dim=1), 
                #                         F.normalize(embed_N, p=2, dim=1)), dim=0)
                # labels = torch.cat((anchors_y, positives_y, negatives_y), dim=0)

                # print('embeddings.size(), labels.size()', embeddings.size(), labels.size())

                similarity_loss = triplet_loss(F.normalize(embed_A, p=2, dim=1), F.normalize(embed_P, p=2, dim=1), F.normalize(embed_N, p=2, dim=1))
                # similarity_loss = triplet_loss(embeddings.to(device), labels.to(device), device=device)
                running_similarity_loss += similarity_loss.item()

                print('similarity loss', similarity_loss.item())
                logger.info(f'==>>> similarity loss: {similarity_loss.item()}')
                similarity_loss.backward()
                optimizer.step()

            elif add_classifier:
                last_layer = nn.Linear(64, args.num_classes)
                snn_classifier = nn.Sequential(siamese_nn, fc_relu, last_layer)

                criterion = nn.CrossEntropyLoss().to(device)
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
                logger.info(f'==>>> classifier_loss: {classifier_loss.item()}')
                classifier_loss.backward()
                optimizer.step()

        similarity_loss_per_epoch = running_similarity_loss / args.iter_per_epoch
        similarity_loss_log.append(similarity_loss_per_epoch)
        logger.info(f'==>>> similarity loss: {similarity_loss}')
        scheduler.step()

    torch.save(snn_classifier, 'task3.pth')

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
    parser = argparse.ArgumentParser(description="Task 3")
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
    parser.add_argument('--train-batch', default=8, type=int,
                        help='train batchsize')
    parser.add_argument('--test-batch', default=32, type=int,
                        help='test batchsize')
    parser.add_argument('--total-iter', default=800*200, type=int,
                        help='total number of iterations to run')
    parser.add_argument('--iter-per-epoch', default=200, type=int,
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
    parser.add_argument("--modelpath", default="../trained_models/task1/c10/task1_c10_4k_t75/task1_c10_4k_t75.pth", 
                        type=str, help="Path to save model")
    parser.add_argument("--dropout", default=0.0, type=float, 
                        help="Dropout rate for model")   

    args = parser.parse_args()

    main(args)