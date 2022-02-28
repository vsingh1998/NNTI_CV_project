import argparse
import math

from dataloader import get_cifar10, get_cifar100
from model.wrn import WideResNet
from utils import accuracy, create_triplet

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt


def main(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # generate dataloaders
    # labeled_loader, unlabeled_loader, test_loader = generate_dataloaders(args)

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
    
    model       = WideResNet(depth= 28, num_classes= args.num_classes, widen_factor=2)
    model       = model.to(device)

    # define loss, optimizer and lr scheduler
    criterion = nn.CrossEntropyLoss().to(device)
    triplet_loss = nn.TripletMarginLoss().to(device) # for snn
    optimizer = optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.1)

    # Pseudo dataset initialization
    pseudo_dataset_x = torch.tensor([]).to(device)
    pseudo_dataset_y = torch.tensor([]).long().to(device)

    args.epoch = math.ceil(args.total_iter / args.iter_per_epoch)
    supervised_epochs = 1
    loss_log = []
    train_acc_log = []

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

            # train on labeled data for specified epochs (T1 in paper)
            if epoch < supervised_epochs:
                pred = model.forward(x_l, mode= 'classifier')
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
                pseudo_elements = torch.numel(pseudo_dataset_x)

                # labeled + pseudo combined training data
                X_train = torch.cat((x_l, pseudo_dataset_x), dim=0)
                Y_train = torch.cat((y_l, pseudo_dataset_y), dim=0)

                # train model on combined data
                model.train()
                pred = model.forward(X_train, mode= 'classifier')

                if pseudo_elements == 0:
                    total_loss = criterion(pred, Y_train)
                else:
                    main_loss   = criterion(pred[:-pseudo_dataset_x.shape[0]], Y_train[:-pseudo_dataset_x.shape[0]])
                    pseudo_loss = criterion(pred[-pseudo_dataset_x.shape[0]:], Y_train[-pseudo_dataset_x.shape[0]:])
                    total_loss  = main_loss + pseudo_loss

                total_loss = criterion(pred, Y_train)
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
                pred_ul = model.forward(x_ul, mode= 'classifier')
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

                pseudo_elements = torch.numel(pseudo_dataset_x)
                X_train = torch.cat((x_l, pseudo_dataset_x), dim=0)
                Y_train = torch.cat((y_l, pseudo_dataset_y), dim=0)

                if pseudo_elements != 0:
                    anchors, positives, negatives = create_triplet(X_train, Y_train)
                    embed_A = model.forward(anchors.to(device), mode = 'embedding')
                    embed_P = model.forward(positives.to(device), mode = 'embedding')
                    embed_N = model.forward(negatives.to(device), mode = 'embedding')

                    similarity_loss = triplet_loss(embed_A, embed_P, embed_N)
                    print('similarity_loss', similarity_loss.item())
                    similarity_loss.backward()
                    optimizer.step()

        
        acc_per_epoch = running_train_acc / args.iter_per_epoch
        train_acc_log.append(acc_per_epoch)
        loss_per_epoch = running_loss / args.iter_per_epoch
        loss_log.append(loss_per_epoch)
        print('Epoch: ', epoch, 'Loss: ', loss_per_epoch, 'Accuracy: ', acc_per_epoch)
        running_loss, running_train_acc = 0.0, 0.0

    # print('========OG=========', model)

    # x = torch.ones(1,3,32,32)
    # model_classifier = model.forward(x, mode = 'classifier')
    # print('=======CLASSIFIER========', list(model_classifier.modules()))

    # model_embed = model.forward(x, mode = 'embedding')
    # print('=======EMBED=======', list(model_embed.modules()))

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
    parser.add_argument('--total-iter', default=100*100, type=int,
                        help='total number of iterations to run')
    parser.add_argument('--iter-per-epoch', default=100, type=int,
                        help="Number of iterations to run per epoch")
    parser.add_argument('--num-workers', default=8, type=int,
                        help="Number of workers to launch during training")
    parser.add_argument('--threshold', type=float, default=0.95,
                        help='Confidence Threshold for pseudo labeling')
    parser.add_argument("--model-depth", type=int, default=28,
                        help="model depth for wide resnet") 
    parser.add_argument("--model-width", type=int, default=2,
                        help="model width for wide resnet")
    parser.add_argument('--milestones', action='append', type=int, default=[40, 80], 
                        help="Milestones for the LR scheduler")# see if useful, else rm
    parser.add_argument("--modelpath", default="./model/task3.pth", 
                        type=str, help="Path to save model")
    parser.add_argument("--dropout", default=0.0, type=float, 
                        help="Dropout rate for model")   

    args = parser.parse_args()

    main(args)