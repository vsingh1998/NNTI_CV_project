import argparse
import math
from sys import prefix
from tokenize import PseudoExtras

from dataloader import get_cifar10, get_cifar100
from utils      import accuracy

from model.wrn  import WideResNet

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data   import DataLoader


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
                                args.num_classes, widen_factor=args.model_width)
    model       = model.to(device)

    ############################################################################
    # TODO: SUPPLY your code

    # define loss, opt, schd
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum, weight_decay=args.wd)
    
    # Pseudo dataset initialization
    pseudo_dataseṭ̣_x = torch.tensor([]).to(device)
    pseudo_dataseṭ̣_y = torch.tensor([]).long().to(device)
    
    ############################################################################
    
    for epoch in range(args.epoch):
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
            # TODO: SUPPLY your code
            ####################################################################
            # create training data
            X_train = torch.cat((x_l, pseudo_dataseṭ̣_x), dim=0)
            Y_train = torch.cat((y_l, pseudo_dataseṭ̣_y), dim=0)
            print(X_train.shape)

            # find output
            pred = model(X_train)
            loss = criterion(pred, Y_train)
            acc = accuracy(pred.data, Y_train, topk=(1,))[0]
            # print(acc)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()

            if i > 1:
                # make prediction on unlabeled data
                pred_ul = model(x_ul)
                # get class probabilities
                pred_prob = nn.functional.softmax(pred_ul, dim=1)
                
                # 
                pseudo_dataseṭ̣_x = torch.tensor([]).to(device)
                pseudo_dataseṭ̣_y = torch.tensor([]).long().to(device)

                #
                for idx_ul, pred in enumerate(pred_prob):
                    
                    max_prob, max_prob_class = torch.max(pred, dim=-1)
                    if max_prob > args.threshold:
                        # pseudo_label = torch.where(pred == max_prob, torch.tensor(1), torch.tensor(0))
                        # print(pseudo_label)

                        pseudo_dataseṭ̣_x = torch.cat((pseudo_dataseṭ̣_x, x_ul[idx_ul].unsqueeze(0)), dim=0)
                        pseudo_dataseṭ̣_y = torch.cat((pseudo_dataseṭ̣_y, max_prob_class.unsqueeze(0)), dim=0)





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
                        help='train batchsize')
    parser.add_argument('--total-iter', default=1024*512, type=int,
                        help='total number of iterations to run')
    parser.add_argument('--iter-per-epoch', default=1024, type=int,
                        help="Number of iterations to run per epoch")
    parser.add_argument('--num-workers', default=1, type=int,
                        help="Number of workers to launch during training")
    parser.add_argument('--threshold', type=float, default=0.95,
                        help='Confidence Threshold for pseudo labeling')
    parser.add_argument("--dataout", type=str, default="./path/to/output/",
                        help="Path to save log files")
    parser.add_argument("--model-depth", type=int, default=28,
                        help="model depth for wide resnet") 
    parser.add_argument("--model-width", type=int, default=2,
                        help="model width for wide resnet")
    
    # Add more arguments if you need them
    # Describe them in help
    # You can (and should) change the default values of the arguments
    
    args = parser.parse_args()

    main(args)