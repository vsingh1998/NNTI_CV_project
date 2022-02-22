import argparse
import math
from sys import prefix
from tokenize import PseudoExtras

from dataloader import get_cifar10, get_cifar100
from utils      import accuracy, L2_norm

from model.wrn  import WideResNet

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data   import DataLoader

import matplotlib.pyplot as plt

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

    test_loader         = DataLoader(test_dataset,
                                    batch_size = args.test_batch,
                                    shuffle = False, 
                                    num_workers=args.num_workers)

    model       = WideResNet(args.model_depth, 
                                args.num_classes, widen_factor=args.model_width,
                                dropRate= args.dropout)
    model       = model.to(device)

    # plot loss per epoch
    with open('../trained_models/task2_cifar10_1e6_8_160k_800_16_8__0.3/loss_log.txt') as f:
        lines = f.readlines()
        loss_log = [float(line.split()[0]) for line in lines]
        
    plt.figure(0)
    plt.plot(loss_log)
    plt.title('Loss per epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()
    plt.savefig('loss.png')

    acc_log = []
    # plot accuracy per epoch
    acc_log = []
    with open('../trained_models/task2_cifar10_1e6_8_160k_800_16_8__0.3/acc_log.txt') as f:
        lines = f.readlines()
        for line in lines:
            acc = line.split()[0]
            acc = float(acc[acc.find("[")+1:acc.rfind("]")])
            acc_log.append(acc)
<<<<<<< HEAD

=======
    
    plt.figure(1)
>>>>>>> 5c31d446df82844fa3b16fc07a3cd0da6181e10d
    plt.plot(acc_log)
    plt.title('Accuracy per epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.savefig('acc.png')

    ### Test
    model.load_state_dict(torch.load(args.modelpath, map_location=device))
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
        acc_log.append(test_accuracy) # check if need
        running_acc = 0.0


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
    parser.add_argument('--total-iter', default=1024*512, type=int,
                        help='total number of iterations to run')
    parser.add_argument('--iter-per-epoch', default=1024, type=int,
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
    parser.add_argument("--vat-eps", default=8.0, type=float, 
                        help="VAT epsilon parameter") 
    parser.add_argument("--vat-iter", default=1, type=int, 
                        help="VAT iteration parameter")

    # added arguments
    parser.add_argument('--milestones', action='append', type=int, default=[], 
                        help="Milestones for the LR scheduler")# see if useful, else rm
    parser.add_argument("--modelpath", default="../trained_models/task2_cifar10_1e6_8_160k_800_16_8__0.3/task2_cifar10_1e6_8_160k_800_16_8__0.3.pth", 
                        type=str, help="Path to save model")
    parser.add_argument("--dropout", default=0.3, type=float, 
                        help="Dropout rate for model") 
    # Add more arguments if you need them
    # Describe them in help
    # You can (and should) change the default values of the arguments
    
    args = parser.parse_args()

    main(args)