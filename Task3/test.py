import argparse

from dataloader import get_cifar10, get_cifar100
from utils import accuracy
from model.wrn import WideResNet

import torch
from torch.utils.data import DataLoader

def main(args):
    if args.dataset == "cifar10":
        args.num_classes = 10
        _, _, test_dataset = get_cifar10(args, args.datapath)

    if args.dataset == "cifar100":
        args.num_classes = 100
        _, _, test_dataset = get_cifar100(args, args.datapath)

    # args.epoch = math.ceil(args.total_iter / args.iter_per_epoch)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_loader  = DataLoader(test_dataset, batch_size = args.test_batch,
                                shuffle = True, num_workers=args.num_workers)
    
    # Test
    snn_classifier = torch.load(args.modelpath, map_location=device)
    snn_classifier.eval()

    running_acc = 0.0

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            pred = snn_classifier(inputs)
            acc = accuracy(pred.data, labels, topk=(1,))[0]
            running_acc += acc

        test_accuracy = running_acc.item() / batch_idx
        print('Accuracy: ', test_accuracy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 3 inference")
    parser.add_argument("--dataset", default="cifar10", 
                        type=str, choices=["cifar10", "cifar100"])
    parser.add_argument("--datapath", default="./data/", 
                        type=str, help="Path to the CIFAR-10/100 dataset")
    parser.add_argument('--num-labeled', type=int, 
                        default=4000, help='Total number of labeled samples')
    parser.add_argument("--expand-labels", action="store_true", 
                        help="expand labels to fit eval steps")
    parser.add_argument('--train-batch', default=4, type=int,
                        help='train batchsize')
    parser.add_argument('--test-batch', default=64, type=int,
                        help='train batchsize')
    parser.add_argument('--total-iter', default=800*800, type=int,
                        help='total number of iterations to run')
    parser.add_argument('--iter-per-epoch', default=800, type=int,
                        help="Number of iterations to run per epoch")
    parser.add_argument('--num-workers', default=8, type=int,
                        help="Number of workers to launch during training")                        
    parser.add_argument("--modelpath", default="../trained_models/task3/c10/task3_c10_4k/task3_c10_4k.pth", 
                        type=str, help="Path to load trained model from")
    
    args = parser.parse_args()

    main(args)