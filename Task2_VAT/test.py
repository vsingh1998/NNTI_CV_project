import torch
from dataloader import get_cifar10, get_cifar100
from model.wrn import WideResNet
from utils import accuracy
from torch.utils.data   import DataLoader

def test_cifar10(testdataset, filepath = "./path/to/model.pth.tar"):
    '''
    args: 
        testdataset : (torch.utils.data.Dataset)
        filepath    : (str) The path to the model file that is saved
    returns : (torch.Tensor) logits of the testdataset with shape 
                [num_samples, 10]. Apply softmax to the logits
    
    Description:
        This function loads the model given in the filepath and returns the 
        logits of the testdataset which is a torch.utils.data.Dataset. You can
        save the arguments needed to load the models (e.g. width, depth etc) 
        with the model file. Assume testdataset is like CIFAR-10. Test this
        function with the testdataset returned by get_cifar10()
    '''
    # define dataloader for test dataset
    test_loader  = DataLoader(testdataset, batch_size = 64,
                                shuffle = False, num_workers=8)

    # check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # define model (we forgot to save parameters)
    model = WideResNet(depth=16, num_classes=10, widen_factor=8)

    # load saved model
    model.load_state_dict(torch.load(filepath, map_location=device))

    # model in evaluation mode
    model.eval()

    predictions = torch.tensor([])
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            # load batch
            inputs, labels = inputs.to(device), labels.to(device)

            # generate predictions from the model
            pred = model(inputs)

            # concatenate predictions
            predictions = torch.cat((predictions, pred), dim=0)

    return torch.nn.functional.softmax(predictions, dim=1)

def test_cifar100(testdataset, filepath="./path/to/model.pth.tar"):
    '''
    args: 
        testdataset : (torch.utils.data.Dataset)
        filepath    : (str) The path to the model file that is saved
    returns : (torch.Tensor) logits of the testdataset with shape 
                [num_samples, 100]. Apply softmax to the logits
    
    Description:
        This function loads the model given in the filepath and returns the 
        logits of the testdataset which is a torch.utils.data.Dataset. You can
        save the arguments needed to load the models (e.g. width, depth etc) 
        with the model file. Assume testdataset is like CIFAR-100. Test this
        function with the testdataset returned by get_cifar100()
    '''

    # define dataloader for test dataset
    test_loader  = DataLoader(testdataset, batch_size = 64,
                                shuffle = False, num_workers=8)

    # check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # define model (we forgot to save parameters)
    model = WideResNet(depth=16, num_classes=100, widen_factor=8)

    # load saved model
    model.load_state_dict(torch.load(filepath, map_location=device))

    # model in evaluation mode
    model.eval()

    predictions = torch.tensor([])
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            # load batch
            inputs, labels = inputs.to(device), labels.to(device)

            # generate predictions from the model
            pred = model(inputs)

            # concatenate predictions
            predictions = torch.cat((predictions, pred), dim=0)

    return torch.nn.functional.softmax(predictions, dim=1)
    