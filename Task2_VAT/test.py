import torch
from dataloader import get_cifar10, get_cifar100
from model.wrn import WideResNet
from utils import accuracy

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
    model = torch.load(filepath)
    model.eval()

    inputs, labels = testdataset

    preds = model(inputs)
    logits = torch.nn.functional.one_hot(preds, num_classes=10)

    return torch.nn.functional.softmax(logits, dim=1)

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
    model = torch.load(filepath)
    model.eval()

    inputs, labels = testdataset

    preds = model(inputs)
    logits = torch.nn.functional.one_hot(preds, num_classes=100)

    return torch.nn.functional.softmax(logits, dim=1)