import torch
import torchvision
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_mnist(data_dir, train_batch_size, val_batch_size):
    data_train = MNIST(data_dir,
                       download=True,
                       train=True,
                       transform=transforms.Compose([transforms.ToTensor()]))
    data_val = MNIST(data_dir,
                     train=False,
                     download=True,
                     transform=transforms.Compose([transforms.ToTensor()]))
    
    dataloader_train = DataLoader(data_train, batch_size=train_batch_size, shuffle=True, num_workers=8)
    dataloader_val   = DataLoader(data_val, batch_size=val_batch_size, num_workers=8)
    
    dataloaders = {"train": dataloader_train, "val": dataloader_val}
    class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    return dataloaders, class_names




def get_cifar10(data_dir, train_batch_size, val_batch_size):
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
    transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test) 
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True, num_workers=2)
    valloader = torch.utils.data.DataLoader(testset, batch_size=val_batch_size, shuffle=False, num_workers=2)
    
    dataloaders = {"train": trainloader, "val": valloader}
    class_names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    return dataloaders, class_names





def import_data(dataset_name, data_dir, train_batch_size, val_batch_size):
    if dataset_name == "CIFAR10":
        dataloaders, class_names = get_cifar10(data_dir, train_batch_size, val_batch_size)
    elif dataset_name == "MNIST":
        dataloaders, class_names = get_mnist(data_dir, train_batch_size, val_batch_size)
    return dataloaders, class_names