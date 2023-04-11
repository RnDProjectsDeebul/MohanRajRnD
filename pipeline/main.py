import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torch import nn
import neptune.new as neptune

from train import train_model
from helpers import save_architecture_txt,get_model
from losses import edl_mse_loss

import warnings
warnings.filterwarnings("ignore")

data_dir = '../../data'
save_path = '../../results/'
parameters = { 'num_epochs':15,
                'num_classes':10,
                'batch_size': 128,
                'model_name':'Resnet18',
                #'loss_function':'Evidential',
                'loss_function': 'Crossentropy',
                'lr': 0.1,
                'weight_decay':5e-4,
                'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                'quantise':False}
logger = True

if parameters['quantise'] == True:
    condition_name = str(parameters['loss_function'])+'_'+str(parameters['model_name'])+'_quant'
    name = "Training" + "-" + str(parameters['model_name']) + "-" + str(parameters['loss_function']) + "-" + "Quant"
    tags = [str(parameters['loss_function']),str(parameters['model_name']),"CIFAR10","Training", "Quant"]
else:
    condition_name = str(parameters['loss_function'])+'_'+str(parameters['model_name'])
    name = "Training" + "-" + str(parameters['model_name']) + "-" + str(parameters['loss_function'])
    tags = [str(parameters['loss_function']),str(parameters['model_name']),"CIFAR10","Training"]
    

model = get_model(parameters['model_name'],num_classes=parameters['num_classes'],weights='DEFAULT')
#save_architecture_txt(model=model,dir_path=save_path,filename=parameters['model_name'])


uncertainty = False
if parameters['loss_function'] == 'Crossentropy':
    loss_function = nn.CrossEntropyLoss()
elif parameters['loss_function'] == 'Evidential':
    loss_function = edl_mse_loss
    uncertainty = True
else:
    raise NotImplementedError


#optimizer = torch.optim.SGD(model.parameters(),lr=parameters['lr'], momentum=0.9)
#lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# optimizer = torch.optim.Adam(model.parameters(),lr=parameters['lr'], weight_decay=parameters['weight_decay'])
# lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
optimizer = torch.optim.SGD(model.parameters(), lr=parameters['lr'], momentum=0.9, weight_decay=parameters['weight_decay'])
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

 
if logger:
    run = neptune.init_run(
    project="mohan20325145/CIFAR10",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJhZWQyMTU4OC02NmU4LTRiNjgtYWE5Zi1lNDg5MjdmZGJhNzYifQ==",
    tags = tags,
    name= name,
    )
    run['config/hyperparameters'] = parameters
    run['config/model'] = type(model).__name__
    run['config/criterion'] = parameters['loss_function']
    run['config/optimizer'] = type(optimizer).__name__
else:
    run = None


def load_data(data_dir):
    #transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), transforms.RandomErasing()])
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
    transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)   
    return trainset, testset


trainset, testset = load_data(data_dir)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=parameters['batch_size'], shuffle=True, num_workers=2)
valloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
dataloaders = {"train": trainloader, "val": valloader}
class_names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


training_results = train_model(model=model,
                                num_epochs=parameters['num_epochs'],
                                uncertainty = uncertainty,
                                criterion=loss_function,
                                optimizer=optimizer,
                                scheduler=lr_scheduler,
                                dataloaders=dataloaders,
                                class_names = class_names ,
                                logger=run,
                                results_file_path =save_path,
                                condition_name=condition_name,
                                quantise=parameters['quantise'])


best_model = training_results
#save_architecture_txt(model=best_model,dir_path=save_path,filename=parameters['model_name']+"_quant")
torch.save(best_model.state_dict(), save_path+str(condition_name)+'_model.pth')
#torch.save(best_model, save_path+str(condition_name)+'_model.pth')