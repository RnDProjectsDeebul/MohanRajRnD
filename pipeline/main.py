import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torch import nn
import neptune
import warnings
warnings.filterwarnings("ignore")

from train import train_model
from helpers import save_architecture_txt,get_model
from losses import edl_mse_loss
from data import import_data




data_dir = '../../data'
save_path = '../../results/'
parameters = { 'num_epochs':5,
                'num_classes':10,
                'batch_size': 128,
                'model_name':'LeNet',#'Resnet18',#"MobileNetV2"
                'loss_function':'Evidential',
                #'loss_function': 'Crossentropy',
                'lr': 0.1,
                'weight_decay':5e-4,
                'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                'dataset': "MNIST",
                #'dataset': "CIFAR10",
                'quantise': True}
logger = False

std_condition_name = str(parameters['loss_function'])+'_'+str(parameters['model_name'])
quant_condition_name = str(parameters['loss_function'])+'_'+str(parameters['model_name'])+'_quant'

# if parameters['quantise'] == True:
#     name = "Training" + "-" + str(parameters['model_name']) + "-" + str(parameters['loss_function']) + "-" + "Quant"
#     tags = [str(parameters['loss_function']),str(parameters['model_name']),str(parameters['dataset']),"Training", "Quant"]
name = "Training" + "-" + str(parameters['model_name']) + "-" + str(parameters['loss_function'])
tags = [str(parameters['loss_function']),str(parameters['model_name']),str(parameters['dataset']),"Training"]
    


model = get_model(parameters['model_name'],num_classes=parameters['num_classes'],weights='DEFAULT')
#save_architecture_txt(model=model,dir_path=save_path,filename=parameters['model_name'])
dataloaders, class_names = import_data(parameters['dataset'], data_dir, parameters['batch_size'], parameters['batch_size'])


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
    
# optimizer = torch.optim.SGD(model.parameters(), lr=parameters['lr'], momentum=0.9, weight_decay=parameters['weight_decay'])
# lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

 
if logger:
    run = neptune.init_run(
    #project="mohan20325145/CIFAR10",
    project="mohan20325145/MNIST",
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



train_model(model=model,
            num_epochs=parameters['num_epochs'],
            uncertainty = uncertainty,
            criterion=loss_function,
            optimizer=optimizer,
            scheduler=lr_scheduler,
            dataloaders=dataloaders,
            class_names = class_names ,
            logger=run,
            results_file_path =save_path,
            condition_name=std_condition_name,
            quantise=parameters['quantise'],
            std_path_to_save=save_path+str(std_condition_name)+'_model.pth',
            quant_path_to_save=save_path+str(quant_condition_name)+'_model.pth')



#save_architecture_txt(model=best_model,dir_path=save_path,filename=parameters['model_name']+"_quant")
#torch.save(best_model, save_path+str(condition_name)+'_model.pth')