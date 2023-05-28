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
from losses import edl_mse_loss, edl_log_loss, edl_digamma_loss
from data import import_data




data_dir = '../../data'
save_path = '../results/'
models_path = '../../results/'
parameters = { 'num_epochs':3,
                'num_classes':10,
                'batch_size': 128,
                #'model_name':'LeNet',
                #'model_name':'Resnet18',
                'model_name':'ResNet_DUQ',
                #'loss_function': 'Crossentropy',
                #'loss_function':'Evidential_MSE',
                #'loss_function':'Evidential_LOG',
                #'loss_function':'Evidential_DIGAMMA',
                'loss_function': 'DUQ',
                'lr': 0.1,
                'weight_decay':5e-4,
                'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                #'dataset': "MNIST",
                'dataset': "CIFAR10",
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
elif parameters['loss_function'] == 'Evidential_MSE':
    loss_function = edl_mse_loss
    uncertainty = True
elif parameters['loss_function'] == 'Evidential_LOG':
    loss_function = edl_log_loss
    uncertainty = True
elif parameters['loss_function'] == 'Evidential_DIGAMMA':
    loss_function = edl_digamma_loss
    uncertainty = True
elif parameters['loss_function'] == 'DUQ':
    loss_function = None
    uncertainty = True
else:
    raise NotImplementedError

    

if parameters['loss_function'] == 'DUQ':
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 50, 75], gamma=0.2)
else:
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
            loss_name = parameters['loss_function'],
            criterion=loss_function,
            optimizer=optimizer,
            scheduler=lr_scheduler,
            dataloaders=dataloaders,
            class_names = class_names ,
            logger=run,
            results_file_path =save_path,
            condition_name=save_path+str(std_condition_name)+'_loss_acc.csv',
            quantise=parameters['quantise'],
            std_path_to_save=models_path+str(std_condition_name)+'_model.pth',
            quant_path_to_save=models_path+str(quant_condition_name)+'_model.pth')



#save_architecture_txt(model=best_model,dir_path=save_path,filename=parameters['model_name']+"_quant")
#torch.save(best_model, save_path+str(condition_name)+'_model.pth')