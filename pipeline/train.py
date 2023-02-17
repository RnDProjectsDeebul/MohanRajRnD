import torch
from helpers import get_device, one_hot_embedding
from torch.autograd import Variable
from torchmetrics import Accuracy
from torch.quantization import quantize_fx
import pandas as pd
import copy
import torchvision
import warnings
warnings.filterwarnings("ignore")


def train_model(model=None,
                num_epochs=None,
                uncertainty = None,
                criterion=None,
                optimizer=None,
                scheduler=None,
                dataloaders=None,
                class_names=None,
                logger=None,
                results_file_path = None,
                condition_name=None,
                quantise=False):


    num_classes = len(class_names)
    device = get_device()
    if torch.cuda.is_available():
        model.to(device)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0


    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  
            else:
                model.eval()  

            running_loss = 0.0
            running_corrects = 0

            for data in dataloaders[phase]:
                inputs, labels = data

                if torch.cuda.is_available():
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    if uncertainty:
                        y = one_hot_embedding(labels=labels,num_classes=num_classes)
                        y.to(device=device)
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, y.float(), epoch, num_classes, 3, device)
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        
                    _, preds = torch.max(outputs, 1)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item()* inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
            if phase == 'train':
                scheduler.step()
                print ("LR :", scheduler.get_lr())


            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            if phase == "train" and logger!= None:
                logger['plots/training/train/loss'].log(epoch_loss)
                logger['plots/training/train/accuracy'].log(epoch_acc.item())
            elif logger!= None:
                logger['plots/training/val/loss'].log(epoch_loss)
                logger['plots/training/val/accuracy'].log(epoch_acc.item())

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                
    
    model.load_state_dict(best_model_wts)
    
    if quantise:
        dataiter = iter(dataloaders['train'])
        images, labels = next(dataiter)
        
        m = copy.deepcopy(model)
        m.to("cpu")
        m.eval()
        qconfig_dict = {"": torch.quantization.get_default_qconfig("fbgemm")}
        model_prepared = quantize_fx.prepare_fx(m, qconfig_dict, images)
        
        with torch.inference_mode():
            for _ in range(10):
                images, labels = next(dataiter)
                model_prepared(images)
        model_quantized = quantize_fx.convert_fx(model_prepared)
        return model_quantized
        
        
    return model