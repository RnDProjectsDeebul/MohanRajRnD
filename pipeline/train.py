import torch
from helpers import get_device, one_hot_embedding
from losses import relu_evidence
from torch.autograd import Variable
from torchmetrics import Accuracy
from torch.quantization import quantize_fx
import pandas as pd
import copy
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
                quantise=False,
                std_path_to_save='./',
                quant_path_to_save='./'):


    num_classes = len(class_names)
    device = get_device()
    if torch.cuda.is_available():
        model.to(device)

    loss_acc_dict = {"epoch_no": [],"train_loss": [],"val_loss": [],"train_acc": [],"val_acc": []}
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0


    # for epoch in range(num_epochs):
    #     print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    #     print('-' * 10)

    #     for phase in ['train', 'val']:

    #         running_loss = 0
    #         running_corrects = 0
    #         total = 0

    #         if phase == 'train':
    #             model.train()
    #             for data in dataloaders['train']:
    #                 inputs, labels = data
    #                 if torch.cuda.is_available():
    #                     inputs = Variable(inputs.cuda())
    #                     labels = Variable(labels.cuda())
    #                 else:
    #                     inputs, labels = Variable(inputs), Variable(labels)
    #                 optimizer.zero_grad()
    #                 outputs = model(inputs)
    #                 if uncertainty:
    #                     y = one_hot_embedding(labels=labels,num_classes=num_classes)
    #                     y.to(device=device)
    #                     loss = criterion(outputs, y.float(), epoch, num_classes, 3, device)
    #                 else:
    #                     loss = criterion(outputs, labels) 
    #                 loss.backward()
    #                 optimizer.step()
    #                 _, preds = outputs.max(1)
                
    #         elif phase == 'val':
    #             model.eval() 
    #             with torch.no_grad():
    #                 for data in dataloaders['val']:
    #                     inputs, labels = data
    #                     if torch.cuda.is_available():
    #                         inputs = Variable(inputs.cuda())
    #                         labels = Variable(labels.cuda())
    #                     else:
    #                         inputs, labels = Variable(inputs), Variable(labels)
    #                     outputs = model(inputs)
    #                     if uncertainty:
    #                         y = one_hot_embedding(labels=labels,num_classes=num_classes)
    #                         y.to(device=device)
    #                         loss = criterion(outputs, y.float(), epoch, num_classes, 3, device)
    #                     else:
    #                         loss = criterion(outputs, labels) 
    #                     _, preds = outputs.max(1)
                
    #         running_loss += loss.item()
    #         running_corrects += preds.eq(labels).sum().item()
    #         total += labels.size(0)
    #         epoch_loss = running_loss / (epoch+1)
    #         epoch_acc = 100.*running_corrects / total
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                print("Training...")
                model.train()  # Set model to training mode
            else:
                print("Validating...")
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0
            correct = 0

            # Iterate over data.
            for i, (inputs, labels) in enumerate(dataloaders[phase]):

                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):

                    if uncertainty:
                        y = one_hot_embedding(labels, num_classes)
                        y = y.to(device)
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(
                            outputs, y.float(), epoch, num_classes, 10, device
                        )

                        match = torch.reshape(torch.eq(preds, labels).float(), (-1, 1))
                        acc = torch.mean(match)
                        evidence = relu_evidence(outputs)
                        alpha = evidence + 1
                        u = num_classes / torch.sum(alpha, dim=1, keepdim=True)

                        total_evidence = torch.sum(evidence, 1, keepdim=True)
                        mean_evidence = torch.mean(total_evidence)
                        mean_evidence_succ = torch.sum(
                            torch.sum(evidence, 1, keepdim=True) * match
                        ) / torch.sum(match + 1e-20)
                        mean_evidence_fail = torch.sum(
                            torch.sum(evidence, 1, keepdim=True) * (1 - match)
                        ) / (torch.sum(torch.abs(1 - match)) + 1e-20)

                    else:
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if scheduler is not None:
                if phase == "train":
                    scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            if phase == "train":
                loss_acc_dict['train_loss'].append(epoch_loss)
                loss_acc_dict['train_acc'].append(epoch_acc.cpu().detach().numpy())
                if logger!= None:
                    logger['plots/training/train/loss'].log(epoch_loss)
                    logger['plots/training/train/accuracy'].log(epoch_acc)
            elif phase == "val":
                loss_acc_dict['val_loss'].append(epoch_loss)
                loss_acc_dict['val_acc'].append(epoch_acc.cpu().detach().numpy())
                if logger!=None:
                    logger['plots/training/val/loss'].log(epoch_loss)
                    logger['plots/training/val/accuracy'].log(epoch_acc)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_model_wts = copy.deepcopy(model.state_dict())
                best_acc = epoch_acc

        loss_acc_dict['epoch_no'].append(epoch+1)
        # scheduler.step()
        # print ("LR :", scheduler.get_lr())
                
    loss_acc_df = pd.DataFrame(loss_acc_dict, columns=["epoch_no","train_loss","val_loss","train_acc","val_acc"])
    loss_acc_df.to_csv(path_or_buf=condition_name)
    
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), std_path_to_save)
    
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
        torch.save(model_quantized.state_dict(), quant_path_to_save)
        