import torch
import torch.nn.functional as F
from helpers import get_device, one_hot_embedding
from losses import relu_evidence
from torch.autograd import Variable
from torchmetrics import Accuracy
from torch.quantization import quantize_fx
import pandas as pd
import copy
import warnings
warnings.filterwarnings("ignore")

def calc_gradients_input(x, y_pred):
        gradients = torch.autograd.grad(outputs=y_pred, inputs=x, grad_outputs=torch.ones_like(y_pred), create_graph=True,)[0]
        gradients = gradients.flatten(start_dim=1)
        return gradients

def calc_gradient_penalty(x, y_pred):
        gradients = calc_gradients_input(x, y_pred)
        grad_norm = gradients.norm(2, dim=1)
        gradient_penalty = ((grad_norm - 1) ** 2).mean()
        return gradient_penalty



def train_model(model=None,
                num_epochs=None,
                uncertainty = None,
                loss_name=None,
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

    
    
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        for phase in ["train", "val"]:
            if phase == "train":
                print("Training...")
                model.train() 
            else:
                print("Validating...")
                model.eval()  

            running_loss = 0.0
            running_corrects = 0.0
            correct = 0

            for i, (inputs, labels) in enumerate(dataloaders[phase]):

                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()


                with torch.set_grad_enabled(phase == "train"):

                    if uncertainty:
                        if loss_name[0:10] == "Evidential":
                            y = one_hot_embedding(labels, num_classes)
                            y = y.to(device)
                            outputs = model(inputs)
                            _, preds = torch.max(outputs, 1)
                            loss = criterion(outputs, y.float(), epoch, num_classes, 10, device)

                            match = torch.reshape(torch.eq(preds, labels).float(), (-1, 1))
                            acc = torch.mean(match)
                            evidence = relu_evidence(outputs)
                            alpha = evidence + 1
                            u = num_classes / torch.sum(alpha, dim=1, keepdim=True)
                            total_evidence = torch.sum(evidence, 1, keepdim=True)
                            mean_evidence = torch.mean(total_evidence)
                            mean_evidence_succ = torch.sum(torch.sum(evidence, 1, keepdim=True) * match
                            ) / torch.sum(match + 1e-20)
                            mean_evidence_fail = torch.sum(torch.sum(evidence, 1, keepdim=True) * (1 - match)
                            ) / (torch.sum(torch.abs(1 - match)) + 1e-20)
                            
                        elif loss_name == "DUQ":         
                            inputs.requires_grad_(True)
                            y_pred = model(inputs)
                            _, preds = torch.max(y_pred, 1)
                            y = F.one_hot(labels, num_classes).float()
                            loss = F.binary_cross_entropy(y_pred, y, reduction="mean")
                            #gp = calc_gradient_penalty(inputs, y_pred)
                            #loss += 0.5 * gp
                            if phase == "train":
                                loss.backward()
                                optimizer.step()
                                inputs.requires_grad_(False)
                                with torch.no_grad():
                                    model.eval()
                                    model.update_embeddings(inputs, y)
                                
                    else:
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        
                    if phase == "train" and loss_name != "DUQ":
                        loss.backward()
                        optimizer.step()


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
        