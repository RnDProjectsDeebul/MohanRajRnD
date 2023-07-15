import pathlib
import random

import torch
import torch.utils.data
from torch.nn import functional as F
from torch.utils.tensorboard.writer import SummaryWriter

from ignite.engine import Events, Engine
from ignite.metrics import Accuracy, Average, Loss
from ignite.contrib.handlers import ProgressBar

from utils.lenet_duq import CNN_DUQ
from utils.datasets import all_datasets

from data import import_data
import copy
import pandas as pd
import torchvision
from torch.quantization import quantize_fx
import torchvision.transforms as transforms
import warnings
warnings.filterwarnings("ignore")


epochs = 7
l_gradient_penalty = 0.05
length_scale = 0.05
batch_size = 128
learning_rate = 0.05
gamma = 0.999
weight_decay = 1e-4
num_classes = 10
milestones = [25, 50, 75]
embedding_size = 256
learnable_length_scale = False  



def main():
    pathlib.Path("../../results/" + "duq_writer").mkdir(exist_ok=True)
    writer = SummaryWriter(log_dir=f"../../results/duq_writer")
    loss_acc_dict = {"epoch_no": [],"train_loss": [],"val_loss": [],"train_acc": [],"val_acc": []}
    
    ds = all_datasets["MNIST"]()
    input_size, num_classes, dataset, test_dataset = ds
    train_dataset = dataset
    val_dataset = test_dataset

    model = CNN_DUQ(num_classes,embedding_size,learnable_length_scale,length_scale,gamma,)
    model = model.cuda()
    
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.2)
    

    def calc_gradient_penalty(x, y_pred_sum):
        gradients = torch.autograd.grad(
            outputs=y_pred_sum,
            inputs=x,
            grad_outputs=torch.ones_like(y_pred_sum),
            create_graph=True,
            retain_graph=True,
        )[0]
        gradients = gradients.flatten(start_dim=1)
        grad_norm = gradients.norm(2, dim=1)
        gradient_penalty = ((grad_norm - 1) ** 2).mean()
        return gradient_penalty

    
    def step(engine, batch):
        model.train()
        optimizer.zero_grad()
        x, y = batch
        x, y = x.cuda(), y.cuda()
        lab = copy.deepcopy(y)
        x.requires_grad_(True)
        y_pred = model(x)
        y = F.one_hot(y, num_classes).float()
        loss = F.binary_cross_entropy(y_pred, y)
        
        if l_gradient_penalty > 0:
            gp = calc_gradient_penalty(x, y_pred.sum(1))
            loss += l_gradient_penalty * gp
            
        loss.backward()
        optimizer.step()
        x.requires_grad_(False)
        
        with torch.no_grad():
            model.eval()
            model.update_embeddings(x, y)
        return {"loss": loss.item(), "y_": lab, "y_pred_": y_pred}

    
    
    def eval_step(engine, batch):
        model.eval()
        x, y = batch
        x, y = x.cuda(), y.cuda()        
        x.requires_grad_(True)
        y_pred = model(x)
        return {"x": x, "y": y, "y_pred": y_pred}

    
    trainer = Engine(step)
    evaluator = Engine(eval_step)

    metric = Average(output_transform=lambda out: (out["loss"]))
    metric.attach(trainer, "loss")

    metric = Accuracy(output_transform=lambda out: (out["y_pred_"], out["y_"]))
    metric.attach(trainer, "train_accuracy")
    
    metric = Accuracy(output_transform=lambda out: (out["y_pred"], out["y"]))
    metric.attach(evaluator, "accuracy")

    def bce_output_transform(out):
        return (out["y_pred"], F.one_hot(out["y"], num_classes).float())

    metric = Loss(F.binary_cross_entropy, output_transform=bce_output_transform)
    metric.attach(evaluator, "bce")
    
    metric = Loss(calc_gradient_penalty, output_transform=lambda out: (out["x"], out["y_pred"]))
    metric.attach(evaluator, "gradient_penalty")

    pbar = ProgressBar(dynamic_ncols=True)
    pbar.attach(trainer)


    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)


    
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_results(trainer):
        # Training metrics
        metrics = trainer.state.metrics
        loss = metrics["loss"]
        acc = metrics["train_accuracy"]
        print((
                f"Train - Epoch: {trainer.state.epoch} "
                f"Loss: {loss:.2f}"
                f"Acc: {acc:.4f} "
              ))
        writer.add_scalar("Loss/train", loss, trainer.state.epoch)
        writer.add_scalar("Accuracy/train", acc, trainer.state.epoch)
        loss_acc_dict['epoch_no'].append(trainer.state.epoch)
        loss_acc_dict['train_loss'].append(loss)
        loss_acc_dict['train_acc'].append(acc)


        # Validation metrics
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        acc = metrics["accuracy"]
        bce = metrics["bce"]
        GP = metrics["gradient_penalty"]
        loss = bce + l_gradient_penalty * GP
        print((
                f"Valid - Epoch: {trainer.state.epoch} "
                f"Loss: {loss:.2f} "
                f"Acc: {acc:.4f} "
            ))
        writer.add_scalar("Loss/valid", loss, trainer.state.epoch)
        writer.add_scalar("Accuracy/valid", acc, trainer.state.epoch)
        loss_acc_dict['val_loss'].append(loss)
        loss_acc_dict['val_acc'].append(acc)
        
        scheduler.step()
        
    trainer.run(train_loader, max_epochs=epochs)
    evaluator.run(test_loader)


    torch.save(model.state_dict(), f"../../results/DUQ_LeNet_DUQ_model.pth")
    writer.close()
    
    condition_name = '../results/'+'DUQ'+'_'+'LeNet_DUQ'+'_loss_acc.csv'
    loss_acc_df = pd.DataFrame(loss_acc_dict, columns=["epoch_no","train_loss","val_loss","train_acc","val_acc"])    
    loss_acc_df.to_csv(path_or_buf=condition_name)






if __name__ == "__main__":
    main()
    
       
    ###############
    #Quantise
    ###############
    model = CNN_DUQ(num_classes,embedding_size,learnable_length_scale,length_scale,gamma,)
    model_path = '../../results/DUQ_LeNet_DUQ_model.pth'
    model.load_state_dict(torch.load(model_path)) 
    dataloader, class_names = import_data("MNIST", '../../data', 128, 128)


    dataiter = iter(dataloader['train'])
    img, lab = next(dataiter)
    m = copy.deepcopy(model)
    m.to("cpu")
    m.eval()
    qconfig_dict = {"": torch.quantization.get_default_qconfig("fbgemm")}
    model_prepared = quantize_fx.prepare_fx(m, qconfig_dict, img)
    with torch.inference_mode():
        for _ in range(10):
            img, lab = next(dataiter)
            model_prepared(img)
    q_model = quantize_fx.convert_fx(model_prepared)
    q_model(img)

    q_model_path = '../../results/DUQ_LeNet_DUQ_quant_model.pth'
    torch.save(q_model.state_dict(), q_model_path)    
    