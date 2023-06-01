import random
import numpy as np

import torch
import torch.utils.data
from torch.nn import functional as F

from ignite.engine import Events, Engine
from ignite.metrics import Accuracy, Loss
from ignite.contrib.handlers.tqdm_logger import ProgressBar

from utils.evaluate_ood import (get_fashionmnist_mnist_ood,get_fashionmnist_notmnist_ood,)
from utils.datasets import get_FastFashionMNIST
from utils.cnn_duq import CNN_DUQ



def train_model(l_gradient_penalty, length_scale, final_model):
    loss_acc_dict = {"epoch_no": [],"train_loss": [],"val_loss": [],"train_acc": [],"val_acc": []}
    _, _, dataset, test_dataset = get_FastFashionMNIST()
    
    idx = list(range(60000))
    random.shuffle(idx)
    if final_model:
        train_dataset = dataset
        val_dataset = test_dataset
    else:
        train_dataset = torch.utils.data.Subset(dataset, indices=idx[:55000])
        val_dataset = torch.utils.data.Subset(dataset, indices=idx[55000:])

        
    num_classes = 10
    embedding_size = 256
    learnable_length_scale = False
    gamma = 0.999

    model = CNN_DUQ(num_classes,embedding_size,learnable_length_scale,length_scale,gamma,)
    model = model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4)

    
    def output_transform_bce(output):
        y_pred, y, _, _ = output
        return y_pred, y

    def output_transform_acc(output):
        y_pred, y, _, _ = output
        return y_pred, torch.argmax(y, dim=1)

    def output_transform_gp(output):
        y_pred, y, x, y_pred_sum = output
        return x, y_pred_sum

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
        y = F.one_hot(y, num_classes=10).float()
        x, y = x.cuda(), y.cuda()
        x.requires_grad_(True)
        y_pred = model(x)
        loss = F.binary_cross_entropy(y_pred, y)
        loss += l_gradient_penalty * calc_gradient_penalty(x, y_pred.sum(1))
        x.requires_grad_(False)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            model.eval()
            model.update_embeddings(x, y)
        return loss.item()

    def eval_step(engine, batch):
        model.eval()
        x, y = batch
        y = F.one_hot(y, num_classes=10).float()
        x, y = x.cuda(), y.cuda()
        x.requires_grad_(True)
        y_pred = model(x)
        return y_pred, y, x, y_pred.sum(1)

    
    trainer = Engine(step)
    evaluator = Engine(eval_step)

    metric = Accuracy(output_transform=output_transform_acc)
    metric.attach(evaluator, "accuracy")

    metric = Loss(F.binary_cross_entropy, output_transform=output_transform_bce)
    metric.attach(evaluator, "bce")

    metric = Loss(calc_gradient_penalty, output_transform=output_transform_gp)
    metric.attach(evaluator, "gradient_penalty")

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20], gamma=0.2)

    dl_train = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0, drop_last=True)
    dl_val = torch.utils.data.DataLoader(val_dataset, batch_size=2000, shuffle=False, num_workers=0)
    dl_test = torch.utils.data.DataLoader(test_dataset, batch_size=2000, shuffle=False, num_workers=0)

    pbar = ProgressBar()
    pbar.attach(trainer)

    
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_results(trainer):
        scheduler.step()

        if trainer.state.epoch % 5 == 0:
            evaluator.run(dl_val)
            _, roc_auc_mnist = get_fashionmnist_mnist_ood(model)
            _, roc_auc_notmnist = get_fashionmnist_notmnist_ood(model)

            metrics = evaluator.state.metrics
            print(
                f"Validation Results - Epoch: {trainer.state.epoch} "
                f"Acc: {metrics['accuracy']:.4f} "
                f"BCE: {metrics['bce']:.2f} "
                f"GP: {metrics['gradient_penalty']:.6f} "
                f"AUROC MNIST: {roc_auc_mnist:.2f} "
                f"AUROC NotMNIST: {roc_auc_notmnist:.2f} "
            )
            print(f"Sigma: {model.sigma}")

            
    trainer.run(dl_train, max_epochs=3)
    evaluator.run(dl_val)
    val_accuracy = evaluator.state.metrics["accuracy"]
    evaluator.run(dl_test)
    test_accuracy = evaluator.state.metrics["accuracy"]
    return model, val_accuracy, test_accuracy





if __name__ == "__main__":

    l_gradient_penalty = 0.0
    length_scale = 0.1
    final_model = False



    model, val_accuracy, test_accuracy = train_model(l_gradient_penalty, length_scale, final_model)
    accuracy, roc_auc_mnist = get_fashionmnist_mnist_ood(model)
    _, roc_auc_notmnist = get_fashionmnist_notmnist_ood(model)

    print(val_accuracy)
    print(test_accuracy)
    print(roc_auc_mnist)
    print(roc_auc_notmnist)