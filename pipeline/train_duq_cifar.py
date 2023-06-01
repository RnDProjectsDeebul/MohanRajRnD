import argparse
import json
import pathlib
import random

import torch
import torch.nn.functional as F
import torch.utils.data
from torch.utils.tensorboard.writer import SummaryWriter

from torchvision.models import resnet18

from ignite.engine import Events, Engine
from ignite.metrics import Accuracy, Average, Loss
from ignite.contrib.handlers import ProgressBar

from utils.wide_resnet import WideResNet
from utils.resnet_duq import ResNet_DUQ
from utils.datasets import all_datasets
from utils.evaluate_ood import get_cifar_svhn_ood, get_auroc_classification


from data import import_data
import copy
import pandas as pd
import torchvision
from torch.quantization import quantize_fx
import torchvision.transforms as transforms
import warnings
warnings.filterwarnings("ignore")


def main(architecture,batch_size,length_scale,centroid_size,learning_rate,l_gradient_penalty,gamma,weight_decay,final_model,
    output_dir,):
    
    writer = SummaryWriter(log_dir=f"../../results/duq_writer")
    loss_acc_dict = {"epoch_no": [],"train_loss": [],"val_loss": [],"train_acc": [],"val_acc": []}

    ds = all_datasets["CIFAR10"]()
    input_size, num_classes, dataset, test_dataset = ds
    train_dataset = dataset
    val_dataset = test_dataset


    model_output_size = 512
    epochs = 3
    milestones = [25, 50, 75]
    feature_extractor = resnet18()
    feature_extractor.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    feature_extractor.maxpool = torch.nn.Identity()
    feature_extractor.fc = torch.nn.Identity()

    if centroid_size is None:
        centroid_size = model_output_size

    model = ResNet_DUQ(feature_extractor,num_classes,centroid_size,model_output_size,length_scale,gamma,)
    model = model.cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.2)

    def calc_gradients_input(x, y_pred):
        gradients = torch.autograd.grad(
            outputs=y_pred,
            inputs=x,
            grad_outputs=torch.ones_like(y_pred),
            create_graph=True,
        )[0]

        gradients = gradients.flatten(start_dim=1)

        return gradients

    def calc_gradient_penalty(x, y_pred):
        gradients = calc_gradients_input(x, y_pred)
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
        loss = F.binary_cross_entropy(y_pred, y, reduction="mean")

        if l_gradient_penalty > 0:
            gp = calc_gradient_penalty(x, y_pred)
            loss += l_gradient_penalty * gp

        loss.backward()
        optimizer.step()
        x.requires_grad_(False)

        with torch.no_grad():
            model.eval()
            model.update_embeddings(x, y)
        #return loss.item()
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

    kwargs = {"num_workers": 4, "pin_memory": True}

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

    
    
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_results(trainer):
        
        ## Training metrics
        metrics = trainer.state.metrics
        loss = metrics["loss"]
        print(f"Train - Epoch: {trainer.state.epoch} Loss: {loss:.2f}")
        writer.add_scalar("Loss/train", loss, trainer.state.epoch)
        
        loss_acc_dict['epoch_no'].append(trainer.state.epoch)
        loss_acc_dict['train_loss'].append(metrics["loss"])
        loss_acc_dict['train_acc'].append(metrics["train_accuracy"])

        
        if trainer.state.epoch > (epochs - 5):
            accuracy, auroc = get_cifar_svhn_ood(model)
            print(f"Test Accuracy: {accuracy}, AUROC: {auroc}")
            writer.add_scalar("OoD/test_accuracy", accuracy, trainer.state.epoch)
            writer.add_scalar("OoD/roc_auc", auroc, trainer.state.epoch)

            accuracy, auroc = get_auroc_classification(val_dataset, model)
            print(f"AUROC - uncertainty: {auroc}")
            writer.add_scalar("OoD/val_accuracy", accuracy, trainer.state.epoch)
            writer.add_scalar("OoD/roc_auc_classification", auroc, trainer.state.epoch)

            
        ## Validation metrics
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        acc = metrics["accuracy"]
        bce = metrics["bce"]
        GP = metrics["gradient_penalty"]
        loss = bce + l_gradient_penalty * GP

        print((
                f"Valid - Epoch: {trainer.state.epoch} "
                f"Acc: {acc:.4f} "
                f"Loss: {loss:.2f} "
                f"BCE: {bce:.2f} "
                f"GP: {GP:.2f} "
            ))

        writer.add_scalar("Loss/valid", loss, trainer.state.epoch)
        writer.add_scalar("BCE/valid", bce, trainer.state.epoch)
        writer.add_scalar("GP/valid", GP, trainer.state.epoch)
        writer.add_scalar("Accuracy/valid", acc, trainer.state.epoch)
        
        loss_acc_dict['val_loss'].append(loss)
        loss_acc_dict['val_acc'].append(acc)
        
        scheduler.step()

        
    trainer.run(train_loader, max_epochs=epochs)
    evaluator.run(test_loader)
    acc = evaluator.state.metrics["accuracy"]

    print(f"Test - Accuracy {acc:.4f}")

    torch.save(model.state_dict(), f"../../results/DUQ_ResNet_DUQ_model.pth")
    writer.close()
    
    condition_name = '../results/'+'DUQ'+'_'+'ResNet_DUQ'+'_loss_acc.csv'
    loss_acc_df = pd.DataFrame(loss_acc_dict, columns=["epoch_no","train_loss","val_loss","train_acc","val_acc"])    
    loss_acc_df.to_csv(path_or_buf=condition_name)


    
    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--architecture",
        default="ResNet18",
        choices=["ResNet18", "WRN"],
        help="Pick an architecture (default: ResNet18)",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size to use for training (default: 128)",
    )

    parser.add_argument(
        "--centroid_size",
        type=int,
        default=None,
        help="Size to use for centroids (default: same as model output)",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.05,
        help="Learning rate (default: 0.05)",
    )

    parser.add_argument(
        "--l_gradient_penalty",
        type=float,
        default=0.5,
        help="Weight for gradient penalty (default: 0.75)",
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.999,
        help="Decay factor for exponential average (default: 0.999)",
    )

    parser.add_argument(
        "--length_scale",
        type=float,
        default=0.1,
        help="Length scale of RBF kernel (default: 0.1)",
    )

    parser.add_argument(
        "--weight_decay", type=float, default=5e-4, help="Weight decay (default: 5e-4)"
    )

    parser.add_argument(
        "--output_dir", type=str, default="results", help="set output folder"
    )

    parser.add_argument(
        "--final_model",
        action="store_true",
        default=True,
        help="Use entire training set for final model",
    )

    args = parser.parse_args()
    kwargs = vars(args)
    print("input args:\n", json.dumps(kwargs, indent=4, separators=(",", ":")))

    pathlib.Path("../../results/" + "duq_writer").mkdir(exist_ok=True)

    main(**kwargs)
    
    
    
    
    
    
    ###############
    #Quantise
    ###############
    
    feature_extractor = resnet18()
    feature_extractor.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    feature_extractor.maxpool = torch.nn.Identity()
    feature_extractor.fc = torch.nn.Identity()

    num_classes = 10
    centroid_size = 512
    model_output_size = 512
    length_scale = 0.1
    gamma = 0.999

    model = ResNet_DUQ(feature_extractor,num_classes,centroid_size,model_output_size,length_scale,gamma,)
    model_path = '../../results/DUQ_ResNet_DUQ_model.pth'
    model.load_state_dict(torch.load(model_path)) 
    dataloader, class_names = import_data("CIFAR10", '../../data', 128, 128)


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
    test_out = q_model(img)

    q_model_path = '../../results/DUQ_ResNet_DUQ_quant_model.pth'
    torch.save(q_model.state_dict(), q_model_path)
    
    
    # q_model.load_state_dict(torch.load(q_model_path))
    # model.eval()
    # model.to('cpu')
    # q_model.eval()
    # q_model.to('cpu')

    # inputs, labels = next(dataiter)
    # inputs.to('cpu')
    # labels.to('cpu')
    # true_labels = np.array(labels)

    # def run_test(net, images):
    #     with torch.no_grad():
    #         out = net(images)
    #         _, preds = torch.max(out, 1)
    #         predict_labels = np.array(preds)

    #     accuracy = accuracy_score(true_labels, predict_labels)
    #     return accuracy

    # acc = run_test(model, inputs)
    # print("Standard: ", acc)
    # acc = run_test(q_model, inputs)
    # print("Quantise: ", acc)