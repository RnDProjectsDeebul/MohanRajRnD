"""
Script for training a single model for OOD detection.
"""

import copy
import pandas as pd
import torchvision
from torch.quantization import quantize_fx
import torchvision.transforms as transforms
import warnings
warnings.filterwarnings("ignore")

import json
import torch
import argparse
from torch import optim
import torch.backends.cudnn as cudnn

# Import dataloaders
import data.ood_detection.cifar10 as cifar10
import data.ood_detection.cifar100 as cifar100
import data.ood_detection.svhn as svhn
import data.dirty_mnist as dirty_mnist

# Import network models
from net.lenet import lenet
from net.resnet import resnet18, resnet50
from net.wide_resnet import wrn
from net.vgg import vgg16

# Import train and validation utilities
from utils.args import training_args
from utils.eval_utils import get_eval_stats
from utils.train_utils import model_save_name
from utils.train_utils import train_single_epoch, test_single_epoch

# Tensorboard utilities
from torch.utils.tensorboard import SummaryWriter


dataset_num_classes = {"cifar10": 10, "cifar100": 100, "svhn": 10, "dirty_mnist": 10}

dataset_loader = {
    "cifar10": cifar10,
    "cifar100": cifar100,
    "svhn": svhn,
    "dirty_mnist": dirty_mnist,
}

models = {
    "lenet": lenet,
    "resnet18": resnet18,
    "resnet50": resnet50,
    "wide_resnet": wrn,
    "vgg16": vgg16,
}


if __name__ == "__main__":

    args = training_args().parse_args()

    print("Parsed args", args)
    print("Seed: ", args.seed)
    torch.manual_seed(args.seed)

    cuda = torch.cuda.is_available() and args.gpu
    device = torch.device("cuda" if cuda else "cpu")
    print("CUDA set: " + str(cuda))

    num_classes = dataset_num_classes[args.dataset]

    net = models[args.model](
        spectral_normalization=args.sn,
        mod=args.mod,
        coeff=args.coeff,
        num_classes=num_classes,
        mnist="mnist" in args.dataset,
    )

    if args.gpu:
        net.cuda()

    opt_params = net.parameters()
    if args.optimiser == "sgd":
        optimizer = optim.SGD(
            opt_params,
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov,
        )
    elif args.optimiser == "adam":
        optimizer = optim.Adam(opt_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[args.first_milestone, args.second_milestone], gamma=0.1
    )

    train_loader, _ = dataset_loader[args.dataset].get_train_valid_loader(
        root=args.dataset_root,
        batch_size=args.train_batch_size,
        augment=args.data_aug,
        val_size=0.1,
        val_seed=args.seed,
        pin_memory=args.gpu,
    )



    for epoch in range(0, args.epoch):
        print("Starting epoch", epoch)
        train_loss = train_single_epoch(epoch, net, train_loader, optimizer, device, loss_function=args.loss_function, loss_mean=args.loss_mean,)
        scheduler.step()
        
    model_path = '../../results/DDU_ResNet_DDU_model.pth'
    torch.save(net.state_dict(), model_path)
    print("Model saved to ", model_path)





    ###############
    #Quantise
    ###############

    def load_train_data(data_dir):
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
        train_data = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)   
        return train_data

    data_dir = '../../data'
    train = load_train_data(data_dir)
    trainloader = torch.utils.data.DataLoader(train, batch_size=128, shuffle=True, num_workers=2)
    dataloader = {"train": trainloader}
    
    

    model = models[args.model](spectral_normalization=args.sn, mod=args.mod, coeff=args.coeff, num_classes=num_classes, mnist="mnist" in args.dataset,)
    model.load_state_dict(torch.load(model_path)) 


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

    q_model_path = '../../results/DDU_ResNet_DDU_quant_model.pth'
    torch.save(q_model.state_dict(), q_model_path)
    print("Quantised Model saved to ", q_model_path)
