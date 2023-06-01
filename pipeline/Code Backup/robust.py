import foolbox as fb
import torch
import torchvision
from torchvision.models import ResNet18_Weights
from helpers import get_model
import numpy as np
import matplotlib.pyplot as plt
import eagerpy as ep

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
testset = torchvision.datasets.CIFAR10(root='../../data', train=False, download=True, transform=transform)

save_path = '../../results/'
models_path = save_path

parameters = {  'num_classes': 10,
                'batch_size': 20, 
                'model_name':'Resnet18',
                #'loss_function': 'Evidential',
                'loss_function': 'Crossentropy',
                'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                'quantise':True}
model_path = str(models_path)+str(parameters['loss_function'])+'_'+str(parameters['model_name'])+'_model.pth'

model = get_model(parameters['model_name'],num_classes=parameters['num_classes'],weights=None)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))) 

#model = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)

model = model.eval()
preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
bounds = (0, 1)
fmodel = fb.PyTorchModel(model, bounds=bounds, preprocessing=preprocessing)
fmodel = fmodel.transform_bounds((0, 1))
images, labels = fb.utils.samples(fmodel, dataset='cifar10', batchsize=16)

accuracy = fb.utils.accuracy(fmodel, images, labels)
print(accuracy)
attack = fb.attacks.LinfDeepFoolAttack()

# raw, clipped, is_adv = attack(fmodel, images, labels, epsilons=0.03)
# accuracy = is_adv.float().mean().item()
# print(accuracy)

epsilons = np.linspace(0.0, 0.005, num=20)
raw, clipped, is_adv = attack(fmodel, images, labels, epsilons=epsilons)

robust_accuracy = 1 - is_adv.float().mean(axis=-1)
print(robust_accuracy)
plt.plot(epsilons, robust_accuracy.cpu().numpy())
plt.savefig('robust_accuracy.png')
a = fb.plot.images(images)
