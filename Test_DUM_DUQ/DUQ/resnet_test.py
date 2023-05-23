import copy
import torch
import torchvision
import numpy as np
from torchvision.models import resnet18
from utils.resnet_duq import ResNet_DUQ
from torch.quantization import quantize_fx
import torchvision.transforms as transforms
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import accuracy_score


feature_extractor = resnet18()
feature_extractor.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
feature_extractor.maxpool = torch.nn.Identity()
feature_extractor.fc = torch.nn.Identity()

num_classes = 10
centroid_size = 512
model_output_size = 512
length_scale = 0.1
gamma = 0.999



model = ResNet_DUQ(
        feature_extractor,
        num_classes,
        centroid_size,
        model_output_size,
        length_scale,
        gamma,)

model_path = '../../../runs/results/model.pt'
model.load_state_dict(torch.load(model_path)) 


def load_test_data(data_dir):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)   
    return testset
data_dir = '../../../data/'
tes = load_test_data(data_dir)
testloader = torch.utils.data.DataLoader(tes, batch_size=100, shuffle=False, num_workers=2)
dataloader = {"test": testloader}
dataiter = iter(dataloader['test'])
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

q_model_path = '../../../runs/results/q_model.pt'
torch.save(q_model.state_dict(), q_model_path)
q_model.load_state_dict(torch.load(q_model_path))





model.eval()
model.to('cpu')
q_model.eval()
q_model.to('cpu')

inputs, labels = next(dataiter)
inputs.to('cpu')
labels.to('cpu')
true_labels = np.array(labels)

def run_test(net, images):
    with torch.no_grad():
        out = net(images)
        _, preds = torch.max(out, 1)
        predict_labels = np.array(preds)

    accuracy = accuracy_score(true_labels, predict_labels)
    return accuracy

acc = run_test(model, inputs)
print("Standard: ", acc)
acc = run_test(q_model, inputs)
print("Quantise: ", acc)