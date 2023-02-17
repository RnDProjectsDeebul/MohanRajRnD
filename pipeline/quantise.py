import onnx
import torch
import torchvision
import torchvision.transforms as transforms
from helpers import get_model
from onnx2pytorch import ConvertModel
from onnxruntime.quantization import quantize_dynamic, QuantType



data_dir = '../../data'
save_path = '../../results/'
models_path = save_path

parameters = {  
                 #'loss_function':'Evidential',
                 'loss_function': 'Crossentropy',
                 'model_name':'Resnet18',
                 'num_classes': 10
             }

condition_name = str(parameters['loss_function'])+'_'+str(parameters['model_name'])

model_path = str(models_path)+str(parameters['loss_function'])+'_Resnet18_model.pth'
onnx_model_path = model_path[:-3]+"onnx"
onnx_quant_model_path = model_path[:-4]+"_quant.onnx"

model = get_model(parameters['model_name'],num_classes=parameters['num_classes'],weights='DEFAULT')
model.load_state_dict(torch.load(model_path))


transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), transforms.RandomErasing()])
trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=40, shuffle=True, num_workers=2)
dataiter = iter(trainloader)
images, labels = next(dataiter)


torch.onnx.export(model,               
                  images,                         
                  onnx_model_path,   
                  export_params=True,             
                  do_constant_folding=True,  
                  input_names = ['input'],   
                  output_names = ['output'], 
                  dynamic_axes={'input' : {0 : 'batch_size'},   
                                'output' : {0 : 'batch_size'}})

_ = quantize_dynamic(onnx_model_path, onnx_quant_model_path)
quantized_model = onnx.load(onnx_quant_model_path)
model = ConvertModel(quantized_model)

torch.save(model.state_dict(), save_path+str(condition_name)+str('_model_quant.pth'))