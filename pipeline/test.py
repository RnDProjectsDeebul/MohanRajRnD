import torch
from torchvision.models import resnet18,mobilenet_v2
from torch.utils.data import random_split
import torchvision.transforms as transforms
import torchvision
from torch import nn
from helpers import get_model, test_one_epoch, get_brier_score, get_expected_calibration_error,get_best_worst_predictions,plot_predicted_images,plot_entropy_correct_incorrect
from helpers import get_accuracy_score,get_precision_score,get_recall_score,get_f1_score,get_classification_report,plot_confusion_matrix1
import os
import time
import copy
import numpy as np
import pandas as pd
import neptune
import matplotlib.pyplot as plt
from helpers import get_multinomial_entropy,get_dirchlet_entropy,plot_calibration_curve
from data import import_data
from torch.quantization import quantize_fx
import warnings
warnings.filterwarnings("ignore")



data_dir = '../../data'
save_path = '../../results/'
models_path = save_path

parameters = {  'num_classes': 10,
                'batch_size': 128, 
                'model_name':'LeNet',#'Resnet18',
                'loss_function': 'Evidential',
                #'loss_function': 'Crossentropy',
                'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                'dataset': "MNIST",
                #'dataset': "CIFAR10",
                'quantise':True}
logger = False

if parameters['quantise'] == True:
    model_path = str(models_path)+str(parameters['loss_function'])+'_'+str(parameters['model_name'])+'_quant_model.pth'
    condition_name = str(parameters['loss_function'])+'_'+str(parameters['model_name'])+'_quant'
    entropy_df_condition = str(parameters['loss_function'])+'-Quant'
    name = "Testing" + "-" + str(parameters['model_name']) + "-" + str(parameters['loss_function']) + "-" + "Quant"
    tags = [str(parameters['loss_function']),str(parameters['model_name']),str(parameters['dataset']),"Testing", "Quant"]
    parameters['device'] = "cpu"
else:
    model_path = str(models_path)+str(parameters['loss_function'])+'_'+str(parameters['model_name'])+'_model.pth'
    condition_name = str(parameters['loss_function'])+'_'+str(parameters['model_name'])
    entropy_df_condition = str(parameters['loss_function'])
    name = "Testing" + "-" + str(parameters['model_name']) + "-" + str(parameters['loss_function'])
    tags = [str(parameters['loss_function']),str(parameters['model_name']),str(parameters['dataset']),"Testing"]


confusion_matrix_name = 'confusion_matrix_'+condition_name
entropy_plot_name = 'entropy_'+condition_name
calibration_plot_name = 'calibration_'+condition_name


if logger:
    run = neptune.init_run(
    #project="mohan20325145/CIFAR10",
    project="mohan20325145/MNIST",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJhZWQyMTU4OC02NmU4LTRiNjgtYWE5Zi1lNDg5MjdmZGJhNzYifQ==",
    tags = tags,
    name= name,
    )
else:
    run = None


dataloader, class_names = import_data(parameters['dataset'], data_dir, parameters['batch_size'], parameters['batch_size'])
test_loader = dataloader['val']
device = parameters['device']
model = get_model(parameters['model_name'],num_classes=parameters['num_classes'],weights=None)


if parameters['quantise'] == True:
    dataiter = iter(dataloader['train'])
    images, labels = next(dataiter)
    print(images.shape)

    m = copy.deepcopy(model)
    m.to("cpu")
    m.eval()
    qconfig_dict = {"": torch.quantization.get_default_qconfig("fbgemm")}
    model_prepared = quantize_fx.prepare_fx(m, qconfig_dict, images)

    with torch.inference_mode():
        for _ in range(10):
            images, labels = next(dataiter)
            model_prepared(images)
    model = quantize_fx.convert_fx(model_prepared)
    print("Model quantised, q-weights need be updated")
    
    
model.load_state_dict(torch.load(model_path))  
#model = torch.load(model_path)
model.eval()
print("Loading trained weights and eval mode is successful")
model.to(device=device)

print("Number of test images : ",len(test_loader)*parameters['batch_size'])

since = time.time()
results = test_one_epoch(model=model,
                         dataloader=test_loader,
                         device=device,
                         loss_function=parameters['loss_function'])
time_elapsed = round(time.time() - since, 3)

# seperate the results
true_labels = results['true_labels']
pred_labels = results['pred_labels']
confidences = results['confidences']
probabilities = results['probabilities']
model_output = results['model_output']
images_list = results['images_list']
labels_list = results['labels_list']

# Get best and worst predictions
best_predictions,worst_predictions = get_best_worst_predictions(confidences)

# classification metrics
accuracy_score = round(get_accuracy_score(true_labels=true_labels,predicted_labels=pred_labels),3)
precision_score = round(get_precision_score(true_labels,pred_labels),3)
recall_score = round(get_recall_score(true_labels,pred_labels),3)
f1_score = round(get_f1_score(true_labels,pred_labels),3)
classification_report = get_classification_report(true_labels,pred_labels,class_names)

# Print results
print("Accuracy score : ", accuracy_score)
print('--'*20)
print("Precision score : ", precision_score)
print('--'*20)
print("Recall score : ", recall_score)
print('--'*20)
print("F1 score : ", f1_score)
print('--'*20)
print("\nClassification report : ",classification_report)
print('--'*20)
print("\nInference time for ", true_labels.shape[0] ," image is : ", time_elapsed, "seconds")
print('--'*20)

# Uncertainty metrics
brier_score = get_brier_score(y_true=true_labels,y_pred_probs=probabilities)
#print("\n Shape of true labels is : ",true_labels.shape)
expected_calibration_error = get_expected_calibration_error(y_true=true_labels,y_pred=probabilities)

print("Brier Score : ", round(brier_score,5))
print('--'*20)
print("Expected calibration error : ", round(expected_calibration_error,5))

# Plot confusion matrix
confusion_mat_fig = plot_confusion_matrix1(true_labels=true_labels,
                                            predicted_labels=pred_labels,
                                            class_names=class_names,
                                            results_path=save_path,
                                            plot_name=confusion_matrix_name)

    
# Calculate entropy values
if parameters['loss_function']=='Crossentropy':
    entropy_values = get_multinomial_entropy(probabilities)
    entropy_values_df = pd.DataFrame(entropy_values)
    entropy_values_save_path = str(save_path)+str(condition_name)+'_entropy(probabilities).csv'
    #entropy_values_df.to_csv(path_or_buf=entropy_values_save_path)
    
    entropy_val = get_multinomial_entropy(model_output)
    entropy_val_df = pd.DataFrame(entropy_val)
    entropy_val_save_path = str(save_path)+str(condition_name)+'_entropy(logits).csv'
    #entropy_val_df.to_csv(path_or_buf=entropy_val_save_path)


elif parameters['loss_function']== 'Evidential':
    entropy_values = get_dirchlet_entropy(model_output)
    

results_dict = {
    "entropy": entropy_values,
    "true_labels": true_labels,
    "pred_labels":pred_labels,
    "condition":entropy_df_condition,
    "accuracy":accuracy_score,
    "precision":precision_score,
    "recall_score":recall_score,
    "f1score":f1_score,
    "brierscore":brier_score,
    "expectedcalibrationerror":expected_calibration_error,
    "inferencetime":time_elapsed,
}
results_df = pd.DataFrame(results_dict)
results_df = results_df.astype({'entropy': 'float64'})
results_df['is_prediction_correct'] = results_df['true_labels'] == results_df['pred_labels']
results_df.to_csv(path_or_buf= save_path+condition_name+'_entropy.csv')

#Plot entropy
entropy_plot_fig = plot_entropy_correct_incorrect(data_df=results_df, save_path=save_path, file_name=entropy_plot_name)

#Plot calibration curve
calibration_curve_fig = plot_calibration_curve(y_prob=probabilities, y_true=true_labels, num_classes=parameters['num_classes'],                                                             save_path=save_path, file_name=calibration_plot_name)

#save model logits as csv
actual_labels = true_labels.reshape(-1, 1)
logits_truelabel = np.concatenate((model_output, actual_labels), axis=1)
logits_df = pd.DataFrame(logits_truelabel)
logits_save_path = str(save_path)+str(condition_name)+'_logits.csv'
#logits_df.to_csv(path_or_buf=logits_save_path)

if run !=None:
    run['config/hyperparameters'] = parameters
    run['config/model'] = type(model).__name__
    
    run['metrics/accuracy'] = accuracy_score
    run['metrics/precision_score'] = precision_score
    run['metrics/recall_score'] = recall_score
    run['metrics/f1_score'] = f1_score
    run['metrics/brier_score'] = brier_score
    run['metrics/expected_calibration_error'] = expected_calibration_error
    run['metrics/classification_report'] = classification_report

    run['metrics/images/confusion_matrix'].upload(confusion_mat_fig)
    run['metrics/images/entropy_correct_incorrect'].upload(entropy_plot_fig)
    run['metrics/images/calibration_plot'].upload(calibration_curve_fig)