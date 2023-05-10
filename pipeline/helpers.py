import torch
from torchvision.models import resnet18,mobilenet_v2,ResNet18_Weights
from torch import nn
import torch.nn.functional as F
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import dirichlet, multinomial
from sklearn.metrics import classification_report,accuracy_score,f1_score,precision_score,recall_score
from scikitplot.metrics import plot_confusion_matrix
import seaborn as sns
import sys
import matplotlib.image as mpimg
from sklearn.calibration import calibration_curve
from lenet import LeNet




def plot_calibration_curve(y_prob, y_true, num_classes, save_path, file_name):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for i in range(num_classes):
        prob_true, prob_pred = calibration_curve(y_true == i, y_prob[:, i], n_bins=10)
        ax.plot(prob_pred, prob_true, "s-", label="Class %d" % i)
    ax.set_title('Calibration curve')
    ax.set_xlabel('Predicted probability')
    ax.set_ylabel('True probability')
    ax.legend()
    plt.tight_layout()
    #plt.savefig(save_path+file_name)
    return fig


def plot_entropy_correct_incorrect(data_df, save_path, file_name):
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.figure(figsize=(10,10))

    d = {True: 'correct', False: 'incorrect'}
    data_df['is_prediction_correct'] = data_df['is_prediction_correct'].replace(d)
    g = sns.boxplot(data=data_df, y="entropy", x="is_prediction_correct", ax=ax)
    #plt.savefig(save_path+file_name)
    return fig




def read_csv_files(file_path):
    df = pd.read_csv(file_path)
    return df

def calculate_multinomial_entropy( p_values ):
    return multinomial(1, p_values).entropy()

def calculate_dirchlet_entropy(alpha_values):
    return dirichlet(alpha_values).entropy()


def get_multinomial_entropy( p_values ):
    entropy_values = []
    for i in p_values:
        entropy_values.append(multinomial(1, i).entropy())
    return entropy_values


def get_dirchlet_entropy(alpha_values):
    entropy_values = []
    for i in alpha_values:
        entropy_values.append(calculate_dirchlet_entropy(i))
    return entropy_values

# def get_multinomial_entropy( p_values ):
#     return multinomial(1, p_values).entropy()

# def get_dirchlet_entropy(alpha_values):    
#     return dirichlet(alpha_values).entropy()


def relu_evidence(y):
    return F.relu(y)


def get_model(model_name,num_classes,weights):
    if model_name == 'Resnet18':
        model = resnet18(weights = ResNet18_Weights.DEFAULT)
        assert model.__class__.__name__ == 'ResNet'
        model.fc = nn.Linear(in_features=512,out_features=num_classes)
    elif model_name == 'MobileNetV2':
        model = mobilenet_v2(weights=weights)
        assert model.__class__.__name__ == 'MobileNetV2'
        model.classifier[1] = nn.Linear(in_features=1280,out_features=num_classes)
    elif model_name == 'LeNet':
        model = LeNet()
    return model


def test_one_epoch(dataloader,num_classes,model,device,loss_function):
    print("Started testing")    

    true_labels = []
    predicted_labels = []

    softmax_probabilities = []
    cross_entropy_output = []

    evidential_probabilities = []
    dirichlet_alpha_output = []
    
    uncertainty = []

    # Begin testing
    with torch.no_grad():
        for batch_idx,(inputs,labels) in enumerate(dataloader):
            inputs,labels = inputs.to(device),labels.to(device)
                
            if loss_function == 'Crossentropy': 
                model.eval()
                model.to(device=device)
                
                output = model(inputs)
                cross_entropy_output.extend(output.cpu().numpy())
                
                _,predictions = torch.max(output,1)
                predicted_labels.extend(predictions.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
                
                probs = torch.softmax(output,dim=1)
                softmax_probabilities.extend(probs.cpu().numpy())
                 
            elif loss_function[0:10] == 'Evidential': 
                model.eval()
                model.to(device=device)
                
                output = model(inputs)
                evidence = relu_evidence(output)
                alpha = evidence + 1
                dirichlet_alpha_output.extend(alpha.cpu().numpy())
                
                _,predictions = torch.max(output,1)
                predicted_labels.extend(predictions.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
                
                probs = alpha / torch.sum(alpha, dim=1, keepdim=True)
                evidential_probabilities.extend(probs.cpu().numpy())
                
                u = num_classes / torch.sum(alpha, dim=1, keepdim=True)
                uncertainty.extend(u.cpu().numpy())


    if loss_function == 'Crossentropy':    
        ce_results_dict = {
                        "true_labels":np.array(true_labels),
                        "pred_labels":np.array(predicted_labels),
                        "probabilities":np.array(softmax_probabilities),
                        "model_output":np.array(cross_entropy_output),
        }
        return ce_results_dict
        
    elif loss_function[0:10] == 'Evidential':
        evi_results_dict = {
                        "true_labels":np.array(true_labels),
                        "pred_labels":np.array(predicted_labels),
                        "probabilities":np.array(evidential_probabilities),
                        "model_output":np.array(dirichlet_alpha_output),
        }
        return evi_results_dict

    
def get_best_worst_predictions(confidence_array):
    confidences_idxs_sorted = np.argsort(confidence_array)
    best_predictions = confidences_idxs_sorted[-9 : ]
    worst_predictions = confidences_idxs_sorted[:9]

    return best_predictions,worst_predictions

def plot_predicted_images(predictions,confidences_pred_images,image_paths,labels,class_names,plot_preds,results_path,plot_name):

    rows = 3
    columns = 3
    fig = plt.figure(figsize=(20,20))
    
    for idx,i in enumerate(plot_preds):
        idx = idx+1
        image = image_paths[i]
        img_ = mpimg.imread(str(image))
        # img_ = images[i].cpu().detach()
        label_true = labels[i]
        label_pred = predictions[i]
        confidence = confidences_pred_images [i]

        fig.add_subplot(rows, columns, idx)
        # showing image
        plt.imshow(img_)
        plt.title("True_label : "+ str(class_names[label_true]))
        plt.xlabel("Confidance : " + str(round(confidence,2)))
        plt.ylabel("Pred_label : " + str(class_names[label_pred]))

        plt.rc('axes', titlesize=20)
        plt.rc('axes', labelsize=20)
        plt.rc('xtick', labelsize=0)
        plt.rc('ytick', labelsize=0)

        # plt.savefig(results_path +'best_pred_evidential_dark_light.png' )
        plt.savefig(str(results_path) + str(plot_name) +'.png' )
    return fig


def get_brier_score(y_true, y_pred_probs):
    return 1 + (np.sum(y_pred_probs ** 2) - 2 * np.sum(y_pred_probs[np.arange(y_pred_probs.shape[0]), y_true])) / y_true.shape[0]


def get_expected_calibration_error(y_true, y_pred, num_bins=15):
    pred_y = np.argmax(y_pred, axis=-1)
    correct = (pred_y == y_true).astype(np.float32)
    prob_y = np.max(y_pred, axis=-1)

    b = np.linspace(start=0, stop=1.0, num=num_bins)
    bins = np.digitize(prob_y, bins=b, right=True)

    o = 0
    for b in range(num_bins):
        mask = bins == b
        if np.any(mask):
            o += np.abs(np.sum(correct[mask] - prob_y[mask]))
  
    x = np.sum(prob_y[mask]/prob_y.shape[0])
    y = np.sum(y_true[mask] /y_true.shape[0])

    return o / y_pred.shape[0]


def one_hot_embedding(labels, num_classes):
    # Convert to One Hot Encoding
    y = torch.eye(num_classes)
    device = get_device()
    y = y.to(device)
    return y[labels]

def save_architecture_txt(model,dir_path,filename):
        complete_file_name = os.path.join(dir_path, filename+"_arch.txt")
        with open(complete_file_name, "w") as f:
                f.write(str(model))
                f.close()
        return None

def get_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device

def get_accuracy_score(true_labels,predicted_labels):
    return accuracy_score(true_labels,predicted_labels)

def get_precision_score(true_labels,predicted_labels):
    return precision_score(true_labels,predicted_labels,average='weighted')

def plot_confusion_matrix1(true_labels,predicted_labels,class_names,results_path,plot_name):
        fig, axs = plt.subplots(figsize=(20, 20))
        plot_confusion_matrix(true_labels,predicted_labels , ax=axs,normalize=True)
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45, fontsize=15)
        plt.yticks(tick_marks, class_names, rotation=45,fontsize=15)
        plt.title('Confusion Matrix', fontsize=20)
        plt.xlabel("Predicted label",fontsize=20)
        plt.ylabel("True label",fontsize=20)
        #plt.savefig(str(results_path)+str(plot_name)+'.png')
        return fig

def get_classification_report(true_labels,predicted_labels,classes):
        class_report = classification_report(true_labels, predicted_labels,target_names=classes)
        return class_report

def get_f1_score(true_labels,predicted_labels):
        f1score = f1_score(true_labels, predicted_labels,average='weighted')
        return f1score

def get_recall_score(true_labels,predicted_labels):
        recall = recall_score(true_labels, predicted_labels,average='weighted')
        return recall

def plot_losses(train_loss,valid_loss,criterion_name,save_path):
        fig, ax = plt.subplots(figsize=(20,20))
        ax.plot(train_loss,label='Training Loss')
        ax.plot(valid_loss,label='Validation Loss')
        plt.xlabel('# Epoch', fontsize=15)
        plt.ylabel(str(criterion_name),fontsize=15)
        plt.title('Loss Plot')
        plt.legend()
        #fig.savefig(str(save_path)+'/losses_plot.png')
        return fig

def plot_accuracies(train_acc,valid_acc,save_path):
        fig, ax = plt.subplots(figsize=(20,20))
        ax.plot(train_acc,label='Training accuracy')
        ax.plot(valid_acc,label='Validation accuracy')
        plt.xlabel('# Epoch')
        plt.ylabel('Accuracy')
        plt.title("Accuracy Plot")
        plt.legend()
        #fig.savefig(str(save_path)+'/accuracies_plot.png')
        return fig

def imshow(img):
    ''' function to show image '''
    img = img / 2 + 0.5 # unnormalize
    npimg = img.numpy() # convert to numpy objects
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# Early stopping
def early_stopping(train_loss,validation_loss,min_delta,tolerance):
        counter = 0
        if (validation_loss-train_loss) > min_delta:
                counter += 1
                if counter>=tolerance:
                        return True
