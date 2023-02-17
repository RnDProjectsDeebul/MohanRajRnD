import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader , random_split
from torchvision import datasets

# helpers to create dataloaders
class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path



def load_data(data_dir,batch_size,logger):
    """ Function to create train, validation and test dataloaders 
    """

    transform = transforms.Compose([
        transforms.Resize((96,96)),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],
        [0.5,0.5,0.5])
        ])

    random_seed = 42
    torch.manual_seed(random_seed)

    data_set = ImageFolderWithPaths(data_dir,transform=transform)
    num_valid = int(np.floor(0.2*len(data_set)))
    num_test = int(np.floor(0.1*len(data_set)))
    num_train = len(data_set)-num_valid - num_test

    train_set,validation_set,test_set = random_split(data_set, [num_train,num_valid,num_test])

    train_dataloader = DataLoader(dataset=train_set,batch_size=batch_size,shuffle=True,num_workers=8)
    validation_dataloader = DataLoader(dataset=validation_set,batch_size=batch_size,shuffle=True,num_workers=8)
    test_dataloader = DataLoader(dataset=test_set,batch_size=batch_size,shuffle=True,num_workers=8)

    dataloaders = {
        "train": train_dataloader,
        "val": validation_dataloader,
        "test": test_dataloader }

    class_names = data_set.classes

    dataset_parameters={'transforms': transform,
                        'random_seed':random_seed,
                        'train_size': len(train_dataloader)*batch_size,
                        'val_size': len(validation_dataloader)*batch_size,
                        'test_size':len(test_dataloader)*batch_size}

    if logger != None:
        logger['config/dataset/'] = dataset_parameters
    else:
        pass

    return dataloaders,class_names


def load_synthetic_data(data_dir,batch_size,logger):
    """ Function to create dataloader for train and validation sets only
    """
    transform = transforms.Compose([
        transforms.Resize((96,96)),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],
        [0.5,0.5,0.5])
        ])
    random_seed = 42
    torch.manual_seed(random_seed)

    data_set = ImageFolderWithPaths(data_dir,transform=transform)
    num_valid = int(np.floor(0.3*len(data_set)))
    num_train = len(data_set)-num_valid

    train_set,validation_set = random_split(data_set, [num_train,num_valid])

    train_dataloader = DataLoader(dataset=train_set,batch_size=batch_size,shuffle=True,num_workers=8)
    validation_dataloader = DataLoader(dataset=validation_set,batch_size=batch_size,shuffle=True,num_workers=8)

    dataloaders = {
        "train": train_dataloader,
        "val": validation_dataloader,
        }

    class_names = data_set.classes

    dataset_parameters={'transforms': transform,
                        'random_seed':random_seed,
                        'train_size': len(train_dataloader)*batch_size,
                        'val_size': len(validation_dataloader)*batch_size,
                        }
    if logger != None:
        logger['config/dataset/'] = dataset_parameters
    else:
        pass

    return dataloaders,class_names
    

def load_test_data(data_dir,batch_size,logger):
    """Function to create test dataloader
    """
    random_seed = 42
    transform = transforms.Compose([
        transforms.Resize((96,96)),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])

    torch.manual_seed(random_seed)

    test_set = ImageFolderWithPaths(data_dir,transform=transform)

    test_dataloader = DataLoader(
                                dataset=test_set,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=8
                                )

    class_names = test_set.classes

    dataloaders = {
        "test": test_dataloader,
        }

    dataset_parameters={'transforms': transform,
                        'random_seed':random_seed,
                        'test_set_size':len(test_dataloader)*batch_size,
                        'num_of_classes': len(class_names)
                        }

    if logger != None:
        logger['config/dataset/'] = dataset_parameters
    else:
        pass    

    return dataloaders,class_names