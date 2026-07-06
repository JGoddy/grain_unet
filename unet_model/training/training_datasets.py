from utility.settings import *
#from utility.transformation import training_transforms
import utility.transformation as t
from unet_model.training.ImageDataset import ImageDataset
from torch.utils.data import random_split
import torch

def get_train_data_loaders(image_path:str = 'training_data/image/', label_path:str = 'training_data/nouveaux_labels/', 
                           training_split:float = 0.9, batch_size:int = 5, 
                           num_workers=0, transform_generator=t.training_transforms(target_resolution=256, prob_flip=1.0)):
    

    transforms = transform_generator #this used to be called, transform_generator() 
    imageset = ImageDataset(
        image_dir=image_path,
        label_dir=label_path,
        transform=transforms
    )

    print('images found = ',imageset.__len__())

    # Define the sizes of your training and validation sets
    train_size = int(float(training_split) * len(imageset))
    val_size = len(imageset) - train_size

    # Split the dataset
    train_dataset, val_dataset = random_split(imageset, [train_size, val_size])


    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    return train_dataloader, val_dataloader