from utility.settings import *
#from utility.transformation import training_transforms
import utility.transformation as t
from unet_model.training.ImageDataset import ImageDataset
from torch.utils.data import random_split
import torch
import numpy as np
import random

def get_train_data_loaders(image_path:str = 'training_data/nonuveaux_and_nuevo_images/', label_path:str = 'training_data/nouveaux_labels/', 
                           training_split:float = 0.9, batch_size:int = 5, 
                           num_workers=0, transform_generator=t.training_transforms(target_resolution=256, prob_flip=1.0)):
    

    transforms = transform_generator #this used to be called, transform_generator() 
    imageset = ImageDataset(
        image_dir=image_path,
        label_dir=label_path,
        transform=transforms
    )

    print('images found = ',imageset.__len__())


    ### 
    #NOTE: I don't think it is necessary to split the dataset into training and validation. 
    # We test the model on a very different dataset than the training dataset, so we will catch 
    # if it overfits to the training dataset, etc. 
    # Also, since the training dataset is small, we want to use all of it for training.


    ### 
    # # Define the sizes of your training and validation sets
    # train_size = int(float(training_split) * len(imageset))
    # val_size = len(imageset) - train_size

    g = torch.Generator()
    g.manual_seed(42)  # Set a seed for reproducibility


    # Split the dataset
  #  train_dataset, val_dataset = random_split(imageset, [train_size, val_size], generator=g)


  


    train_dataloader = torch.utils.data.DataLoader(
        #dataset=train_dataset,
        dataset=imageset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g
    )

    # val_dataloader = torch.utils.data.DataLoader(
    #     dataset=val_dataset,
    #     batch_size=batch_size,
    #     shuffle=True,
    #     num_workers=num_workers,
    #     worker_init_fn=seed_worker,
    #     generator=g
    # )

    #return train_dataloader, val_dataloader
    return train_dataloader 

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)