# from utility.settings import init, init_training
# init()
# init_training() 
#from utility.settings import *

#from utility.transformation import training_transforms
import utility.transformation as t

from utility.plotting import plot_loss_points
from pathlib import Path


from unet_model.training.train import epoch
import unet_model.unet as unet
from unet_model.training import training_datasets
from utility import user_interface
import re

from unet_model.losses import dice_loss

import torch
import torch.nn as nn

import datetime
import os

train_losses = []
val_losses = []

# def train_loop(model:unet.UNet=UNET_MODEL, loss_fn=LOSS_FN, 
#                optimizer=torch.optim.Adam, device=DEVICE_COMPUTE_PLATFORM, 
#                num_epochs=NUM_EPOCHS, num_class = 1,
#                val_dataloader=None, train_dataloader=None):

def train_loop(model=unet, pretrained_weights=None, loss_fn=nn.BCEWithLogitsLoss(), 
                optimizer=torch.optim.Adam, start_epoch=0, num_epochs=100, num_class = 1, saving_rate=5, 
                learning_rate=0.001, weights_save_folder_path = f"model_weights/nouveaux_dangling_endpoints_penalty_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}/",
                val_dataloader=None, train_dataloader=None,
                device='cpu'): # , image_path='training_data/image/', label_path='training_data/nouveaux_labels/'):
    


    if device == 'cpu':
        raise ValueError("CPU is not supported for training. \n Please use a GPU for training.")

    #loss_fn = nn.BCEWithLogitsLoss()
    model, optimizer = unet.initialize_model_training(pretrained_weights=pretrained_weights, num_class = num_class,
                                                      device=device, learning_rate=learning_rate)


    for epoch_ii in range(start_epoch,num_epochs):
        train_loss, val_loss = epoch(model, train_dataloader, val_dataloader, loss_fn, optimizer, device, epoch_ii, num_epochs, saving_rate, weights_save_folder_path)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
    

# def train(model:unet.UNet=MODEL_PARAMS, image_path=IMAGE_PATH, label_path=LABEL_PATH, 
#           loss_fn:callable=LOSS_FN, optimizer=torch.optim.Adam, 
#           device=DEVICE_COMPUTE_PLATFORM, num_epochs = NUM_EPOCHS, transforms=None):

def train(model:unet, loss_fn:callable=nn.BCEWithLogitsLoss(), optimizer=torch.optim.Adam, 
         num_epochs=100, image_transforms=t.training_transforms(target_resolution=256, prob_flip=1.0),
         pretrained_weights=None, num_class=1, saving_rate=5, learning_rate=0.001, 
         image_path = 'training_data/image/', label_path = 'training_data/nouveaux_labels/',
         weights_save_folder_path = f"model_weights/nouveaux_dangling_endpoints_penalty/{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}/", 
         training_split = 0.9, batch_size = 5, num_workers = 0, device='cpu'):
    
    if device == 'cpu':
        raise ValueError("CPU is not supported for training. \n Please use a GPU for training.")


    #NOTE: assuming weights file name contains epoch_XXX
    if pretrained_weights != None:
        if weights_save_folder_path == None:
            weights_save_folder_path = Path(pretrained_weights).parent
        start_epoch = int(re.search(r'(?<=epoch_)\d+',pretrained_weights)[0])
    else:
        start_epoch=0


    #create the folder to save the weights
    os.makedirs(weights_save_folder_path,exist_ok=True)
    # # I programatically put the timestamp 
    # # in the folder path name (I probably should change this but I don't know a better way for now)
    # # so I want the folder path to get set when the training starts and not change as the 
    # # time changes during training 
    # weights_save_folder_path = str(weights_save_folder_path)

    # Train the model 
    #transforms = training_transforms
    print(f'image path: { image_path}')
    train_dataloader, val_dataloader = training_datasets.get_train_data_loaders(image_path=image_path, label_path=label_path, 
                                                                                training_split = training_split, batch_size=batch_size,
                                                                                num_workers=num_workers, transform_generator=image_transforms)
    user_interface.startTrainingLogoPrint()
    print(f"Training the model using {loss_fn} and binarization enforcement with dangling endpoints penalty")
    #print(f"Training the model using {loss_fn}")

    train_loop(model=model, pretrained_weights=pretrained_weights, loss_fn=loss_fn, optimizer=optimizer, start_epoch = start_epoch,
                num_epochs=num_epochs, num_class = num_class, saving_rate=saving_rate, weights_save_folder_path = weights_save_folder_path,
                learning_rate=learning_rate, train_dataloader=train_dataloader, val_dataloader=val_dataloader, device=device)
    plot_loss_points(train_losses, val_losses)
    print("Training complete.")

# uncomment out to run from the command line
#train()