# from utility.settings import init, init_training
# init()
# init_training() 
#from utility.settings import *

from unet_model.training.train import epoch
import unet_model.unet as unet
from unet_model.training import training_datasets
from utility import user_interface
#from utility.transformation import training_transforms
import utility.transformation as t

from utility.plotting import plot_loss_points

from unet_model.losses import dice_loss

import torch
import torch.nn as nn

train_losses = []
val_losses = []

# def train_loop(model:unet.UNet=UNET_MODEL, loss_fn=LOSS_FN, 
#                optimizer=torch.optim.Adam, device=DEVICE_COMPUTE_PLATFORM, 
#                num_epochs=NUM_EPOCHS, num_class = 1,
#                val_dataloader=None, train_dataloader=None):

def train_loop(model=unet, loss_fn=nn.BCEWithLogitsLoss(), 
                optimizer=torch.optim.Adam, pretrained_weights=None,
                num_epochs=100, num_class = 1, saving_rate=5, learning_rate=0.001,
                val_dataloader=None, train_dataloader=None,
                device='cpu', image_path='training_data/image/', label_path='training_data/nouveaux_labels/'):
    
    
    if device == 'cpu':
        raise ValueError("CPU is not supported for training. \n Please use a GPU for training.")

    #loss_fn = nn.BCEWithLogitsLoss()
    model, optimizer = unet.initialize_model_training(pretrained_weights=pretrained_weights, num_class = num_class,
                                                      device=device, learning_rate=learning_rate)


    for epoch_ii in range(num_epochs):
        train_loss, val_loss = epoch(model, train_dataloader, val_dataloader, loss_fn, optimizer, device, epoch_ii, num_epochs, saving_rate)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
    

# def train(model:unet.UNet=MODEL_PARAMS, image_path=IMAGE_PATH, label_path=LABEL_PATH, 
#           loss_fn:callable=LOSS_FN, optimizer=torch.optim.Adam, 
#           device=DEVICE_COMPUTE_PLATFORM, num_epochs = NUM_EPOCHS, transforms=None):

def train(model:unet, image_path='training_data/image/', label_path='training_data/nouveaux_labels/', 
          loss_fn:callable=nn.BCEWithLogitsLoss(), optimizer=torch.optim.Adam, 
          device='cuda', num_epochs = 100, image_transforms=t.training_transforms(target_resolution=256, prob_flip=1.0)):
    
    if device == 'cpu':
        raise ValueError("CPU is not supported for training. \n Please use a GPU for training.")

    # Train the model 
    #transforms = training_transforms
    print(f'image path: { image_path}')
    train_dataloader, val_dataloader = training_datasets.get_train_data_loaders(image_path=image_path, label_path=label_path, transform_generator=image_transforms)
    user_interface.startTrainingLogoPrint()
    print(f"Training the model using {loss_fn} and dangling endpoints penalty")
    train_loop(model=model, loss_fn=loss_fn, optimizer=optimizer, device=device, num_epochs=num_epochs, train_dataloader=train_dataloader, val_dataloader=val_dataloader)
    #TODO: fix plot_loss_points function to not use MyUnetConfig.ini file
    plot_loss_points(train_losses, val_losses)
    print("Training complete.")

# uncomment out to run from the command line
#train()