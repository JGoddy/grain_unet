'''
Training process for Unet model
'''

__author__ = "Matthew Patrick, Lauren Grae, Rosnel Leyva-Cortes" 


import tqdm
import torch 
import torch.nn.functional as F
import numpy as np
from utility.settings import *
from utility.plotting import *
import datetime
init_training()
import matplotlib.pyplot as plt

#--------------------------------------TRAINING STEP------------------------------------------------------------------------


def epoch(model, train_dataloader, val_dataloader, loss_fn, optimizer, device, epoch, num_epochs, saving_rate, weights_save_folder_path):
    
    model.train(True) 
    train_losses = []
    val_losses = []

    tqdm_train_dataloader = tqdm.tqdm(train_dataloader, desc=f"Epoch {epoch}/{num_epochs} - Training") 

    for images,labels, names in tqdm_train_dataloader: 

        #NOTE: The training and validation outputs are not being used so they don't have to be returned 
        # print("images", images.shape)
        # print("labels", labels.shape)
        #loss, outputs = training_step(images, labels, model, loss_fn, optimizer, device)    

        train_loss = training_step(images, labels, model, loss_fn, optimizer, device) 
        train_losses.append(train_loss.item())


    #saves model params after a certain amount of epochs 
    if epoch % saving_rate == 0: 
        ##TODO: think about whether to save as epoch (python starts from 0)
        # or epoch + 1 (to start from 1, as in the print statements)
        # after the first (zeroth) epoch, the model saves 
        #torch.save(model.state_dict(), f'{MODEL_PARAMS.split(".")[0]}_{epoch}_{datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}.pth')

        torch.save(model.state_dict(),f"{weights_save_folder_path}/epoch_{epoch}.pth")

        #validation step
    tqdm_val_dataloader = tqdm.tqdm(val_dataloader, desc=f"Epoch {epoch}/{num_epochs} - Validation")

    with torch.no_grad():
        for images, labels, names in tqdm_val_dataloader:
            #val_loss, validation_outputs = validation_step(images, labels, model, loss_fn, device)
            val_loss = validation_step(images, labels, model, loss_fn, device)

            val_losses.append(val_loss.item())
    
    # Logging the metrics
    train_loss = sum(train_losses) / len(train_losses)
    val_loss = sum(val_losses) / len(val_losses)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
   
    print(f"Epoch {epoch+1}/{num_epochs}:")
    print(f"Training Loss: {train_loss}")
    print(f"Validation Loss: {val_loss}")

    

    return train_loss, val_loss


def training_step(batch_images, batch_labels, model, loss_fn, optimizer, device='cpu'):
    if device == 'cpu':
        raise ValueError("CPU is not supported for training. \n Please use a GPU for training.")
    model.train(True) 
    optimizer.zero_grad() 
    batch_images = batch_images.to(device)
    batch_labels = batch_labels.to(device)
    outputs = model.forward(batch_images)
    # what is the shape of outputs? 
    #num_dangling_endpoints = num_dangling_endpoints(outputs)
    #are_there_dangling_endpoints = are_there_dangling_endpoints(outputs)
    #loss = loss_fn(outputs, batch_labels)

    #NOTE: hardcoded for these 1 channel images, outputs (from dataloader) has shape 
    # (batch_size, 1, image_width, image_height) 
    # or maybe (batch_size, 1, image_height, image_width)
    # I'm not sure which is the height and which is the width since 
    # the images are square so the height and width are the same
    #TODO: check this with Wayne's validation data, for example, 
    # since those images are not square


    #NOTE:The dangling enpoints check doesn't work 
    # unless the image is binary pixel values all 0 and 1 (or 255)
    #because otherwise the boundary is not all completely black
    # 
     
    # binarization_loss_factor = 1 
    # dangling_endpoints_loss_factor=1

    #outputs[outputs !=0] = 1
    #outputs[outputs !=1] = 0

    #loss_factor = 1
   # binarized_outputs = (outputs > 0.5).float().detach().cpu() 
   # print("starting loop ...")
    for i in range(outputs.shape[0]):
         
        # if np.shape(np.where(outputs[i,0].detach().cpu()==0.0))[1]+np.shape(np.where(outputs[i,0].detach().cpu()==1.0))[1] < outputs[i,0].flatten().shape[0]:
        #     # loss = 10*loss_fn(outputs,batch_labels)
        #     # break
        #     binary = False 
        #     loss_factor=5
            
        # else: 
        #     binary = True

        # elif np.max(outputs[i,0]) <= 1.0: 
        #     if np.shape(np.where(outputs[i,0]==0.0))[1]+np.shape(np.where(outputs[i,0]==1.0))[1] < outputs[i,0].flatten().shape():
        #         loss = 10*loss_fn(outputs,batch_labels)
        #         break     
        # print("*"*20)
        # print("interation:", i)
        # print("maximum outputs value:" ,torch.max(outputs[i,0]))
        # print("minimum outputs value:", torch.min(outputs[i,0]))

        # print("maximum labels value:", torch.max(batch_labels[i,0]))
        # print("minimum labels value:", torch.min(batch_labels[i,0]))

        # print("individual loss:", loss_fn(outputs[i,0], batch_labels[i,0]))
        # print("batch loss:", loss_fn(outputs, batch_labels))
        # print("*"*20)


        # if binary and ... 
        # print("*"*20)
        # print("iteration:",i)
        # # print("max output:",torch.max(outputs[i,0]))
        # print("min output:",torch.min(outputs[i,0]))
        # print("max binarized output:",torch.max(binarized_outputs[i,0]))
        # print("min binarized output:",torch.min(binarized_outputs[i,0]))
        # print("this iteration loss function:", loss_fn(outputs[i,0], batch_labels[i,0]))
        # print("loss function:", loss_fn(outputs,batch_labels))
        #print("starting dangling endpoint check")
        #if torch.max(outputs[i,0]) > 0.8 and are_there_dangling_endpoints(outputs[i,0].detach().cpu() ):


        # if torch.min(outputs[i,0])<0.9:
        #     threshold = 0.95
        #elif torch.min(outputs[i,0])<0.999
        #   threshold = 0.95
        #else:
        #   threshold=1.0

        ### START HERE ###
        # if threshold == 0.5: 
            # if torch.max(outputs[i,0]) < 0.99 and torch.min(outputs[i,0]) <= 0.5: 
            #   threshold = 0.5
            # elif torch.max(outputs[i,0]) < 0.99 and torch.min(outputs[i,0]) < 0.8: 
            #   threshold = 0.8
            # elif torch.where(outputs[i,0]==0.0))[0].shape[0]+torch.where(outputs[i,0]==1.0))[0].shape[0] == outputs[i,0].flatten().shape[0]:
            #   threshold = 1.0 #although in this case, any threshold >0 is the same, so maybe this is unnecessary? 
            # elif torch.max(outputs[i,0]) > 0.9999 and torch.min(outputs[i,0]) < 0.5:
            #   threshold = 0.99
            # elif torch.max(outputs[i,0]) > 0.999 and torch.min(outputs[i,0]) < 0.6: 

            # for epoch 10, the threshold should be 0.75 
            # ZOOM IN ON THAT AREA AND TEST THIS 
            # ANY WHY IT IS TRUE FOR THRESHOLD = 0.85 BUT FALSE FOR 0.8
        # 


       


        if are_there_dangling_endpoints_test(outputs[i,0]):

        
         #   print("There are dangling endpoints")
            loss = 10*loss_fn(outputs, batch_labels)
            break
           # dangling_endpoints = True
           # loss_factor*=2
        else:
          #  print("There are not dangling endpoints")
            if i == outputs.shape[0]-1:
           #     print("final iteration of batch")
                loss = loss_fn(outputs,batch_labels)
            #dangling_endpoints = False
       # print("*"*20)
        # if binary == False or dangling_endpoints == True:
        #     break

        # else:
        #     loss = loss_fn(outputs, batch_labels)

    #loss = loss_fn(outputs, batch_labels) #*loss_factor  #+ num_dangling_endpoints #maybe multiply by a constant?
  #  print("ended loop")
    loss.backward()
    optimizer.step() 
    return loss  #, outputs

def validation_step(batch_images, batch_labels, model, loss_fn, device='cpu'):
    if device == 'cpu':
        raise ValueError("CPU is not supported for training. \n Please use a GPU for training.")

    model.train(False)  # Set the model to evaluation mode
    with torch.no_grad():
        model.train(False) 
        batch_images = batch_images.to(device)
        batch_labels = batch_labels.to(device)
        outputs = model.forward(batch_images)
        
        #NOTE: see note in training_step

        #outputs[outputs !=0] = 1 
        #outputs[outputs != 1] = 0

        # binarization_loss_factor = 1 
        # dangling_endpoints_loss_factor=1

        #loss_factor = 1

        #outputs = (outputs > 0.5).float()

        #binarized_outputs = (outputs > 0.5).float().detach().cpu() 

        for i in range(outputs.shape[0]):
            # if np.shape(np.where(outputs[i,0].detach().cpu()==0.0))[1]+np.shape(np.where(outputs[i,0].detach().cpu()==1.0))[1] < outputs[i,0].flatten().shape[0]:
            #     binary = False 
            #     loss_factor=5
            
            # else: 
            #     binary = True

            #if torch.max(outputs[i,0]) > 0.8 and are_there_dangling_endpoints(outputs[i,0].detach().cpu()):
            if are_there_dangling_endpoints_test(outputs[i,0]):
                loss = 10*loss_fn(outputs, batch_labels)
                break
                # dangling_endpoints = True
                # loss_factor*=2
            else:
                if i == outputs.shape[0]-1:
                    loss = loss_fn(outputs,batch_labels)
                #dangling_endpoints = False
           
        #     if binary == False or dangling_endpoints == True:
        #         break
           
        # loss = loss_fn(outputs, batch_labels)*loss_factor

        return loss #, outputs
                

    # #loss = loss_fn(outputs, batch_labels) + num_dangling_endpoints #maybe multiply by a constant?

# I don't think this function gets used 
#TODO: figure out what the MODEL_PARAMS terms is and how else to generate it 
def save_model(model, num_epochs=-1, epoch=-1):
    torch.save(model.state_dict(),  f'{MODEL_PARAMS.split(".")[0]}_{epoch}.pth')
    torch.save(model.state_dict(),  f'{MODEL_PARAMS.split(".")[0]}_{num_epochs}.pth')


def are_there_dangling_endpoints(image):
    """
    Checks if there are any dangling endpoints in the image.
    """
    for (row,col), val in np.ndenumerate(image): 
        if val<=0.5 and row>0 and col>0 and row<len(image)-1 and col<len(image)-1: 

            num_neighbors = sum((image[row+1,col]<=0.5, image[row,col+1]<=0.5,  
                    image[row-1,col]<=0.5, image[row,col-1]<=0.5, 
                    image[row+1,col+1]<=0.5, image[row-1,col-1]<=0.5, 
                    image[row+1,col-1]<=0.5, image[row-1,col+1]<=0.5))

            if num_neighbors == 1:
                return True
          
            elif num_neighbors == 2:
               
                if ((image[row-1,col-1]<=0.5 and image[row-1,col]<=0.5) or
                    (image[row-1,col-1]<=0.5 and image[row,col-1]<=0.5) or 

                    (image[row-1,col+1]<=0.5 and image[row-1,col]<=0.5) or
                    (image[row-1,col+1]<=0.5 and image[row,col+1]<=0.5) or

                    (image[row+1,col-1]<=0.5 and image[row+1,col]<=0.5) or
                    (image[row+1,col-1]==0 and image[row,col-1]<=0.5) or

                    (image[row+1,col+1]<=0.5 and image[row+1,col]<=0.5) or
                    (image[row+1,col+1]<=0.5 and image[row,col+1]<=0.5)):

                        return True
                   
            elif num_neighbors == 3:
               if ((image[row-1,col-1]<=0.5 and image[row-1,col]<=0.5 and image[row,col-1]<=0.5) or
                
                (image[row-1,col+1]<=0.5 and image[row-1,col]<=0.5 and image[row,col+1]<=0.5) or

                (image[row+1,col-1]<=0.5 and image[row+1,col]<=0.5 and image[row,col-1]<=0.5) or

                (image[row+1,col+1]<=0.5 and image[row+1,col]<=0.5 and image[row,col+1]<=0.5)):

                        return True
           
    return False


def num_dangling_endpoints(image):
    """
    Counts the number of dangling endpoints in the image.

    Args:
        image: The image to count the number of dangling endpoints in.

    Returns:
        The number of dangling endpoints in the image.
    """
   
    image_endpoints = [] 
    for (row,col), val in np.ndenumerate(image): 
        if val==0 and row>0 and col>0 and row<len(image)-1 and col<len(image)-1: 

            num_neighbors = sum((image[row+1,col]==0, image[row,col+1]==0,  
                    image[row-1,col]==0, image[row,col-1]==0, 
                    image[row+1,col+1]==0, image[row-1,col-1]==0, 
                    image[row+1,col-1]==0, image[row-1,col+1]==0))

            if num_neighbors == 1:
                image_endpoints.append((row,col))

            elif num_neighbors == 2:
                if ((image[row-1,col-1]==0 and image[row-1,col]==0) or
                (image[row-1,col-1]==0 and image[row,col-1]==0) or 

                (image[row-1,col+1]==0 and image[row-1,col]==0) or
                (image[row-1,col+1]==0 and image[row,col+1]==0) or

                (image[row+1,col-1]==0 and image[row+1,col]==0) or
                (image[row+1,col-1]==0 and image[row,col-1]==0) or

                (image[row+1,col+1]==0 and image[row+1,col]==0) or
                (image[row+1,col+1]==0 and image[row,col+1]==0)):

                        image_endpoints.append((row,col))

            elif num_neighbors == 3:
                if ((image[row-1,col-1]==0 and image[row-1,col]==0 and image[row,col-1]==0) or
                
                (image[row-1,col+1]==0 and image[row-1,col]==0 and image[row,col+1]==0) or

                (image[row+1,col-1]==0 and image[row+1,col]==0 and image[row,col-1]==0) or

                (image[row+1,col+1]==0 and image[row+1,col]==0 and image[row,col+1]==0)):

                        image_endpoints.append((row,col))   
               
    return len(image_endpoints)


# Create this once (outside the function)
_NEIGHBOR_KERNEL = torch.tensor(
    [[1, 1, 1],
     [1, 0, 1],
     [1, 1, 1]],
    dtype=torch.float32
)

_KERNEL_CACHE = {}

def get_kernel(device):
    if device not in _KERNEL_CACHE:
        _KERNEL_CACHE[device] = _NEIGHBOR_KERNEL.to(device)
    return _KERNEL_CACHE[device]

def are_there_dangling_endpoints_test(tensor,threshold=0.95):
    """
    Detect dangling endpoints in a binary segmentation.

    Parameters
    ----------
    image : (H,W) torch.Tensor
        Tensor on either CPU or GPU.
        Pixels <= threshold are treated as boundary pixels.

    Returns
    -------
    bool
    """


    mask = tensor <= threshold

    # Count 8-connected neighbors
    kernel = _NEIGHBOR_KERNEL.to(tensor.device)

    kernel = get_kernel(tensor.device)

    neighbor_count = F.conv2d(
        mask.float()[None, None],
        kernel[None, None],
        padding=1
    )[0, 0]

    # Ignore image border (same as original implementation)
    C = mask[1:-1, 1:-1]

    N  = mask[:-2, 1:-1]
    NE = mask[:-2, 2:]
    E  = mask[1:-1, 2:]
    SE = mask[2:, 2:]
    S  = mask[2:, 1:-1]
    SW = mask[2:, :-2]
    W  = mask[1:-1, :-2]
    NW = mask[:-2, :-2]

    nc = neighbor_count[1:-1, 1:-1]

    #
    # One-neighbor endpoint
    #
    dangling1 = C & (nc == 1)

    #
    # Two-neighbor endpoint patterns
    #
    northwest_tip = (NW & N) | (NW & W)
    northeast_tip = (NE & N) | (NE & E)
    southwest_tip = (SW & S) | (SW & W)
    southeast_tip = (SE & S) | (SE & E)

    cond2 = (
        northwest_tip |
        northeast_tip |
        southwest_tip |
        southeast_tip
    )

    dangling2 = C & (nc == 2) & cond2

    #
    # Three-neighbor endpoint patterns
    #
    northeast_corner = N & NE & E #N & NW & NE
    northwest_corner = N & NW & W  #E & NE & SE
    southeast_corner = S & SE & E #S & SW & SE
    southwest_corner  = S & SW & W #W & NW & SW

    cond3 = (
        northeast_corner |
        northwest_corner |
        southeast_corner |
        southwest_corner
    )

    dangling3 = C & (nc == 3) & cond3

    return torch.any(dangling1 | dangling2 | dangling3)