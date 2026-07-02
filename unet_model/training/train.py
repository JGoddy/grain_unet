'''
Training process for Unet model
'''

__author__ = "Matthew Patrick, Lauren Grae, Rosnel Leyva-Cortes" 


import tqdm
import torch 
from utility.settings import *
from utility.plotting import *
import datetime
init_training()

#--------------------------------------TRAINING STEP------------------------------------------------------------------------


def epoch(model, train_dataloader, val_dataloader, loss_fn, optimizer, device, epoch):
    
    model.train(True) 
    train_losses = []
    val_losses = []

    tqdm_train_dataloader = tqdm.tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} - Training") 

    for images,labels, names in tqdm_train_dataloader: 

        print("images", images.shape)
        print("labels", labels.shape)
        loss, outputs = training_step(images, labels, model, loss_fn, optimizer, device)        
        train_losses.append(loss.item())


    #saves model params after a certain amount of epochs 
    if epoch % SAVING_RATE == 0: 
            torch.save(model.state_dict(), f'{MODEL_PARAMS.split(".")[0]}_{epoch}_{datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}.pth')
       

        #validation step
    tqdm_val_dataloader = tqdm.tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} - Validation")

    with torch.no_grad():
        for images, labels, names in tqdm_val_dataloader:
            loss, vlaidation_outputs = validation_step(images, labels, model, loss_fn, device)
            val_losses.append(loss.item())
    
    # Logging the metrics
    train_loss = sum(train_losses) / len(train_losses)
    val_loss = sum(val_losses) / len(val_losses)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
   
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}:")
    print(f"Training Loss: {train_loss}")
    print(f"Validation Loss: {val_loss}")

    

    return train_loss, val_loss


def training_step(batch_images, batch_labels, model, loss_fn, optimizer, device=DEVICE_COMPUTE_PLATFORM):
    model.train(True) 
    optimizer.zero_grad() 
    batch_images = batch_images.to(device)
    batch_labels = batch_labels.to(device)
    outputs = model.forward(batch_images)
    # what is the shape of outputs? 
    num_dangling_endpoints = num_dangling_endpoints(outputs)
    loss = loss_fn(outputs, batch_labels) + num_dangling_endpoints #maybe multiply by a constant?
    loss.backward()
    optimizer.step() 
    return loss, outputs

def validation_step(batch_images, batch_labels, model, loss_fn, device=DEVICE_COMPUTE_PLATFORM):
    model.train(False)  # Set the model to evaluation mode
    with torch.no_grad():
        model.train(False) 
        batch_images = batch_images.to(device)
        batch_labels = batch_labels.to(device)
        outputs = model.forward(batch_images)
        num_dangling_endpoints = num_dangling_endpoints(outputs)
        loss = loss_fn(outputs, batch_labels) + num_dangling_endpoints #maybe multiply by a constant?
        return loss, outputs

# I don't think this function gets used 
def save_model(model, num_epochs=-1, epoch=-1):
    torch.save(model.state_dict(),  f'{MODEL_PARAMS.split(".")[0]}_{epoch}.pth')
    torch.save(model.state_dict(),  f'{MODEL_PARAMS.split(".")[0]}_{num_epochs}.pth')




def num_dangling_endpoints(image):
    """
    Counts the number of dangling endpoints in the image.

    Args:
        image: The image to count the number of dangling endpoints in.

    Returns:
        The number of dangling endpoints in the image.
    """
   
    tracing_image_endpoints = [] 
    for (row,col), val in np.ndenumerate(tracing_image): # is this y, x or x, y?? 
        if val==0 and row>0 and col>0 and row<len(tracing_image)-1 and col<len(tracing_image)-1: 

            num_neighbors = sum((tracing_image[row+1,col]==0, tracing_image[row,col+1]==0,  
                    tracing_image[row-1,col]==0, tracing_image[row,col-1]==0, 
                    tracing_image[row+1,col+1]==0, tracing_image[row-1,col-1]==0, 
                    tracing_image[row+1,col-1]==0, tracing_image[row-1,col+1]==0))

            if num_neighbors == 1:
                tracing_image_endpoints.append((row,col))

            elif num_neighbors == 2:
                if ((tracing_image[row-1,col-1]==0 and tracing_image[row-1,col]==0) or
                (tracing_image[row-1,col-1]==0 and tracing_image[row,col-1]==0) or 

                (tracing_image[row-1,col+1]==0 and tracing_image[row-1,col]==0) or
                (tracing_image[row-1,col+1]==0 and tracing_image[row,col+1]==0) or

                (tracing_image[row+1,col-1]==0 and tracing_image[row+1,col]==0) or
                (tracing_image[row+1,col-1]==0 and tracing_image[row,col-1]==0) or

                (tracing_image[row+1,col+1]==0 and tracing_image[row+1,col]==0) or
                (tracing_image[row+1,col+1]==0 and tracing_image[row,col+1]==0)):

                        tracing_image_endpoints.append((row,col))

            elif num_neighbors == 3:
                if ((tracing_image[row-1,col-1]==0 and tracing_image[row-1,col]==0 and tracing_image[row,col-1]==0) or
                
                (tracing_image[row-1,col+1]==0 and tracing_image[row-1,col]==0 and tracing_image[row,col+1]==0) or

                (tracing_image[row+1,col-1]==0 and tracing_image[row+1,col]==0 and tracing_image[row,col-1]==0) or

                (tracing_image[row+1,col+1]==0 and tracing_image[row+1,col]==0 and tracing_image[row,col+1]==0)):

                        tracing_image_endpoints.append((row,col))   
               
    return len(tracing_image_endpoints)




