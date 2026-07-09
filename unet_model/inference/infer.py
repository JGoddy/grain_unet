#Standard Library Imports
import torch
import numpy as np
import os
from pathlib import Path
#Local Imports
from utility.settings import *
import utility.file_manager as fm
import unet_model.unet as unet
from tqdm import tqdm
import utility.transformation as t

# Function to run inference
def run_inference(model:unet.UNet, image:str|Path|np.ndarray|torch.Tensor, 
                  target_resolution = 256, image_transform = t.inference_transforms(256), device = 'mps')->np.ndarray:
    # Load and transform image
    if target_resolution != 256 and image_transform != t.inference_transforms(256):
        image_transform = t.inference_transforms(target_resolution)
    print("image:", image if isinstance(image, Path) else image.shape)
    if isinstance(image, torch.Tensor):
        input_tensor = image
    else:
        input_tensor = fm.load_image_tensor(image)
        print("input_tensor shape:", input_tensor.shape)
        print("device:", device)
        input_tensor = image_transform(input_tensor).unsqueeze(0).to(device) #* 255.0  # Add batch dimension and move to deviceto(device) 
    print("input_tensor", input_tensor.shape)
    # Perform inference and apply sigmoid to get probabilities
    with torch.no_grad():
    
        output = model(input_tensor)
        output = torch.sigmoid(output)
        output_np = output.squeeze().cpu().numpy()  # Remove batch dimension and move to CPU as a numpy array

    return output_np

# Main function to run single inference process
def single_inference(image_path:str|Path, output_path:str|Path, model:unet.UNet|str|Path, target_resolution = 256, image_transform = t.inference_transforms(256), device = None): 

    device = torch.device(device)

    if isinstance(model, unet.UNet):
        model = model
    elif isinstance(model, str|Path):   
        print("Running a single inference. loading model: ", model)
        model = unet.load_model_weights(model, device) 
    else:
        raise ValueError('The model parameter must be an instance of the UNet class or a path to a model weights file.')

    output_np = run_inference(model = model, image = image_path, target_resolution = target_resolution, image_transform = image_transform, device = device)
    print("output_path:", output_path)
    fm.save_output(output_np, output_path)
    #[REF] save only once, there's a bug here

def multi_inference(image_paths, output_paths, model:unet.UNet|str|Path, target_resolution = 256, image_transform = t.inference_transforms(256), device = None): #add model_pth for bulk export
 
    if isinstance(model, unet.UNet):
        model = model
    else:
        model = unet.load_model_weights(model, device = device)

    for image_path, output_path in tqdm(zip(image_paths, output_paths), total = len(image_paths), desc= 'Inference Progress'):
        if Path(image_path).name[0] == '.':
            continue

        single_inference(image_path=image_path, output_path=output_path, model=model, target_resolution=target_resolution, image_transform=image_transform, device=device)

        # these lines already occur in the single_inference function
        # output_np = run_inference(model, image_path, DEVICE_COMPUTE_PLATFORM)
        # fm.save_output(output_np, output_path)
# THIS IS REALLY SINGLE FOLDER INFERENCE
# TODO: CHANGE FUNCTION NAME
# TODO: multi_folder_inference and in_situ_inference currently contain the same code,
# # but the run_inference call multi_folder_inference if the mode is in_situ 
def multi_folder_inference(model_path=None, folder='', pattern:str="fov*/*.tif", 
    exclude = ['/.', 'trace'], include = [],
    target_resolution = 256, image_transform = t.inference_transforms(256), 
    prefix = None, device = None):

    ''' This function will take in the test data directory and create inferences for the different fovs of those images'''
    
    folder = Path(folder)
    
    model = unet.load_model_weights(model_path=model_path, device = device)
    print(f"Loaded model from {model_path}")
    
    print("Compute platform is: ", device)

    print(f"Looking for images in: {folder}/{pattern} excluding: {exclude}, including: {include}")
    image_paths = fm.get_file_names(folder, pattern = pattern, exclude = exclude, include = include)
    if len(list(image_paths)) == 0:
        raise ValueError('\n\nNo images found in the specified folder')
    else:
        print(f"Found {len(list(image_paths))} images")
    
    save_paths = [os.path.join(path.parent.parent, f'predict_{prefix}_{target_resolution}', f'predict_{path.name}') for path in image_paths]
    multi_inference(image_paths=image_paths, output_paths=save_paths, model=model,target_resolution=target_resolution,image_transform=image_transform,device=device)

def in_situ_inference(model_path=None, folder='', pattern=None, 
    exclude = ['trace'], include = [], 
    target_resolution=256, image_transform = t.inference_transforms(256),
    prefix = None, device = None):

    ''' This function will take in the test data directory and create inferences for the different fovs of those images'''
    folder = Path(folder)
    model = unet.load_model_weights(model_path=model_path, device = device)
    print(f"Loaded model from {model_path}")
    print("Compute platform is: ", device)
    print(f"Looking for images in: {folder}/{pattern} excluding: {exclude}, including: {include}")
    image_paths = fm.get_file_names(folder, pattern = pattern, exclude = exclude, include = include)
    
    if len(list(image_paths)) == 0:
        raise ValueError('\n\nNo images found in the specified folder')
    else:
        print(f"Found {len(list(image_paths))} images")
    #save_paths = [os.path.join(path.parent, f'predict_{prefix}_{target_resolution}', f'predict_{path.name}') for path in image_paths]
    #save_paths = [os.path.join(path.parent, f'compiled/{model_type}/{model_path.split("/")[-2]}_{model_path.split("/")[-1].split(".")[0]}',f"compiled_{path.name}") for path in image_paths]
    save_paths = [os.path.join(path.parent, 'inferenced', f'{prefix}',f"inference_{path.name}") for path in image_paths]

    multi_inference(image_paths = image_paths, output_paths = save_paths, model = model, target_resolution=target_resolution, image_transform=image_transform, device=device)

   # return os.path.join(folder, f'predict_{model_path}_{target_resolution}') #return the folder where the predictions are saved
