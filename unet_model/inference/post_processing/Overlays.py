
from skimage import  io
import numpy as np
import warnings
import configparser
from pathlib import Path
from utility import file_manager as fm
import utility.transformation as t

#from utility.settings import *
#load params
config = configparser.ConfigParser()
config.read('MyUnetConfig.ini')
from utility.settings import *

# PREDICT_DATA_DIR = config['INFERENCE_PARAMS']['OVERLAY_DATA_DIR']
# TARGET_RESOLUTION = int(config['INFERENCE_PARAMS']['OUTPUT_RESOLUTION'])


#TODO: I think this function is funky. 
# what is the point of checking if the first element is a Path or str?
def compile_imgs(imgs, compilation='min', **kwargs):
    '''Compiles grayscale images

    imgs - a numpy depth-wise stack of images (i.e. shape (512,512,n))
    compilation - compilation technique. 'min' or 'max' supported
    '''
    imgs_tmp = []
    if isinstance(imgs[0], (Path, str)):
        imgs = stack_images(imgs)
    elif isinstance(imgs[0], (np.ndarray, torch.Tensor)):
        return imgs # see note in stack_images
    if compilation == 'min':
        img_compiled = np.amin(imgs, axis=-1)
    elif compilation == 'max':
        img_compiled = np.amax(imgs, axis=0)
    elif compilation == 'sum':

        img_compiled = np.add.reduce(imgs, axis=-1)
        img_compiled = ((img_compiled / img_compiled.max()))*255

    else:
        warnings.warn(f'Compilation technique \'{compilation}\' not recognized', stacklevel=2)
        img_compiled = imgs[:,:,0]

    return img_compiled

def stack_images(fnames:list[str|Path], target_resolution:int=256, 
                 image_transform = t.inference_transforms(256))->np.ndarray:
    '''Stacks images
    imgs(list[str]) - a list of image file names
    returns: a numpy array of stacked images
    '''
    ii=0
    if len(fnames) == 0:
        raise ValueError('No images to stack')
    if target_resolution != 256 and image_transform != t.inference_transforms(256):
        image_transform = t.inference_transforms(target_resolution)

    # This function currently only supports stacking 
    # image that it loads from file names, NOT 
    # numpy arrays or torch tensors. TODO: fix this.
    # for now, assume that if the image is already loaded 
    # as a numpy array or torch tensor, it is already stacked so 
    # it is length 1 and should just be returned. 
    for fname in (fnames):
        if isinstance(fname, (np.ndarray, torch.Tensor)):
            predictions = predictions = fm.load_image_tensor(fname)
            #predictions = image_transform(predictions).squeeze()
            break
        print(fname)
        if Path(fname).stem[0] == '.':
            continue
        if ii == 0: 
            #predictions = io.imread(fname)
            predictions = fm.load_image_tensor(fname)
            predictions = image_transform(predictions).squeeze() #.unsqueeze(0)
            ii+=1
        else:
            #img = io.imread(fname)
            img = fm.load_image_tensor(fname)
            img = image_transform(img).squeeze() #.unsqueeze(0)
            print("img shape:", img.shape)

            predictions = np.dstack((predictions,img))

    return predictions

def overlay(fnames:list[str|Path], compilation:str = 'min')->np.ndarray:
    stacked = stack_images(fnames)
    return compile_imgs(stacked)

def overlay_fov_generator(folder:str, pattern:str):
    
    #Find all the fov folders in the top-level folder
    print("folder", folder)
    print("pattern", pattern)
    predicted_fovs_folders = fm.list_fovs(folder, pattern)
    print("predicted_fovs_folders", predicted_fovs_folders)

    for fov_folder in predicted_fovs_folders:
        #for each folder, find the predictions *.png files, stack, and compile. Append to a master list
        
        fnames = fm.get_file_names(fov_folder, pattern = "*.png", exclude=['post_process', 'compiled'])
        compiled_img = compile_imgs(fnames, compilation = args_pp['compilation'])
    
        yield {'fov_folder':fov_folder,'img': compiled_img, 'fname':fnames[0]}


# if __name__ == '__main__':



#     args = { # for post processing 
#             'compilation': 'min',
#             'n_dilations': 3,
#             'liberal_thresh': 161,
#             'conservative_thresh': 212,
#             'invert_double_thresh': True,
#     }


#     from skimage import io

#     image_formats = ('.tif', '.png')  # Image formats to look for
#     #folder = PREDICT_DATA_DIR
#     try:
#         base_folder = sys.argv[1]
#     except:
#         base_folder = PREDICT_DATA_DIR
#     pattern     = '*'
#     for FOV_predictions in  Path(base_folder).glob(pattern):
#         #print(str(FOV_predictions))
#         fnames = []
#         if FOV_predictions.is_dir():
#             #print('is dir')
#             for image in FOV_predictions.glob('*tif'):
#                 print(str(image))
#                 fnames.append(str(Path(image)))
#         for ii, fname in enumerate(fnames):
#             print(fname)
#             if ii == 0: 
#                 predictions = io.imread(fname)
#                 #print(fname)
#             else:
#                 print(str(fname))
#                 img = io.imread(fname)
#                 predictions = np.dstack((predictions,img))
#         compiled = compile_imgs(predictions)
#         save_path_comp = os.path.join(FOV_predictions,f'compiled_{Path(fname).name}.png')
#         save_path_post = os.path.join(FOV_predictions,f'postprocess_{Path(fname).name}.png')
#         save_output(compiled*255,save_path_comp)
#         post_processed = pp(predictions, **args)
#         save_output(post_processed, save_path_post)
       