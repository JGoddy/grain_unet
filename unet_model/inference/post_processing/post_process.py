'''
kwargs:
    'img_compiled': the compiled image
    'n_dilations': (default 3) Number of dilations to apply in closing
    'min_grain_area': (default 100) Max size of a hole to close
    'prune_size': (default 30) Size to prune with plantcv
    'convert_to_trans': (default True) convert the image to transparent
    'invert_double_thresh': (default True) Changes < to > in double threshold
    'conservative_thresh': (default 160) conservative threshold for double threshold
    'liberal_thresh': (default 200) liberal threshold for double threshold

    'out_dict': (default False) return a dict with all the intermediate steps
    'debug': (default False) print debug information
        
    'compilation': (default 'min') defines the image compilation technique #TODO: USE THIS

'''

#Standard Library Imports
import sys, os
import numpy as np
from pathlib import Path
from skimage import morphology, io
from plantcv import plantcv as pcv
from tqdm import tqdm
import matplotlib.pyplot as plt

#Local Imports
import utility.file_manager as fm
from unet_model.inference.post_processing.thresholding import double_thresh
from unet_model.inference.post_processing import Overlays
from utility.settings import *
import utility.transformation as t


# Test inline comment #min_grain_area=100, prune_size=0
# n_dilations = 3
# default values are for Wayne validation data
# min_grain_area=70, prune_size=50, 
def post_process(img_compiled, n_dilations=3, min_grain_area=0, prune_size=5, 
         convert_to_trans = True, invert_double_thresh=True, 
        conservative_thresh=160, liberal_thresh=200, out_dict=False, debug=False, **kwargs):
    '''This tries to make clean skeletons with N Unet output image(s) from an FOV
    '''
    print("min_grain_area:", min_grain_area)
    print("prune_size:", prune_size)

    print("inside post_process, compile:", compile)
    print("img_compiled shape:", img_compiled.shape)
    #if len(imgs.shape) > 2 and compile:
    # if compile:
    #     print("compiling images")
    #     img_compiled = Overlays.compile_imgs(imgs, target_resolution=target_resolution, image_transform=t.inference_transforms(target_resolution), **kwargs)
    # else:
    #     img_compiled = imgs
    #     print("not compiling images")

    #print("img_compiled", img_compiled[0])
    #print("img_compiled shape", img_compiled.shape)
    print("Double thresholding")
    
    # double_tresh returns white lines on black background
    # since the labels are black on white, we need to invert the double threshold

    if np.max(img_compiled) <= 1: # assuming min value is 0 and max value is 1, if not, then convert to 0-255 scale
        img_compiled = img_compiled * 255.0
    img_double_thresh = double_thresh(img_compiled, invert_double_thresh=invert_double_thresh, conservative_thresh=conservative_thresh, liberal_thresh=liberal_thresh, **kwargs)
    
    img_dilated = np.copy(img_double_thresh)
    print("Dilating")
    for _ in range(n_dilations):
        img_dilated = morphology.binary_dilation(img_dilated)
    img_closed = morphology.remove_small_holes(img_dilated, area_threshold=min_grain_area)

    print("Skeletonizing")
    skeleton = morphology.skeletonize(img_closed)
    print("Pruning")
    pruned_skeleton, _, _ = pcv.morphology.prune(skeleton.astype('uint8'), prune_size)
    #print("pruned_skeleton", pruned_skeleton)
    if out_dict:
        out_dict = {'compiled': img_compiled,
                'double_thresh': img_double_thresh,
                'conservative_thresh': conservative_thresh,
                'liberal_thresh': liberal_thresh,
                'dilated': img_dilated,
                'closed': img_closed,
                'skeleton': skeleton,
                'pruned_skeleton': pruned_skeleton
                }
        #print("out_dict", out_dict)       
    # if convert_to_trans:
    #     pruned_skeleton = convert_black_to_transparent(pruned_skeleton)
    else:
        out_dict = None

        # convert pruned_skeleton from black on white to white on black
    return [-1.0*(pruned_skeleton-1.0), out_dict]


def bulk_compile_and_pp_single_fov(pattern=f'fov*/predict_{PREFIX}_{TARGET_RESOLUTION}/', folder=PREDICT_DATA_DIR, post_process_option=True):
    """
    Compiles predictions from multiple images and optionally applies post-processing.
    This function is for when the images are in a single (fov) folder.

    Parameters:
    pattern (str): The pattern to match prediction directories.
    folder (str): The top-level directory containing the FoVs you wish to post-process.
    post_process_option (bool): Whether to apply post-processing to the compiled images.
    """
    os.makedirs(os.path.join(folder, 'post_processed'), exist_ok=True)

    images = fm.get_file_names(folder, pattern = '[!.]*.png')

    for image in images:
        save_path_comp = os.path.join(folder,f'post_process/compiled_{Path(image).stem}.png')
        save_path_post = os.path.join(folder,f'post_process/postprocess_{Path(image).stem}.png')

        #fm.save_output(compiled_img, save_path_comp)
        post_processed = post_process(image, **args_pp)
        fm.save_output(post_processed, save_path_post) 


def bulk_compile_and_pp(pattern=f'fov*/predict_{PREFIX}_{TARGET_RESOLUTION}/', folder=PREDICT_DATA_DIR, post_process_option=True):
    """
    Compiles predictions from multiple images and optionally applies post-processing.

    Parameters:
    pattern (str): The pattern to match prediction directories.
    folder (str): The top-level directory containing the FoVs you wish to post-process.
    post_process_option (bool): Whether to apply post-processing to the compiled images.

    Returns:
    None
    """

    print("pattern:", pattern)
    print("folder:", folder)
    print("post_process_option:", post_process_option)

    compiled_fovs = Overlays.overlay_fov_generator(folder, pattern)
   
    print("compiled_fovs:", compiled_fovs)


    for compiled_fov in compiled_fovs:
        print("compiling fov")
        fov_predictions_folder, compiled_img, fname = compiled_fov['fov_folder'], compiled_fov['img'], compiled_fov['fname']

        print("Saving compiled image")
        os.makedirs(os.path.join(fov_predictions_folder, 'post_process'), exist_ok=True)
        save_path_comp = os.path.join(fov_predictions_folder,f'post_process/compiled_{Path(fname).stem}.png')
        print("Saving compiled image")
        save_path_post = os.path.join(fov_predictions_folder,f'post_process/postprocess_{Path(fname).stem}.png')
        
        fm.save_output(compiled_fov['img'], save_path_comp)
        print("post_process_option", post_process_option)
        if post_process_option:
            post_processed = post_process(compiled_img, **args_pp)
            print("Saving post-processed image")
            fm.save_output(post_processed, save_path_post)
            
            if FINAL_OUTPUT_DIRECTORY:
                final_save_dir = os.path.join(FINAL_OUTPUT_DIRECTORY, f'predict_{PREFIX}_{TARGET_RESOLUTION}')
                if not os.path.isdir(final_save_dir):
                    os.makedirs(final_save_dir, exist_ok=True)
                save_path_final_output = os.path.join(final_save_dir,f'postprocess_{Path(fname).stem}.png')
                fm.save_output(post_processed, save_path_final_output)

def in_situ_post_process(in_folder, out_folder, folders_pattern = "fov*/predict/", folders_exclude = [], folders_include = [],
    postprocess_pattern = '[!.]*.png', postprocess_exclude = ['trace'], postprocess_include = [],
    compile=True, invert_double_thresh=True, conservative_thresh=160, liberal_thresh=200, integration = 3, out_dict=False,
    target_resolution = 256, image_transform = t.inference_transforms(256)):
 
    print("in_folder", in_folder)
    print("folders_pattern", folders_pattern)
    print("folders_exclude", folders_exclude)
    print("folders_include", folders_include)
    print("postprocess_pattern", postprocess_pattern)
    print("postprocess_exclude", postprocess_exclude)
    print("postprocess_include", postprocess_include)

    # combine the images in each folder instead of a fixed number of images
    if integration == "folders":
        folders = fm.list_fovs(in_folder, pattern = folders_pattern, exclude = folders_exclude, include = folders_include)
        if len(folders) == 0:
            raise ValueError('No folders found in the specified folder')
        elif len(folders) > 1:
            print("there are", len(folders), "folders to post-process:")
        else:
            print("there is only one folder to post-process")
        print("******")
        print("folder(s):")
        print(folders)
        print("******")
        #fovs = {}
        for folder in folders:
            images = fm.get_file_names(folder, pattern = postprocess_pattern, exclude = postprocess_exclude, include = postprocess_include)
            if len(images) == 0:
                raise ValueError('No images found in the specified folder')
            images = [str(image) for image in images]
            images.sort()
           # fovs[folder] = images
            if len(images) > 1:
                print("there are", len(images), "images to post-process in folder:", folder)
            else:
                print("there is only one image to post-process in folder:", folder)
            print("images", images)
            print("******")
            image = images[0]

            # TODO: ADD THE NUMBER OF EPOCHS TO THE SAVE PATH
            if compile and len(images) > 1:
                img_compiled = Overlays.compile_imgs(images, target_resolution=target_resolution, image_transform=image_transform, **args_pp)

                save_and_post_process(images[0],img_compiled, out_folder, 
                    invert_double_thresh=invert_double_thresh, 
                    conservative_thresh=conservative_thresh,
                    liberal_thresh=liberal_thresh,
                    out_dict=out_dict)

            else:
                for image in images:
                    img = plt.imread(image)
                    save_and_post_process(image,img, out_folder, 
                        invert_double_thresh=invert_double_thresh, 
                        conservative_thresh=conservative_thresh,
                        liberal_thresh=liberal_thresh,
                        out_dict=out_dict)

    else:
        images = fm.get_file_names(in_folder, pattern = postprocess_pattern, exclude = postprocess_exclude, include = postprocess_include)
        images = [str(image) for image in images]
        
        if len(images) == 0:
            raise ValueError('No images found in the specified folder')
        elif len(images) > 1: 
            print("there are", len(images), "images to post-process")
            print("images", images)
        else:
            print("there is only one image to post-process")
        images.sort()

        
    # if integration is "folders" 
    # then the number to loop over for ii is the number of folders, 


        print("integration", integration)
        for ii in tqdm(range(0, len(images), integration), desc='Post-processing', total=len(images)//integration):
            if integration > 1: # TODO: is this necessary or will integration=1 take care of this?
                print(f"combining {integration} images")
                image_compiled_path = images[ii] #TODO: did not test renaming 'image' to 'image_path'

                if ii + integration > len(images):
                    break
                img = []
                for jj in range(integration):
                    img.append(images[ii + jj])
                print("img", img)
                # TODO: ADD THE NUMBER OF EPOCHS TO THE SAVE PATH
                img_compiled = Overlays.compile_imgs(img, target_resolution=target_resolution, image_transform=image_transform, **args_pp)
                # save_path_comp = os.path.join(out_folder,f'compiled_{Path(image).stem}_95_512.png')
                # save_path_post = os.path.join(out_folder,f'postprocess_{Path(image).stem}_95_512.png')

                
            else:
                image_compiled_path = images[ii]
                img_compiled = io.imread(image_compiled_path)
                # save_path_comp = os.path.join(f"{out_folder}/compiled",f'compiled_{Path(img_compiled_path).stem}_95.png')
                # save_path_post = os.path.join(f"{out_folder}/postprocessed",f'postprocess_{Path(img_compiled_path).stem}_95.png')

        # TODO: fix so that if integration >1, save_and_post_process
        # is only called once at the end of the loop, but if 
        # integration = 1, then it is called for each image.
        #print("img_compiled", img_compiled)
        save_and_post_process(image_compiled_path,img_compiled, out_folder, 
            invert_double_thresh=invert_double_thresh, 
            conservative_thresh=conservative_thresh,
            liberal_thresh=liberal_thresh,
            out_dict=out_dict)

def save_and_post_process(image,img_compiled, out_folder, invert_double_thresh=True, conservative_thresh=160, liberal_thresh=200, out_dict=False):
    
    save_path_comp = os.path.join(out_folder,f'compiled_{Path(image).stem}_95_512.png')
    save_path_post = os.path.join(out_folder,f'postprocess_{Path(image).stem}_95_512.png')
    print("save_path_comp", save_path_comp)
    print("save_path_post", save_path_post)
    fm.save_output(img_compiled, save_path_comp)
    #print("inside in_situ_post_process, compile:", compile)
    post_processed = post_process(img_compiled, 
    invert_double_thresh=invert_double_thresh,
    conservative_thresh=conservative_thresh,
    liberal_thresh=liberal_thresh,
    out_dict=out_dict)
    #print("post_processed", post_processed)
    fm.save_output(post_processed[0], save_path_post)
    # return post_processed # debugging purposes

if __name__ == '__main__':
    from skimage import io

    fnames = ['../data/test_all/10HR/2400/predict/10hr2400_1.png',
              '../data/test_all/10HR/2400/predict/10hr2401_2.png',
              '../data/test_all/10HR/2400/predict/10hr2402_3.png']

    for ind, fname in enumerate(fnames):
        if ind == 0:
            predictions = io.imread(fname)
        else:
            img = io.imread(fname)
            predictions = np.dstack((predictions, img))

    args = {
            'compilation': 'min',
            'n_dilations': 3,
            'liberal_thresh': 200,
            'conservative_thresh': 160,
            'invert_double_thresh': True,
    }

    post_process(predictions, debug=True, **args)
