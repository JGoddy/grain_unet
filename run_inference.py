'''
Inference Runner
Reads config for paths containing images and outputs segmentations without post-processing
'''
__author__ = "Matthew Patrick, Lauren Grae, Rosnel Leyva-Cortes, Julian Goddy" 

from utility.settings import *


import sys
import utility.user_interface as user_interface
from   unet_model.inference.post_processing.post_process import bulk_compile_and_pp, in_situ_post_process
from   unet_model.inference.infer  import single_inference, multi_folder_inference, in_situ_inference
#from unet_model.inference.infer import in_situ_inference
#from utility.settings import MODEL_NAME, TARGET_RESOLUTION, INFERENCE, PP_ACTIVATE, TEST_DATA_DIR, PREDICT_DATA_DIR, MODEL_PARAMS
#import argparse
#import utility.user_interface as user_interface
import utility.transformation as t




# if __name__ == "__main__":
#     import argparse
#     import utility.user_interface as user_interface
    #pattern = "fov*/raw/*.tif" #Original for new "test data", do not lose

# TODO: FIX THE PATHS SO THAT THE SUMMARY CSV IS SAVED IN THE TRAINING_SUMMARIES FOLDER
# TODO: NOT THE LABELS FOLDER 
    
def run_inference(
        model_path = None, #TODO: put a better default here
        prefix = None, #TODO: put a better default here
        mode=None, 
        inference = True, 
        post_process = True,
        image_folder_path = None,  
        folders_pattern = None, folders_exclude = [], folders_include = [],
        inference_pattern = None, postprocess_pattern = None, 
        inference_exclude = ['trace'], inference_include = [],
        postprocess_exclude = ['trace'], postprocess_include = [],
        image_path=None, output_path=None, compile=False, 
        integration=3, target_resolution = 256, 
        image_transform = t.inference_transforms(256),
        invert_double_thresh=True,
        conservative_thresh=160, liberal_thresh=200,
        n_dilations=3, min_grain_area=0, prune_size=5,
        out_dict=False, device = None):
    # user_interface.logoPrint()
    # print("sys.argv", sys.argv)
    # if len(sys.argv) == 2:
    #     parser = argparse.ArgumentParser(description="U-Net Inference Runner")
    #     parser.add_argument("mode", type=str, help="mode")
    #     args = parser.parse_args()

    #     print("args", args)
    #     print("args.mode", args.mode)

        if mode == "multi_folder_inference":
            
            if inference:
                # TODO: SHOULD THIS BE IN_SITU INFERENCE?
                # TODO: FIX THE PATHS HERE
                multi_folder_inference(
                    model_path = model_path,
                    folder = image_folder_path, pattern = inference_pattern, 
                    exclude = inference_exclude, include = inference_include,
                    target_resolution = target_resolution, image_transform = image_transform,
                    prefix = prefix, device = device) 
            if post_process: 
                in_situ_post_process(in_folder = image_folder_path,
                    folders_pattern = folders_pattern, folders_exclude = folders_exclude, 
                    folders_include = folders_include,
                    postprocess_pattern = postprocess_pattern, postprocess_exclude = postprocess_exclude, 
                    postprocess_include=postprocess_include, 
                    out_folder = f"{image_folder_path}/post_process", 
                    integration = integration, invert_double_thresh=invert_double_thresh,
                    conservative_thresh=conservative_thresh, liberal_thresh=liberal_thresh,
                    compile = compile, out_dict= out_dict,
                    target_resolution = target_resolution,
                    n_dilations=n_dilations, min_grain_area=min_grain_area, prune_size=prune_size,
                    image_transform = image_transform)

        elif mode == 'in situ':
            if inference:
                in_situ_inference(
                    model_path=model_path, 
                    folder=image_folder_path, pattern=inference_pattern, 
                    exclude=inference_exclude, include=inference_include, 
                    target_resolution=target_resolution,
                    image_transform=image_transform,
                    prefix=prefix,device=device)
            if post_process:
               in_situ_post_process(
                    #in_folder = image_folder_path,
                    in_folder=f"{image_folder_path}/inferenced/{prefix}",
                    folders_pattern = folders_pattern, folders_exclude = folders_exclude, 
                    folders_include = folders_include,
                    postprocess_pattern = postprocess_pattern, postprocess_exclude = postprocess_exclude, 
                    postprocess_include = postprocess_include, 
                    out_folder = f"{image_folder_path}/post_process/{prefix}", 
                    integration = integration, invert_double_thresh=invert_double_thresh,
                    conservative_thresh=conservative_thresh, liberal_thresh=liberal_thresh,
                    compile = compile, out_dict= out_dict,
                    target_resolution = target_resolution,
                    n_dilations=n_dilations, min_grain_area=min_grain_area, prune_size=prune_size,
                    image_transform = image_transform)
            #return post_processed # debugging purposes
           
            # the input Nouveaux images are at
            # training_data/nouveaux_and_nuevo_images/01.png
            #the inferenced Nouveaux images are at
            #training_data/nouveaux_and_nuevo_images/predict_predict_unet_nouveaux_dangling_enpoints_penalty_96_256/predict_01.png
            #should_be
            #training_data/nouveaux_and_nuevo_images/compiled/unet_nouveaux_dangling_endpoints_penalty_2026-07-08_11:02_42_epoch_96/compiled_01.png
            
            #image_folder_path = training_data/nouveaux_and_nuevo_images
            #prefix SHOULD BE unet_nouveaux_dangling_endpoints_penalty
            #f"{image_folder_path}/compiled/{prefix}
            #postprocess_include: compiled 

            #path.parent, f"compiled/{model_path.split("/")[-2]}_{model_path.split("/")[-1].split(".")[0]}",f'compiled_{path.name}')




            #the postprocessed images are at
            #training_data/nouveaux_and_nuevo_images/postprocess/postprocess_01_95_512.png
            #should be 96
            # the weights file is 
            #model_weights/nouveaux_dangling_endpoints_penalty/2026-07-08_11:02:42/epoch_96.pth
             

        elif mode == "post_process":
            #print(f"fov*/predict_{MODEL_NAME}_{TARGET_RESOLUTION}/")

            bulk_compile_and_pp(folder=PREDICT_DATA_DIR, 
                pattern = N_TEST_PATTERN, 
                post_process_option=True)
       
        elif mode == "single_inference":

       
            # parser = argparse.ArgumentParser(description="U-Net Inference Runner")
            # parser.add_argument("image_path", type=str, help="Path to the input image")
            # parser.add_argument("output_path", type=str, help="Path to save the output image")
            # args = parser.parse_args()
            if image_path is None or output_path is None:
                print("Image path and output path are required for single inference")
                sys.exit(1)
            single_inference(image_path, output_path, model = MODEL_PARAMS)
        
        elif mode == None: #no args are passed, assuming n_test functionality use your own pattern
            
            if (inference):
                print("Running inference")
                multi_folder_inference(folder = TEST_DATA_DIR, pattern = N_TEST_PATTERN) 

            if (post_process):
                print("Running post-processing")

                bulk_compile_and_pp(folder=PREDICT_DATA_DIR, pattern = '*.png', post_process_option=True)
                print("Post-processing completed")
            else:
                pass
        else:
            print("Invalid mode. Use 'in situ' or 'post process'.")
            sys.exit(1)