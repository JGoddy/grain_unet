'''
Inference Runner
Reads config for paths containing images and outputs segmentations without post-processing
'''
__author__ = "Matthew Patrick, Lauren Grae, Rosnel Leyva-Cortes" 

from utility.settings import *


import sys
import utility.user_interface as user_interface
from   unet_model.inference.post_processing.post_process import bulk_compile_and_pp, in_situ_post_process
from   unet_model.inference.infer  import single_inference, multi_folder_inference
from unet_model.inference.infer import in_situ_inference
from utility.settings import MODEL_NAME, TARGET_RESOLUTION, INFERENCE, PP_ACTIVATE, TEST_DATA_DIR, PREDICT_DATA_DIR, MODEL_PARAMS
import argparse
import utility.user_interface as user_interface





# if __name__ == "__main__":
#     import argparse
#     import utility.user_interface as user_interface
    #pattern = "fov*/raw/*.tif" #Original for new "test data", do not lose
def run_inference(
        model_path = None, #TODO: put a better default here
        prefix = None, #TODO: put a better default here
        mode=None, inference = True, post_process = True,
        image_folder_path = None, inference_pattern = None, 
        folders_pattern = None, folders_exclude = [], folders_include = [],
        images_pattern = None, images_exclude = ['trace'], images_include = [],
        image_path=None, output_path=None, compile=False, 
        integration=3, target_resolution = 256, 
        invert_double_thresh=True,out_dict=False, device = None):
    # user_interface.logoPrint()
    # print("sys.argv", sys.argv)
    # if len(sys.argv) == 2:
    #     parser = argparse.ArgumentParser(description="U-Net Inference Runner")
    #     parser.add_argument("mode", type=str, help="mode")
    #     args = parser.parse_args()

    #     print("args", args)
    #     print("args.mode", args.mode)

        if mode == "in_situ":
        
            if inference:
                # TODO: SHOULD THIS BE IN_SITU INFERENCE?
                multi_folder_inference(
                    model_path = model_path,
                    folder = image_folder_path, pattern = inference_pattern, exclude = images_exclude, include = images_include,
                    target_resolution = target_resolution, prefix = prefix, device = device) 
            if post_process: 
                in_situ_post_process(in_folder = image_folder_path,
                    folders_pattern = folders_pattern, folders_exclude = folders_exclude, 
                    folders_include = folders_include,
                    images_pattern = images_pattern, images_exclude = images_exclude, 
                    images_include = images_include, 
                    out_folder = f"{image_folder_path}/post_process", 
                    integration = integration, invert_double_thresh=invert_double_thresh,
                    compile = compile, out_dict= out_dict)
            #return post_processed # debugging purposes
           
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