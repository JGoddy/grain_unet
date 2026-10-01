# from Akinlar and Chrome: PEL: A Predictive Edge Linking Algorithm (2016)
# Cuneyt Akinlar, Edward Chome,
# PEL: A Predictive Edge Linking algorithm,
# Journal of Visual Communication and Image Representation,
# Volume 36,
# 2016,
# Pages 159-171,
# ISSN 1047-3203,
# https://doi.org/10.1016/j.jvcir.2016.01.017.

import numpy as np

## Step 1: Fill gaps 

def fillGaps(image):
    """
    Fills one pixel gaps in the binary grain boundary (edge) map
    NOTE: This function assumes that the grain boundary (edge) image is already binarized
    """
    #loop over the image for now
    # TODO: change to more effecient method, look at train.are_there_dangling_endpoints_test(image)
    for (row,col), val in np.ndenumerate(image):
       
        #first, only select the pixels constituting the grain boundary (black and not on the edge)  
        if val == 0 and row > 0 and col > 0 and row < len(image) -2 and col < len(image) -2:

            #select grain boundary pixels with only one neighbor 
            #these are the edgels
            #TODO: is there a better way to determine this?

            num_neighbors = sum((image[row+1,col]<=0.5, image[row,col+1]<=0.5,  
                    image[row-1,col]<=0.5, image[row,col-1]<=0.5, 
                    image[row+1,col+1]<=0.5, image[row-1,col-1]<=0.5, 
                    image[row+1,col-1]<=0.5, image[row-1,col+1]<=0.5))

            if num_neighbors == 1: 

                #there are eight configurations of where the neighbor could be. 
                #enumerate them for now and then combine later 
                #NOTE: (x,y) pixel coordinates starting from top-left
                # x is col, y is row
                

                # | (row-2,col-2) |(row-2,col-1) | (row-2,col) | (row-2,col+1) | (row-2,col+2)| 
                # | (row-1,col-2) |(row-1,col-1) | (row-1,col) | (row-1,col+1) | (row-1,col+2)| 
                # | (row,col-2)   |(row,col-1)   |  (row,col)  | (row,col+1)   | (row,col+2)  |
                # | (row+1,col-2) |(row+1,col-1) | (row+1,col) | (row+1, col+1)| (row+1,col+2)| 
                # | (row+2,col-2) |(row+2,col-1) | (row+2,col) | (row+2, col+1)| (row+2,col+2)| 

                # configuration 1: neighboring edge pixel is up-left: (row-1,col-1)
                # first identify this configuration
                if image[row-1,col-1]==0:
                    #then check opposite the neighboring pixel
                    if image[row+1,col+2]==0 or image[row+2,col+1] or image[row+2,col+2]==0:
                        image[row+1,col+1]=0
                    #then check if there is an edgel down (the gap is down)
                    elif image[row+2,col]==0:
                        image[row+1,col]=0
                    #finally, check if there is an edgel to the right (the gap is to the right)
                    elif image[row,col+2]==0:
                        image[row,col+1]=0

                # configuration 2: neighboring edge pixel is up: (row-1,col) 
                # first, identify this configuration
                elif image[row-1,col]==0:
                    #first check if the gap is down (opposite of the neighboring pixel)
                    if image[row+2,col]==0:
                        image[row+1,col] = 0
                    #then check if the gap is down right
                    elif image[row+1,col+2]==0 or image[row+2,col+1]==0 or image[row+2,col+2]==0:
                        image[row+1,col+1]=0
                    #finally, check if the gap is down left
                    elif image[row+1,col-2]==0 or image[row+2,col-2]==0 or image[row+2,col-1]==0:
                        image[row+1,col-1]=0
        
                # configuration 3: neighboring edge pixel is up right: (row-1,col+1)
                # first, identify this configuration
                elif image[row-1,col+1]==0:
                    #first check if the gap is down left (opposite neighboring pixel)
                    if image[row+1,col-2] or image[row+2,col-2]==0 or image[row+2,col-1]==0:
                        image[row+1,col-1]=0
                    #then, check if the gap is down
                    elif image[row+2,col]==0:
                        image[row+1,col]=0
                    #finally check if the gap is to the left
                    elif image[row,col-2]==0:
                        image[row,col-1]=0

                # configuration 4: neighboring edge pixel is left: (row,col-1) 
                #first, identify this configuration
                elif image[row,col-1]==0:
                    #the edge is going right, gap is to the right (opposite of the neighboring pixel)
                    if image[row,col+2] == 0:
                        #fill in the gap 
                        image[row,col+1] = 0
                    #otherwise, first check if the gap is up right
                    elif image[row-2,col+1] == 0 or image[row-2,col+2] or image[row-1,col+2] ==0:
                        image[row-1,col+1] = 0
                    #otherwise, check the if the gap is down right
                    elif image[row+1,col+2] ==0 or image[row+2,col+1]==0 or image[row+2,col+2]==0:
                        image[row+1,col+1]= 0 

                # configuration 5: neighboring edge pixel is right: (row,col+1)
                #first, identify this configuration
                elif image[row,col+1]==0:
                    #first check if the gap is left (opposite the neighboring pixel)
                    if image[row,col-2]==0:
                        image[row,col-1]=0
                    #then check if the gqp is up left
                    elif image[row-2,col-2]==0 or image[row-2,col-1]==0 or image[row-1,col-2]==0:
                        image[row-1,col-1]=0
                    #finally check if the gap is down left
                    elif image[row+1,col-2]==0 or image[row+2,col-2]==0 or image[row+2,col-1]==0:
                        image[row+1,col-1]=0

                # configuration 6: neighboring edge pixel is down-left: (row+1,col-1)
                # first identify this configuration
                elif image[row+1,col-1]==0:
                    #first check if the gap is up right (opposite the neighboring pixel)
                    if image[row-2,col+1]==0 or image[row-2,col+2]==0 or image[row-1,col+2]==0:
                        image[row-1,col+1]=0
                    #then check if the gap is up
                    elif image[row-2,col]==0:
                        image[row-1,col]=0
                    #finally check if the gap is right
                    elif image[row,col+2]==0:
                        image[row,col+1]=0

                # configuration 7: neighboring edge pixel is down (row+1,col)
                #first identify this configuration
                elif image[row+1,col]==0:
                    #first check if the gap is up (opposite the neighboring pixel)
                    if image[row-2,col]==0:
                        image[row-1,col]=0
                    #then check if the gap is up left
                    elif image[row-2,col-2]==0 or image[row-2,col-1]==0 or image[row-1,col-2]==0:
                        image[row-1,col-1]=0
                    #finally, check if the neighboring pixel is up right
                    elif image[row-2,col+1]==0 or image[row-2,col+2]==0 or image[row-1,col+2]==0:
                        image[row-1,col+1]=0

                # configuration 8: neighboring edge pixel is down-right: (row+1,col+1)
                #first identify the configuration
                elif image[row+1,col+1]==0:
                    #first check if the gap is up-left (opposite the neighboring pixel)
                    if image[row-2,col-2]==0 or image[row-2,col-1]==0 or image[row-1,col-2]==0:
                        image[row-1,col-1]=0
                    #then check if the gap is up
                    elif image[row-2,col]==0:
                        image[row-1,col]=0
                    #finally check if the gap is left
                    elif image[row,col-2]==0:
                        image[row,col-1]=0


## Step 2: createSegments: linking continuous edgels to create pixel chains 
# NOTE: Skipping this step because I don't think it is necessary for our purpose.
# We want to identify the boundaries and triple junctions but can do that separately
# and can just output an image. Also, I don't like the result in Fig. 5. In particular,
# I think point (9,10) is important. This method does not seem very robust.        

## Step 3. joinSegments: Extending nearby edge segments 
# Skip this for the same reason as for step 3. 
#TODO: remove dangling endpoints at the end. 

## Step 4. thinSegments: Thinning down and cleaning up edge segments

# See Matthew's postprocessing code. Probably the skeletonizing etc. can be used instead. 

