# from Akinlar and Chrome: PEL: A Predictive Edge Linking Algorithm (2016)
# Cuneyt Akinlar, Edward Chome,
# PEL: A Predictive Edge Linking algorithm,
# Journal of Visual Communication and Image Representation,
# Volume 36,
# 2016,
# Pages 159-171,
# ISSN 1047-3203,
# https://doi.org/10.1016/j.jvcir.2016.01.017.



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
        if val == 0 and row > 0 and col > 0 and row < len(image) -2 and col < len(image) -2

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
                

                #(x,y-1) -> (row-1,col)
                #(x-1,y) -> (row-1,col)

                # || (row-1,col-1) | (row-1,col) | (row-1,col+1) ||
                # || (row,col-1)   | (row, col)  | (row,col+1)   ||
                # || (row+1,col-1) | (row+1,col) | (row+1,col+1) ||

        

                # configuration 1: neighboring edge pixel is up-left: (row-1,col-1)
                

                # configuration 2: neighboring edge pixel is up: (row-1,col) 

                if image[row+2,col]==0:
                    image[row+1,col] = 0
                elif image[row+1,col+2]==0 or image[row+2,col+1]==0 or image[row+2,col+2]==0:
                    image[row+1,col+1]=0
                elif image[row+1,col-2]==0 or image[row+2,col-2]==0 or image[row+2,col-1]==0:
                    image[row+1,col-1]=0
                


                # 
                # configuration 3: neighboring edge pixel is up-right: (row-1,col+1)
                # 
                # configuration 4: neighboring edge pixel is left: (row,col-1) 
                
                #the edge is going right, so first check if there is an edgel to the right of the gap
                if image[row,col+2] == 0:
                    #fill in the gap 
                    image[row,col+1] = 0
                #otherwise, first check the upper pixels
                elif image[row-2,col+1] == 0 or image[row-2,col+2] or image[row-1,col+2] ==0:
                    image[row-1,col+1] = 0
                #otherwise, check the lower pixels
                elif image[row+1,col+2] ==0 or image[row+2,col+1]==0 or image[row+2,col+2]==0:
                    image[row+1,col+1]= 0 


                

                # configuration 5: neighboring edge pixel is right: (row,col+1)
                #
                # configuration 6: neighboring edge pixel is down-left: (row+1,col-1)
                #
                # configuration 7: neighboring edge pixel is down (row+1,col)

                # configuration 8: neighboring edge pixel is down-right: (row+1,col+1)




        