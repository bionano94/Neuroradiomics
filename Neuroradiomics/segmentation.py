import itk
import numpy as np


from Neuroradiomics.normalization import *
from Neuroradiomics.resampler import *
from Neuroradiomics.skull_stripping import *



######################
# Indexing Functions #
######################


def indexing (image, mask):
    '''
    This function takes an image and a mask and creates a 1D array with only the Grey Levels of the pixels masked.
    
    Parameters
    ----------
        image: itk image object
            The image you want to transform.
            
        mask: itk image object
            Binary mask of the part of the image you want to transform.
            
    
    Return
    ------
        image_array: 1D list of int
            The array of the image.
        
        index_array: 1D list of itk.Index
            The array with the ITK indexes of the pixel masked. This is useful to rebuild the image.
    '''
    
    
    Dimension = 3
    
    index = itk.Index[Dimension]()
    
    
    image_array = [] #The array with the grey values of the masked pixels
    index_array = [] #The array with the itk indexes of the pixels
    
    
    for index[0] in range( image.GetLargestPossibleRegion().GetSize()[0] ):
    
        for index[1] in range( image.GetLargestPossibleRegion().GetSize()[1] ):
        
            for index[2] in range( image.GetLargestPossibleRegion().GetSize()[2] ):
            
            #Only if the pixel is under the mask then the function will take that pixel
                if mask.GetPixel(index) != 0:
                    image_array.append( image.GetPixel(index) )
                    index_array.append( [ index[0], index[1], index[2] ] )
                
    return image_array, index_array



def de_indexing (image_array, index_array, reference_image, first_label_value = None ):
    '''
    This function takes a 1D array with only the Grey Levels and builds an image with . 
    Useful to build an ITK labels image from a 1D vector.
    
    Parameters
    ----------
        image_array: 1D list of int
            The array of the image.
        
        index_array: 1D list of itk.Index
            The array with the ITK indexes of the pixel in the array.
            
        reference_image: itk Image
            The image you want to use as referement to build the image. Must be of the same Size of the original indexed image.
            
        first_label_value: int number
            The first value from which you assign the gray level values. A sort of "zero value".
            It is useful when you build a labels image to set the first label value.
            Default is None.
            Useful to put =1 when de_indexing a label so the background (the part of the image not in the index_array) will be 0.
            
    Return
    ------
        image: itk Image object
            The image obtained from the array. Pixels will be casted to int type.
        
    '''
    
    Dimension = 3
    ImageType = itk.template(reference_image)[1]
    
    #Creation of the new itk image
    image = itk.Image[ImageType].New()
    
    #Creation of the itk Index object
    index = itk.Index[Dimension]()
    
    #Setting the new image space as the one of the original (reference) image.
    image.SetRegions( reference_image.GetLargestPossibleRegion() )
    image.SetSpacing( reference_image.GetSpacing() )
    image.SetOrigin( reference_image.GetOrigin() )
    image.SetDirection( reference_image.GetDirection() )
    image.Allocate()
    
    if first_label_value != None:
        for i in range(len(index_array)):
            #Set the itk index as the i_th index of the index_array
                index[0] = int( index_array[i][0] )
                index[1] = int( index_array[i][1] )
                index[2] = int( index_array[i][2] )
            
            #Set the Pixel value of the image as the one in the array
                image.SetPixel( index, int(image_array[i]) + first_label_value )
    else:
        for i in range(len(index_array)):
            #Set the itk index as the i_th index of the index_array
                index[0] = int( index_array[i][0] )
                index[1] = int( index_array[i][1] )
                index[2] = int( index_array[i][2] )
            
            #Set the Pixel value of the image as the one in the array
                image.SetPixel( index, int(image_array[i]) )
           
    return image

    

#####################
# Weights Functions #
#####################
    
    
#Three Classes Weights
    
def find_prob_weights (wm_mask, gm_mask, csf_mask):
    '''
    This function finds the proportions of the sizes of the white matter, grey matter mask, and csf mask of a brain.
    
    Parameters
    ----------
        wm_mask: itk image
            The wm mask.
        
        gm_mask: itk image
            The gm mask.
            
        csf_mask: itk image
            The csf mask.
    
    Return
    ------
        weigths: 1D list of floats.
            A list with the weights of the wm [0], gm [1], csf[2]. The sum is normalized to 1.
    
    '''
    
    #Getting arrays from masks
    wm_array = itk.GetArrayFromImage(wm_mask)
    gm_array = itk.GetArrayFromImage(gm_mask)
    csf_array = itk.GetArrayFromImage(csf_mask)
    
    #creating a sort of total brain mask summing the masks arrays 
    tot_array = wm_array + gm_array + csf_array
    
    #creating a 4dim array in order to use argmax
    four_dim_array = [wm_mask, gm_mask, csf_mask]
    
    #finding for every pixel which is its most probable type
    prob_array = np.argmax(four_dim_array, 0)
    
    #finding number of pixels for gm and csf
    gm_pixels = np.count_nonzero(prob_array == 1)
    csf_pixels = np.count_nonzero(prob_array == 2)
    
    #finding the total number of pixels
    tot_pixel = np.count_nonzero( tot_array ) 
    
    #because both background and wm will be labelled as 0, the wm number of pixels is find using subtraction.
    wm_pixels = tot_pixel - gm_pixels - csf_pixels
    
    #finding weights for the masks
    wm_weight = wm_pixels/tot_pixel
    gm_weight = gm_pixels/tot_pixel
    csf_weight = csf_pixels/tot_pixel
    
    #creating a list with all the weights.
    weights = [wm_weight, gm_weight, csf_weight]
    print ('The estimated weights are: wm = ', weights[0] ,'; gm = ', weights[1] ,'; csf = ', weights[2])
    
    return weights


#Four Classes weights

def find_prob_4_weights (wm_mask, gm_mask, csf_mask):
    '''
    This function finds the proportions of the sizes of the white matter, grey matter mask, and csf mask of a brain, including
    a fourth class that represents the indecision between white matter and grey matter.
    
    Parameters
    ----------
        wm_mask: itk image
            The wm mask.
        
        gm_mask: itk image
            The gm mask.
            
        csf_mask: itk image
            The csf mask.
    
    Return
    ------
        weigths: 1D list of floats.
            A list with the weights of the wm [0], gm [1], csf[2], indecision_class[3]. The sum is normalized to 1.
    
    '''
    
    #Getting arrays from masks
    wm_array = itk.GetArrayFromImage(wm_mask)
    gm_array = itk.GetArrayFromImage(gm_mask)
    csf_array = itk.GetArrayFromImage(csf_mask)
    idk_array = wm_array + gm_array
    
    #setting the rules to decide when a pixel must be classified as uncertain
    wm_gm_bool = np.logical_or((wm_array > 0.51) , (gm_array > 0.51) )
    csf_bool = np.logical_and((csf_array > wm_array), (csf_array > gm_array))
    idk_bool = np.logical_or( wm_gm_bool , csf_bool )
    
    #classifing the uncertain pixels.
    idk_array = np.where(idk_bool, 0, idk_array)
    
    
    #creating a sort of total brain mask summing the masks arrays 
    tot_array = wm_array + gm_array + csf_array
    
    
    #creating a 4dim array in order to use argmax
    four_dim_array = [wm_mask, gm_mask, csf_mask, idk_array]
    
    #creating a 4dim array in order to use argmax
    prob_array = np.argmax(four_dim_array, 0)
    
    #finding number of pixels for every class
    gm_pixels = np.count_nonzero(prob_array == 1) 
    csf_pixels = np.count_nonzero(prob_array == 2)
    idk_pixels = np.count_nonzero(prob_array == 3)
    
    #finding total pixels
    tot_pixel = np.count_nonzero( tot_array ) 
    
    #finding wm number of pixels.
    wm_pixels = tot_pixel - gm_pixels - csf_pixels - idk_pixels
    
    #finding weights.
    wm_weight = wm_pixels/tot_pixel
    gm_weight = gm_pixels/tot_pixel
    csf_weight = csf_pixels/tot_pixel
    idk_weight = idk_pixels/tot_pixel
    
    weights = [wm_weight, gm_weight, csf_weight, idk_weight]
    print ('The estimated weights are: wm = ', weights[0] ,'; gm = ', weights[1] ,'; csf = ', weights[2], '; undefined = ', weights[3])
    
    return weights



