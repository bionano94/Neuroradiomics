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

    
