import itk
import numpy as np

from Neuroradiomics.normalization import itk_gaussian_normalization
from Neuroradiomics.registration import elastix_multimap_registration
from Neuroradiomics.resampler import match_physical_spaces




#####################
# Masking Functions #
#####################

def negative_3d_masking (image, mask):
    '''
    This function apply the negative of the image loaded as "mask" to a 3D image.
    
    Parameters
    ----------
        image: 3D itk object.
                The image you want to apply the mask on.
        
        mask: 3D itk object.
                The binary image of the mask you want to apply.
    
    Returns
    -------
        masked_image: 3D itk object.
                        The masked image.
    '''
    
    Dimension = 3
    ImageType = itk.template(image)[1]
    masked_image = itk.Image[ImageType].New()
    
    masked_image.SetRegions( image.GetLargestPossibleRegion() )
    masked_image.SetSpacing( image.GetSpacing() )
    masked_image.SetOrigin( image.GetOrigin() )
    masked_image.SetDirection( image.GetDirection() )
    masked_image.Allocate()
    
    index = itk.Index[Dimension]()
    
    #This is necessary to prevent problems with ITK python wrapping issues. This way the function should work with every type of 3D images.
    for index[0] in range( mask.GetLargestPossibleRegion().GetSize()[0] ):
        for index[1] in range( mask.GetLargestPossibleRegion().GetSize()[1] ):
            for index[2] in range( mask.GetLargestPossibleRegion().GetSize()[2] ):
                if mask.GetPixel(index) < 1e-01:
                    masked_image.SetPixel(index, 0)
                else: 
                    masked_image.SetPixel(index, image.GetPixel(index))
                
    
    return masked_image



def masking (image, mask):
    '''
    This function apply the image loaded as "mask" to a 3D image.
    
    Parameters
    ----------
        image: 3D itk object.
                The image you want to apply the mask on.
        
        mask: 3D itk object.
                The binary image of the mask you want to apply.
    
    Returns
    -------
        masked_image: 3D itk object.
                        The masked image.
    '''
    
    Dimension = 3
    ImageType = itk.template(image)[1]
    masked_image = itk.Image[ImageType].New()
    
    masked_image.SetRegions( image.GetLargestPossibleRegion() )
    masked_image.SetSpacing( image.GetSpacing() )
    masked_image.SetOrigin( image.GetOrigin() )
    masked_image.SetDirection( image.GetDirection() )
    masked_image.Allocate()
    
    index = itk.Index[Dimension]()
    
    #This is necessary to prevent problems with ITK python wrapping issues. This way the function should work with every type of 3D images.
    for index[0] in range( mask.GetLargestPossibleRegion().GetSize()[0] ):
        for index[1] in range( mask.GetLargestPossibleRegion().GetSize()[1] ):
            for index[2] in range( mask.GetLargestPossibleRegion().GetSize()[2] ): 
                if mask.GetPixel(index) > 0.5:
                    masked_image.SetPixel(index, 0)
                else: 
                    masked_image.SetPixel(index, image.GetPixel(index))
                
    
    return masked_image



##########################
# Thresholding Functions #
##########################

def binarize ( image, low_value = 0.1, hi_value = None ):
    '''
    This function applies a threshold to an image to binarize it. It uses a low_value (default = 0.1) to exclude the background and eventually an hi_value.
    
    Parameters
    ----------
        image: itk Image object.
            The image you want to binarize
        
        low_value: float value
            The low value for the thresholding
            
        hi_value: float value. Default = None
            If given is the upper value for the thresholding.
            
    Return
    ------
        final_image: itk Image object
            The binarized thresholded image.
    '''
    
    #Check the image type. It must be itk.F for the thresholding filter.
    OutputType = itk.Image[itk.F, 3]
    
    if type(image) != OutputType:
        
        #cast to the pixel value supported by itk python wrapping
        cast_filter = itk.CastImageFilter[type(image), OutputType].New()
        cast_filter.SetInput(image)
        cast_filter.Update()
        c_image = cast_filter.GetOutput()
    
    else: c_image = image
    
    #Thresholding the image
    thresholdFilter = itk.BinaryThresholdImageFilter[OutputType, OutputType].New()
    thresholdFilter.SetInput(c_image)
    thresholdFilter.SetLowerThreshold(low_value)

    #If an hi_value is given then it is set to that value
    if hi_value != None : 
        thresholdFilter.SetUpperThreshold(hi_value)

    thresholdFilter.SetOutsideValue(0)
    thresholdFilter.SetInsideValue(1)
    thresholdFilter.Update()
    
    
    if type(image) != OutputType:
        
        #if the image was casted then it is casted again to return on the same type of the input image.
        cast_filter = itk.CastImageFilter[OutputType, type(image)].New()
        cast_filter.SetInput( thresholdFilter.GetOutput() )
        cast_filter.Update()
    
        final_image = cast_filter.GetOutput()
        
    else:
        final_image = thresholdFilter.GetOutput()
    
    return final_image 


#########################
# Specialized Functions #
#########################

def find_largest_connected_region (image):
    '''
    This function find the largest connected region in a binary image.
    
    Parameters
    ----------
        image: itk image object
            The binary image which you want to find the largest connected region
            
    Returns
    -------
        final_image: itk image object
            The same image with only the largest connected region as foreground(1) and everything else as background(0)
    '''
    
    #eseguo il cast per essere sicuro che la funzione venga applicata
    OutputType = itk.Image[itk.SS, 3]
    cast_filter = itk.CastImageFilter[type(image), OutputType].New()
    cast_filter.SetInput(image)
    cast_filter.Update()
    
    
    #cerco gli oggetti connessi dell'immagine
    connected_filter = itk.ConnectedComponentImageFilter[OutputType, OutputType].New()
    connected_filter.SetInput(cast_filter.GetOutput())
    connected_filter.Update()
    
    #relabling the images in increasing order for decreasing dimension
    label_filter = itk.RelabelComponentImageFilter[OutputType, OutputType].New()
    label_filter.SetInput(connected_filter.GetOutput())
    label_filter.Update()
    
    #tengo solo la regione connessa più grande
    thresholdFilter = itk.ThresholdImageFilter[OutputType].New()
    thresholdFilter.SetInput(label_filter.GetOutput())
    thresholdFilter.ThresholdOutside (1,1)
    thresholdFilter.SetOutsideValue(0)
    thresholdFilter.Update()
    
    #rifaccio il cast in modo da restituire l'immagine dello stesso tipo dell'input
    cast_filter = itk.CastImageFilter[OutputType, type(image)].New()
    cast_filter.SetInput(thresholdFilter.GetOutput())
    cast_filter.Update()
    
    return cast_filter.GetOutput()




#######################
# Filtering Functions #
#######################


def binary_eroding (image, radius = 1):
    '''
    This function applies an eroding filter with a ball structuring element to a binary image.
    
    Parameters
    ----------
        image: itk image object
            The binary image you want to erode
        
        radius: int number
            The radius of the structurin element. Default = 1.
    
    Returns
    -------
        eroded_image: itk image object
            The image eroded.
    '''
    
    #creo l'oggetto per l'erosione
    struct_element = itk.FlatStructuringElement[3].Ball(radius)
    
    erodingFilter = itk.BinaryErodeImageFilter[type(image), type(image), type(struct_element)].New()
    erodingFilter.SetInput(image)
    erodingFilter.SetKernel(struct_element)
    erodingFilter.SetBackgroundValue(0)
    erodingFilter.SetForegroundValue(1)
    erodingFilter.Update()
    
    return erodingFilter.GetOutput()


    
def binary_dilating (image, radius=1):
    '''
    This function applies a dilating filter with a ball structuring element to a binary image.
    
    Parameters
    ----------
        image: itk image object
            The binary image you want to dilate.
        
        radius: int number
            The radius of the structuring element. Default = 1.
    
    Returns
    -------
        dilated_image: itk image object
            The dilated image.
    '''
         
    
    #Eseguo una dilatazione di raggio 'radius'

    struct_element = itk.FlatStructuringElement[3].Ball(radius)


    dilatingFilter = itk.BinaryDilateImageFilter[type(image), type(image), type(struct_element)].New()
    dilatingFilter.SetInput(image)
    dilatingFilter.SetKernel(struct_element)
    dilatingFilter.SetDilateValue(1) 
    dilatingFilter.Update()
    
    return dilatingFilter.GetOutput()



def hole_filler(image):
    '''
    This function applies an ITK Binary Fillhole Filter to the input binary image
    
    Parameters
    ----------
        image: itk Image object
            The input binary image.
            
    Returns
    -------
        filled_image: itk Image object
            The image in output of the filter
    
    '''
    
    #eseguo il cast per essere sicuro che la funzione venga applicata
    OutputType = itk.Image[itk.SS, 3]
    cast_filter = itk.CastImageFilter[type(image), OutputType].New()
    cast_filter.SetInput(image)
    cast_filter.Update()
    
    hole_filler_filter = itk.BinaryFillholeImageFilter[OutputType].New()
    hole_filler_filter.SetInput(cast_filter.GetOutput())
    hole_filler_filter.SetForegroundValue(1)
    hole_filler_filter.Update()
    
    #rifaccio il cast in modo da restituire l'immagine dello stesso tipo dell'input
    cast_filter = itk.CastImageFilter[OutputType, type(image)].New()
    cast_filter.SetInput(hole_filler_filter.GetOutput())
    cast_filter.Update()
    
    return cast_filter.GetOutput()




#######################
# Stripping Functions #
#######################

def skull_stripping_mask (image, atlas, mask, transformation_return = False):
    '''
    This function creates a mask to extract the brain from an head image.
    
    Parameters
    ----------
        image: itk image object
            The head mri image.
            
        atlas: itk image object
            The atlas for the head mri.
            
        mask: itk image object
            Binary mask of the brain for the atlas.
            
        transformation_return: boolean parameter. Default is False.
            If True the function returns also the transformation parameters of the registration.
            
    Returns
    -------
        brain_mask: itk image object
            A binary image with the brain mask for the image in input.
            
        Transformation_parameters: elastix transformation object.
            The transformation parameters of the registration. Returned only if transformation_return is set to True.
    '''
    
    reg_atlas_obj = elastix_multimap_registration( image, atlas )
    
    #apply the transformation to the mask and then make it again a binary image
    reg_mask = itk.transformix_filter( mask, reg_atlas_obj.GetTransformParameterObject() )
    bin_reg_mask = binarize( reg_mask )
    
    
    #dilating the mask to be more conservative
    bin_reg_mask = binary_dilating(bin_reg_mask)
    
    
    #do a first skull stripping
    first_brain = negative_3d_masking( image, bin_reg_mask )
    
    #binarize the first_brain to obtain a mask useful for the normalization
    first_brain_mask = binarize( first_brain )
    
    
    #Casting the first_brain_mask to be used by the normalization filter
    OutputType = itk.Image[itk.UC, 3]
    cast_filter = itk.CastImageFilter[type(first_brain_mask), OutputType].New()
    cast_filter.SetInput(first_brain_mask)
    cast_filter.Update()
    first_brain_mask = cast_filter.GetOutput()
    
    #normalize the first obtained brain
    normalized_first_brain = itk_gaussian_normalization( first_brain, first_brain_mask )
    normalized_first_brain.Update()
    
    #Normalized image must be casted
    OutputType = itk.Image[itk.D, 3]
    cast_filter = itk.CastImageFilter[type(normalized_first_brain.GetOutput() ), OutputType].New()
    cast_filter.SetInput(normalized_first_brain)
    normalized_first_brain = cast_filter.GetOutput()
    
    #thresholding the normalized_brain
    InputType = type(normalized_first_brain)
    motsu = itk.OtsuMultipleThresholdsImageFilter[InputType, InputType].New()
    motsu.SetInput(normalized_first_brain)
    motsu.SetNumberOfThresholds(3)
    motsu.Update()
    
    thresholded_first_brain = binarize(motsu.GetOutput(), hi_value = 3 )
    
    second_brain = negative_3d_masking( image, thresholded_first_brain )
    
    #Casting the thresholded_first_brain to be used by the normalization filter
    OutputType = itk.Image[itk.UC, 3]
    cast_filter = itk.CastImageFilter[type(thresholded_first_brain), OutputType].New()
    cast_filter.SetInput(thresholded_first_brain)
    cast_filter.Update()
    
    #normalize the second obtained brain
    normalized_second_brain = itk_gaussian_normalization( second_brain, cast_filter.GetOutput() )
    normalized_second_brain.Update()
  
    
    thresholded_second_brain = binarize( normalized_second_brain.GetOutput(), -3, 1.5 )  
    
    #eroding the mask to better find the largest connected region
    eroded_mask = binary_eroding( thresholded_second_brain )
    
    #find the largest connected region
    first_mask = find_largest_connected_region( eroded_mask )
    
    #apply a dilation to the mask and a hole filler
    final_mask = hole_filler( binary_dilating(first_mask, 2) )
    
    
    if transformation_return == False:
        return final_mask
    else:
        return final_mask, reg_atlas_obj.GetTransformParameterObject()

    
    
def skull_stripper (image, atlas, mask):
    '''
    This function extract the brain from an head image.
    
    Parameters
    ----------
        image: itk image object
            The head mri image.
            
        atlas: itk image object
            The atlas for the head mri.
            
        mask: itk image object
            Binary mask of the brain for the atlas.
            
    Returns
    -------
        brain: itk image object
           The brain extracted from the head image in input.
    '''
    
    brain = negative_3d_masking( image, skull_stripping_mask( image, atlas, mask ) )
    
    return brain
    
    
