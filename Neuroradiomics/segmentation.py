import itk
import numpy as np
import sys
from sklearn.mixture import GaussianMixture

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
        image_list: 1D list of int
            The array of the image.
        
        index_list: 1D list of itk.Index
            The array with the ITK indexes of the pixel masked. This is useful to rebuild the image.
    '''
    
    
    Dimension = 3
    
    index = itk.Index[Dimension]()
    
    
    image_list = [] #The array with the grey values of the masked pixels
    index_list = [] #The array with the itk indexes of the pixels
    
    
    #This part permits to store in a list only the pixels covered by a defined mask
    for index[0] in range( image.GetLargestPossibleRegion().GetSize()[0] ):
    
        for index[1] in range( image.GetLargestPossibleRegion().GetSize()[1] ):
        
            for index[2] in range( image.GetLargestPossibleRegion().GetSize()[2] ):
            
            #Only if the pixel is under the mask then the function will take that pixel
                if mask.GetPixel(index) != 0:
                    image_list.append( image.GetPixel(index) )
                    index_list.append( [ index[0], index[1], index[2] ] )
                
    return image_list, index_list



def label_de_indexing (image_array, index_array, reference_image, first_label_value = 0 ):
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
            Default is 0.
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
    
    
    #create a black image. This is necessary to get around issues with types for itk python wrapping
    for index[0] in range( image.GetLargestPossibleRegion().GetSize()[0] ):
    
        for index[1] in range( image.GetLargestPossibleRegion().GetSize()[1] ):
        
            for index[2] in range( image.GetLargestPossibleRegion().GetSize()[2] ):
                
                image.SetPixel(index, 0)

    for i in range(len(index_array)):
        #Set the itk index as the i_th index of the index_array
            index[0] = int( index_array[i][0] )
            index[1] = int( index_array[i][1] )
            index[2] = int( index_array[i][2] )

        #Set the Pixel value of the image as the one in the array
            image.SetPixel( index, image_array[i] + first_label_value ) 
           
    return image

    

#####################
# Weights Functions #
#####################
    
    
#Three Classes Weights
    
def find_prob_weights (csf_mask, gm_mask, wm_mask):
    '''
    This function finds the proportions of the sizes of the white matter, grey matter mask, and csf mask of a brain.
    
    Parameters
    ----------
        csf_mask: itk image object. Must be binary.
            The probability mask for the csf.
            
        gm_mask: itk image object. Must be binary.
            The probability mask for the gm.
            
        wm_mask: itk image object. Must be binary.
            The probability mask for the wm.
    
    Return
    ------
        weigths: 1D list of floats.
            A list with the weights of the wm [2], gm [1], csf[0]. The sum is normalized to 1.
            (If the given masks are empty an error will be print and None will be returned instead.)
    
    '''
    
    #Getting arrays from masks
    wm_array = itk.GetArrayFromImage(wm_mask)
    gm_array = itk.GetArrayFromImage(gm_mask)
    csf_array = itk.GetArrayFromImage(csf_mask)
    
    #creating a sort of total brain mask summing the masks arrays 
    tot_array = wm_array + gm_array + csf_array
    
    #creating a 4dim array in order to use argmax
    four_dim_array = [wm_array, gm_array, csf_array]
    
    #finding for every pixel which is its most probable type
    prob_array = np.argmax(four_dim_array, 0)
    
    #finding number of pixels for gm and csf
    gm_pixels = np.count_nonzero(prob_array == 1)
    csf_pixels = np.count_nonzero(prob_array == 2)
    
    #finding the total number of pixels
    tot_pixel = np.count_nonzero( tot_array )
    
    if tot_pixel == 0 :
        print( 'Error in function "find_prob_weights": all the provided masks are empty! None will be returned instead.', file = sys.stderr)
        return None #None allows for the automatic initialization fo the parameters in the segmentation function
    
    #because both background and wm will be labelled as 0, the wm number of pixels is find using subtraction.
    wm_pixels = tot_pixel - gm_pixels - csf_pixels
    
    #finding weights for the masks
    wm_weight = wm_pixels/tot_pixel
    gm_weight = gm_pixels/tot_pixel
    csf_weight = csf_pixels/tot_pixel
    
    #creating a list with all the weights.
    weights = [csf_weight, gm_weight, wm_weight]
    
    return weights


#Four Classes weights

def find_prob_4_weights (csf_mask, gm_mask, wm_mask):
    '''
    This function finds the proportions of the sizes of the white matter, grey matter mask, and csf mask of a brain, including
    a fourth class that represents the indecision between white matter and grey matter.
    
    Parameters
    ----------
        csf_mask: itk image object. Must be binary.
            The probability mask for the csf.
            
        gm_mask: itk image object. Must be binary.
            The probability mask for the gm.
            
        wm_mask: itk image object. Must be binary.
            The probability mask for the wm.
    
    Return
    ------
        weigths: 1D list of floats.
            A list with the weights of the wm [2], gm [1], csf[0], indecision_class[3]. The sum is normalized to 1.
    
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
    four_dim_array = [wm_array, gm_array, csf_array, idk_array]
    
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
    
    weights = [csf_weight, gm_weight, wm_weight, idk_weight]
    
    return weights




###################
# Means Functions #
###################

def gaussian_pixel_distribution_params_evaluation (image, label):
    '''
    This function finds mean and standard deviation for the pixels of an image under a mask.
    
    Parameters
    ----------
        image: itk image object.
            The image you want to evaluate the parameters of.
            
        label: itk image object. Must be binary.
            The mask you want to use. Only pixels under it will be evaluated.
            
    Returns
    -------
        results: list of float.
            results[0] = mean value. results[1] = standard eviation
    '''
   
    OutputType = itk.Image[itk.SS, 3]

    cast_filter = itk.CastImageFilter[type(label), OutputType].New()
    cast_filter.SetInput(label)
    cast_filter.Update()
    cast_label = cast_filter.GetOutput()
    
    label_filter = itk_label_shape_statistics(image, cast_label)
    label_filter.Update()
    
    results = [label_filter.GetMean(1), label_filter.GetSigma(1)]
    
    return results


def find_means (brain, brain_mask, csf_mask, gm_mask, wm_mask):
    '''
    This function finds a mean for the masked pixels.
    
    Parameters
    ----------
        brain: itk image object.
            The brain image of which you want to evaluate the means values.
            
        brain_mask: itk image object
            Binary mask used for the skull stripping. A binary image of the brain.
            
        csf_mask: itk image object. Must be binary.
            The probability mask for the csf.
            
        gm_mask: itk image object. Must be binary.
            The probability mask for the gm.
            
        wm_mask: itk image object. Must be binary.
            The probability mask for the wm.
            
    
    Returns
    -------
        means: list of float.
            The evaluated means.
            means[0] = csf_mean
            means[1] = gm_mean
            means[2] = wm_mean
    '''
    
    #converting everything in numpy array. This is done because ITK functions in python are not always wrapped for every type. 
    brain_array = itk.GetArrayFromImage(brain) 
    wm_array = itk.GetArrayFromImage(wm_mask)
    gm_array = itk.GetArrayFromImage(gm_mask)
    csf_array = itk.GetArrayFromImage(csf_mask)
    brain_mask_array = itk.GetArrayFromImage(brain_mask)

    #setting a greater value for bg in order to have it correctly selected in prob array
    background = np.where(brain_mask_array == 0, 2, 0)
    
    #finding for each pixel what is its more probable class (bg = 0, wm = 3, gm = 2, csf = 1) 
    four_dim_array = [background, csf_array, gm_array, wm_array]
    prob_array = np.argmax(four_dim_array, 0)
    
    #recreating the pixels array
    wm_array = np.where(prob_array == 3, 1, 0)
    gm_array = np.where(prob_array == 2, 1, 0)
    csf_array = np.where(prob_array == 1, 1, 0)

    #creating arrays in which there are only pixels of the brain where those pixels are of that class
    wm_brain = np.where (wm_array == 1, brain_array, 0)
    gm_brain = np.where (gm_array == 1, brain_array, 0)
    csf_brain = np.where (csf_array == 1, brain_array, 0)
    
    
    #find means
    wm_mean = np.sum(wm_brain)/np.count_nonzero(wm_array)
    gm_mean = np.sum(gm_brain)/np.count_nonzero(gm_array)
    csf_mean = np.sum(csf_brain)/np.count_nonzero(csf_array)
    
    means = [csf_mean, gm_mean, wm_mean]
        
    return means


def find_4_means (brain, csf_mask, gm_mask, wm_mask, prob_threshold = 0.7):
    '''
    This function finds a rough mean for the masked pixels. This is useful for the 4 class segmentation, with the uncerain pixels. (not sure if wm or gm)
    
    Parameters
    ----------
        brain: itk image object.
            The brain image of which you want to evaluate the means values.
            
        csf_mask: itk image object. Must be binary.
            The probability mask for the csf.
            
        gm_mask: itk image object. Must be binary.
            The probability mask for the gm.
            
        wm_mask: itk image object. Must be binary.
            The probability mask for the wm.
            
        prob_threshold: float value.
            The min value of probability you want to consider as valid. Default is 0.51
            
    
    Returns
    -------
        means: list of float.
            The evaluated means.
            means[0] = csf_mean
            means[1] = gm_mean
            means[2] = wm_mean
            means[3] = uncertain_mean
    '''
    
    #binarizing for the probability higher than the fixed threshold
    min_wm = binarize(wm_mask, prob_threshold)
    min_gm = binarize(gm_mask, prob_threshold)
    min_csf = binarize(csf_mask, prob_threshold)
    
    #probability between 0.1 and the fixed threshold
    idk = binarize(wm_mask, 0.1, prob_threshold)
    
    wm_mean = gaussian_pixel_distribution_params_evaluation(brain, min_wm)[0]
    gm_mean = gaussian_pixel_distribution_params_evaluation(brain, min_gm)[0]
    csf_mean = gaussian_pixel_distribution_params_evaluation(brain, min_csf)[0]
    idk_mean = gaussian_pixel_distribution_params_evaluation(brain, idk)[0]
    
    means = [csf_mean, gm_mean, wm_mean, idk_mean]
        
    return means




#########################
# Segmentation Function #
#########################


def brain_segmentation ( brain, brain_mask, wm_mask, gm_mask, csf_mask, auto_mean = False, undefined = False, proba = False ):
    '''
    This function segment a brain image.
    
    Parameters
    ----------
        brain: itk image object
            The brain image. The brain must be already extracted.
            
        brain_mask: itk image object
            Binary mask used for the skull stripping. A binary image of the brain.
            
        wm_mask: itk image object.
            The wm probability mask. It must be already in the brain space and it must be masked with the same brain
            mask of the brain.
            
        gm_mask: itk image object.
            The gm probability mask. It must be already in the brain space and it must be masked with the same brain
            mask of the brain.
            
        csf_mask: itk image object.
            The csf probability mask. It must be already in the brain space and it must be masked with the same brain
            mask of the brain.
        
        auto_mean: boolean. Default = False.
            If True the segmentation will try to find the mean values for each class. If false are used default ones.
            
            
        undefined: boolean. Default = False.
            If True the segmentation will find also a fourth classe with the not certain pixels.
            
        proba: boolean. Default = False
            If True the algorithm will return 3 probability maps, one for each tissue, as float images with values between 0 and 1.
            
    Returns
    -------
        label_image: itk image object.
            ONLY IF proba = False.
            The label image.
            0 is background
            1 is csf
            2 is gm
            3 is wm
            Is uncertain is setted to True 4 are the uncertain pixels. 
        
        wm: itk Image object.
            ONLY IF proba = True.
            The probability map for the white matter. Pixel values are float between 0 and 1.
            
        gm: itk Image object.
            ONLY IF proba = True.
            The probability map for the grey matter. Pixel values are float between 0 and 1.
            
        csf: itk Image object.
            ONLY IF proba = True.
            The probability map for the cerebrospinal fluid. Pixel values are float between 0 and 1.
    
    '''
    
    #Brain normalization
    
    #casting of the mask for the normalization.
    OutputType = itk.Image[itk.UC, 3]
    cast_filter = itk.CastImageFilter[type(brain_mask), OutputType].New()
    cast_filter.SetInput(brain_mask)
    cast_filter.Update()
    brain_mask = cast_filter.GetOutput()
    
    norm_brain_filter = itk_gaussian_normalization (brain, brain_mask)
    
    #I change the physical space because the normalization have changed it.
    matching_filter = match_physical_spaces(norm_brain_filter.GetOutput(), brain)
    matching_filter.Update()
    brain = matching_filter.GetOutput()
    
    #mask back to float
    cast_filter = itk.CastImageFilter[type(brain_mask), OutputType].New()
    cast_filter.SetInput(brain_mask)
    cast_filter.Update()
    brain_mask = cast_filter.GetOutput()
    
    
    #linearize and indexing the brain obtaining the image array and the index array
    #(the index array is useful to build the itk label image)
    brain_array, index_array = indexing (brain, brain_mask)
    
    
    #Masking also the Probability maps
    wm_mask = negative_3d_masking(wm_mask, brain_mask)
    gm_mask = negative_3d_masking(gm_mask, brain_mask)
    csf_mask = negative_3d_masking(csf_mask, brain_mask)

    
    #INITIALIZING THE MODELS PARAMETERS    
    if  auto_mean : #if automean the proposed function will be used to find the means values
    
        if undefined :
              means = np.reshape(find_4_means(brain, csf_mask, gm_mask, wm_mask), (-1,1))
            
        else: means = np.reshape(find_means(brain, brain_mask, csf_mask, gm_mask, wm_mask), (-1,1))
                
        #Adding a check for the mean values. 
        #We should obtain mean values sufficiently different and in order csf < gm < wm.
        #if not then the mean values will be initialized by the k-means++ algorithm of Scikit-Learn
        if ( (means[0]+0.1) >= means[1] ) or ( (means[1] + 0.1) >= means[2] ) :
            means = None
    
    else: means = None #The k-means++ algorithm of Scikit-Learn will be used to initialize the means
    
    
    if undefined:
        n_classes = 4
        weights = find_prob_4_weights(csf_mask, gm_mask, wm_mask)
        
    else:
        n_classes = 3
        weights = find_prob_weights(csf_mask, gm_mask, wm_mask)
        
        
    
    model = GaussianMixture(
                n_components = n_classes,
                covariance_type = 'full',
                tol = 0.01,
                max_iter = 10000,
                init_params = 'k-means++',
                means_init = means,
                weights_init = weights
                )
            
    
    #updating the funtion to find the labels
    model.fit( np.reshape( brain_array, (-1,1) ) )
    
    #Return 3 probability masks
    if proba:
        label_array = model.predict_proba( np.reshape( brain_array, (-1,1) ) )
        wm = label_de_indexing (label_array[:,2], index_array, brain)
        gm  = label_de_indexing (label_array[:,1], index_array, brain)
        csf  = label_de_indexing (label_array[:,0], index_array, brain)
        
        return wm, gm, csf
    
    else:
        label_array = model.predict( np.reshape( brain_array, (-1,1) ))
        label_image = label_de_indexing (label_array, index_array, brain, 1.)
        return label_image




#############
# Utilities #
#############

def label_selection (label_image, value):
    '''
    This function select only a label value from a label image.
    
    Parameters
    ----------
        label_image: itk image object
            The label image obtained from a segmentation
            
        value: int value.
            The value of the label you want to isolate.
            
    Returns
    -------
        selected_label: itk image object
            The image with only the label of the selected value.
    '''
    
    #Check if the itk types are corrected
    OutputType = itk.Image[itk.F, 3]
    
    if type(label_image) != OutputType:
        
        #eseguo il cast per essere sicuro che la funzione venga applicata
        cast_filter = itk.CastImageFilter[type(label_image), OutputType].New()
        cast_filter.SetInput(label_image)
        cast_filter.Update()
        c_image = cast_filter.GetOutput()
    
        #applico la thresholding
        thresholdFilter = itk.BinaryThresholdImageFilter[OutputType, OutputType].New()
        thresholdFilter.SetInput(c_image)
        thresholdFilter.SetLowerThreshold( value - 0.5 )
        thresholdFilter.SetUpperThreshold( value + 0.5)
        thresholdFilter.SetOutsideValue(0)
        thresholdFilter.SetInsideValue(1)
        thresholdFilter.Update()
    
    
        #eseguo il cast per restituire l'immagine dello stesso tipo dell'input
        cast_filter = itk.CastImageFilter[OutputType, type(label_image)].New()
        cast_filter.SetInput( thresholdFilter.GetOutput() )
        cast_filter.Update()
        
        selected_label = cast_filter.GetOutput()
        
    else:
        
        #applico la thresholding
        thresholdFilter = itk.BinaryThresholdImageFilter[OutputType, OutputType].New()
        thresholdFilter.SetInput( label_image )
        thresholdFilter.SetLowerThreshold( value - 0.5 )
        thresholdFilter.SetUpperThreshold( value + 0.5 )
        thresholdFilter.SetOutsideValue(0)
        thresholdFilter.SetInsideValue(1)
        thresholdFilter.Update()
        
        selected_label = thresholdFilter.GetOutput()
        
        
    return selected_label