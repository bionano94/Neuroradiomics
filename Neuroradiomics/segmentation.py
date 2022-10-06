import itk
import numpy as np
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
    print ('The estimated weights are: wm = ', weights[0] ,'; gm = ', weights[1] ,'; csf = ', weights[2], '; uncertains = ', weights[3])
    
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


def find_means (brain, wm_mask, gm_mask, csf_mask, prob_threshold = 0.7):
    '''
    This function finds a rough mean for the masked pixels.
    
    Parameters
    ----------
        brain: itk image object.
            The brain image of which you want to evaluate the means values.
            
        wm_mask: itk image object. Must be binary.
            The probability mask for the wm.
            
        gm_mask: itk image object. Must be binary.
            The probability mask for the gm.
            
        csf_mask: itk image object. Must be binary.
            The probability mask for the csf.
            
        prob_threshold: float value.
            The min value of probability you want to consider as valid. Default is 0.51
            
    
    Returns
    -------
        means: list of float.
            The evaluated means.
            means[0] = wm_mean
            means[1] = gm_mean
            means[2] = csf_mean
    '''
    
    min_wm = binarize(wm_mask, prob_threshold)
    min_gm = binarize(gm_mask, prob_threshold)
    min_csf = binarize(csf_mask, prob_threshold)
    
    wm_mean = gaussian_pixel_distribution_params_evaluation(brain, min_wm)[0]
    gm_mean = gaussian_pixel_distribution_params_evaluation(brain, min_gm)[0]
    csf_mean = gaussian_pixel_distribution_params_evaluation(brain, min_csf)[0]
    
    means = [wm_mean, gm_mean, csf_mean]
    
    print ('The estimated means are: wm = ', means[0] ,'; gm = ', means[1] ,'; csf = ' , means[2])
    
    return means


def find_4_means (brain, wm_mask, gm_mask, csf_mask, prob_threshold = 0.7):
    '''
    This function finds a rough mean for the masked pixels. This is useful for the 4 class segmentation, with the uncerain pixels. (not sure if wm or gm)
    
    Parameters
    ----------
        brain: itk image object.
            The brain image of which you want to evaluate the means values.
            
        wm_mask: itk image object. Must be binary.
            The probability mask for the wm.
            
        gm_mask: itk image object. Must be binary.
            The probability mask for the gm.
            
        csf_mask: itk image object. Must be binary.
            The probability mask for the csf.
            
        prob_threshold: float value.
            The min value of probability you want to consider as valid. Default is 0.51
            
    
    Returns
    -------
        means: list of float.
            The evaluated means.
            means[0] = wm_mean
            means[1] = gm_mean
            means[2] = csf_mean
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
    
    means = [wm_mean, gm_mean, csf_mean, idk_mean]
    
    print ('The estimated means are: wm = ', means[0] ,'; gm = ', means[1] ,'; csf = ' , means[2], '; uncertains = ', means[3])
    
    return means




#########################
# Segmentation Function #
#########################


def brain_segmentation ( brain, brain_mask, wm_mask, gm_mask, csf_mask, auto_mean = False, undefined = False ):
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
            It True the segmentation will try to find the mean values for each class. If false are used default ones.
            
            
        undefined: boolean. Default = False.
            It True the segmentation will find also a fourth classe with the not certain pixels.
            
    Returns
    -------
        label_image: itk image object.
            The label image.
            0 is background
            1 is wm
            2 is gm
            3 is csf
            Is uncertain is setted to True 4 are the uncertain pixels.
    
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
    
    
    
    #defining the model to be used to the segmentation
    
    if  auto_mean :
        
        print ('The mean values and the weights will be found automatically')
        #Matching the physical spaces of the masks and the brain
    
        matching_filter = match_physical_spaces(wm_mask, brain)
        matching_filter.Update()
        wm_mask = matching_filter.GetOutput()
    
    
        matching_filter = match_physical_spaces(gm_mask, brain)
        matching_filter.Update()
        gm_mask = matching_filter.GetOutput()
    
        matching_filter = match_physical_spaces(csf_mask, brain)
        matching_filter.Update()
        csf_mask = matching_filter.GetOutput()
        
        if  undefined :
            n_classes = 4
            model = GaussianMixture(
                        n_components = n_classes,
                        covariance_type = 'full',
                        tol = 0.01,
                        max_iter = 10000,
                        means_init = np.reshape( find_4_means ( brain, wm_mask, gm_mask, csf_mask), (-1,1) ),
                        weights_init = find_prob_4_weights (wm_mask, gm_mask, csf_mask) 
                        )
            
        else :
            n_classes = 3
            model = GaussianMixture(
                        n_components = n_classes,
                        covariance_type = 'full',
                        tol = 0.01,
                        max_iter = 1000,
                        means_init = np.reshape( find_means ( brain, wm_mask, gm_mask, csf_mask), (-1,1) ),
                        weights_init = find_prob_weights (wm_mask, gm_mask, csf_mask) 
                        )
    else :
        
        print ('The mean values will be defaults ones and the weights will be found automatically.')
        
        #Default mean values
        wm_mean  = 0.55
        gm_mean  = 0
        csf_mean = -1.5
        
        print ('The mean values used are: wm: ',wm_mean ,'; gm: ',gm_mean ,'; csf: ', csf_mean)
        
        n_classes = 3
        model = GaussianMixture(
                        n_components = n_classes,
                        covariance_type = 'full',
                        tol = 0.01,
                        max_iter = 1000,
                        means_init = np.reshape( (wm_mean, gm_mean, csf_mean), (-1,1) ),
                        weights_init = find_prob_weights (wm_mask, gm_mask, csf_mask) 
                        )
            
    
    #updating the funtion to find the labels
    model.fit( np.reshape( brain_array, (-1,1) ) )
    label_array = model.predict( np.reshape( brain_array, (-1,1) ) )
    
    #transforming the label array into an image. The 1st label value is 1 so wm is 1 and only bg is 0.
    label_image = de_indexing (label_array, index_array, brain, 1)
    
    label_image
    
    print ('Your Brain is segmented')
    
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