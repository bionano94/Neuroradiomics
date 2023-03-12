import itk
import numpy as np

from Neuroradiomics.normalization import *
from Neuroradiomics.registration import *
from Neuroradiomics.resampler import *
from Neuroradiomics.skull_stripping import *
from Neuroradiomics.segmentation import *




#MASK EVALUATION
def evaluate_mask(mask, ground_mask):
    '''
    This function evaluate the goodness of an obtained mask comparing it to a ground truth mask.
    
    Parameters
    ----------
        mask = itk Image object
            The mask you want to evaluate
            
        ground_mask = itk Image object
            The mask you want to use as a ground truth.
            
    Returns
    -------
        results: tuple
            A tuple with the results of the measurements:
             1 = Dice Coefficient
             2 = Volume Similarity
             3 = Hausdorff Distance
             4 = Average Hausdorf Distance
    '''
    
    # Matching the physical space
    matching_filter = match_physical_spaces (ground_mask, mask)
    matching_filter.Update()
    
    #casting the iages to be used by the filters
    ImageType = itk.Image[itk.SS, 3]
    cast_filter = itk.CastImageFilter[type(mask), ImageType].New()
    cast_filter.SetInput(mask)
    cast_filter.Update()
    c_mask = cast_filter.GetOutput()

    cast_filter = itk.CastImageFilter[type(ground_mask), ImageType].New()
    cast_filter.SetInput( matching_filter.GetOutput() )
    cast_filter.Update()
    c_ground_mask = cast_filter.GetOutput()
    
    #use the filter to obtain the desired measures
    overlapping_filter = itk.LabelOverlapMeasuresImageFilter[type(c_mask)].New()
    overlapping_filter.SetSourceImage(c_mask)
    overlapping_filter.SetTargetImage(c_ground_mask)
    overlapping_filter.Update()

    hausdorff_filter = itk.HausdorffDistanceImageFilter[type(c_mask), type(c_ground_mask)].New()
    hausdorff_filter.SetInput1(c_mask)
    hausdorff_filter.SetInput2(c_ground_mask)
    hausdorff_filter.Update()
    
    #create a vetor with all the measures
    results = (overlapping_filter.GetDiceCoefficient(), overlapping_filter.GetVolumeSimilarity(), hausdorff_filter.GetHausdorffDistance (), hausdorff_filter.GetAverageHausdorffDistance ())
    
    return results





# REGISTRATION EVALUATION
def evaluate_registration_mse(fixed_image, deformed_image, ax = None):
    """
    This function find the MSE between the 2 images. It's useful to evaluate the registration.
    
    Parameters
    ----------
    
    fixed_image : itk image object
        The fixed image of your registration.
        
    
    deformed_image : itk image object
        The result of your registration.
        
        
    ax : boolean or int
        The axis you want to compute the mean of the squares on: 0 on columns, 1 on rows.
        Default is None: the mean of the flattered array.
        
        
    Returns
    -------
    
    mse : float object
        The calcuated mse
    """
    
    fix_im_array = itk.GetArrayFromImage(fixed_image)
    def_im_array = itk.GetArrayFromImage(deformed_image)
    
    
    mse = np.mean( np.square( fix_im_array - def_im_array ), axis = ax)
    
    
    return mse