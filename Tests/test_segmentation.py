import pytest
import hypothesis.strategies as st
from hypothesis import given, settings, assume
from hypothesis import HealthCheck as HC


from Neuroradiomics.registration import *
from Neuroradiomics.skull_stripping import *
from Neuroradiomics.normalization import *
from Neuroradiomics.segmentation import *


import itk
import numpy as np
import os
from datetime import datetime
import glob




# ███████ ████████ ██████   █████  ████████ ███████  ██████  ██ ███████ ███████
# ██         ██    ██   ██ ██   ██    ██    ██      ██       ██ ██      ██
# ███████    ██    ██████  ███████    ██    █████   ██   ███ ██ █████   ███████
#      ██    ██    ██   ██ ██   ██    ██    ██      ██    ██ ██ ██           ██
# ███████    ██    ██   ██ ██   ██    ██    ███████  ██████  ██ ███████ ███████



@st.composite
def masking_random_image_strategy(draw):
    
    '''
    This function generates a 3D random itk image.
    '''
    
    PixelType = itk.US
    ImageType = itk.Image[PixelType, 3]
    Size = (200, 200, 200)

    rndImage = itk.RandomImageSource[ImageType].New()
    rndImage.SetSize(Size)
    rndImage.SetNumberOfWorkUnits(1)
    rndImage.UpdateLargestPossibleRegion()

    return rndImage.GetOutput()

@st.composite
def masking_cube_mask_strategy(draw):
    '''
    This function generates an itk image with a 3D cube of random side. It has to be used as a mask.
    '''
    
    x_max = 200
    y_max = 200
    z_max = 200
    image = np.zeros([x_max, y_max, z_max], np.short)
    
    side = draw (st.integers(40,60))
    x1 = draw (st.integers(10,40))
    y1 = draw (st.integers(10,40))
    z1 = draw (st.integers(10,40))
    x2 = x1 + side
    y2 = y1 + side
    z2 = z1 + side
    
    for x in range (x1, x2):
        for y in range (y1, y2):
            for z in range (z1, z2):
                image[x,y,z] = 1
                
    image = itk.image_view_from_array(image)
    
    return image


@st.composite
def binary_uniform_cube_image_strategy(draw):
    '''
    This function generates an itk image with a 3D cube of random side. It has to be used as a mask.
    '''
    
    x_max = 200
    y_max = 200
    z_max = 200
    image = np.zeros([x_max, y_max, z_max], np.short)
    
    for x in range (x_max):
        for y in range (y_max):
            for z in range (z_max):
                image[x,y,z] = 1
                
    image = itk.image_view_from_array(image)
    
    return image

@st.composite
def label_image_strategy(draw):
    '''
    '''
    
    x_max = 300
    y_max = 300
    z_max = 300
    image = np.zeros([x_max, y_max, z_max], np.short)
    
    side1 = draw (st.integers(40,60))
    x11 = draw (st.integers(10,40))
    y11 = draw (st.integers(10,40))
    z11 = draw (st.integers(10,40))
    x21 = x11 + side1
    y21 = y11 + side1
    z21 = z11 + side1
    
    for x in range (x11, x21):
        for y in range (y11, y21):
            for z in range (z11, z21):
                image[x,y,z] = 1
    
    
    side2 = draw (st.integers(40,60))
    x12 = draw (st.integers(110,140))
    y12 = draw (st.integers(110,140))
    z12 = draw (st.integers(110,140))
    x22 = x21 + side2
    y22 = y21 + side2
    z22 = z21 + side2
    
    for x in range (x21, x22):
        for y in range (y21, y22):
            for z in range (z21, z22):
                image[x,y,z] = 1
                
    image = itk.image_view_from_array(image)
    
    return image
    



# ████████ ███████ ███████ ████████ ███████ 
#    ██    ██      ██         ██    ██      
#    ██    █████   ███████    ██    ███████
#    ██    ██           ██    ██         ██
#    ██    ███████ ███████    ██    ███████



######################
# Indexing Functions #
######################


#Testing the indexing function

@given (image = masking_random_image_strategy(), mask = masking_cube_mask_strategy())
@settings(max_examples=20, deadline = None)
def test_indexing (image, mask):
    
    image_array, index_array = indexing(image, mask)
    
    masked_image = masked_image = negative_3d_masking (image, mask)
    
    assert len(image_array) == len(index_array)
    assert np.shape(image_array)[0] == np.shape(index_array)[0]
    assert np.shape(index_array)[1] == 3
    assert np.count_nonzero(itk.GetArrayFromImage(masked_image)) == np.count_nonzero(image_array)
    

    
    
#Testing the de-indexing function

@given (image = masking_random_image_strategy(), mask = masking_cube_mask_strategy())
@settings(max_examples=20, deadline = None)
def test_de_indexing (image, mask):
    
    image_array, index_array = indexing(image, mask)
    
    de_indexed_image = de_indexing (image_array, index_array, image)
    
    masked_image = negative_3d_masking (image, mask)
    
    #The Same Size
    assert np.all(de_indexed_image.GetLargestPossibleRegion().GetSize() == image.GetLargestPossibleRegion().GetSize())
    #Same Spacing
    assert np.all(de_indexed_image.GetSpacing() == image.GetSpacing())
    

    
###################
# Means Functions #
###################
    
    
#Testing the Gaussian function

@given (image1 = binary_uniform_cube_image_strategy(), image2 = masking_cube_mask_strategy(), mask = masking_cube_mask_strategy())
@settings(max_examples=20, deadline = None, suppress_health_check = (HC.too_slow,))
def test_gaussian_prameters (image1, image2, mask):
    
    params1 = gaussian_pixel_distribution_params_evaluation(image1, mask)
    
    params2 = gaussian_pixel_distribution_params_evaluation(image2, mask)
    
    assert params1[0] == 1
    assert params1[1] == 0
    assert params2[0] <= 1
    assert params2[0] >= 0
    assert params2[1] >= 0
    
    
    
#####################à#
# Utilities Functions #
#######################


#Testing the label_selection function
@given (labels = label_image_strategy(), value = st.integers( 0, 2 ) ) 
@settings(max_examples=20, deadline = None, suppress_health_check = (HC.too_slow,))
def test_label_selection (labels, value):
    
    selected_label = label_selection(labels, value)
    
    selected_label_array = itk.GetArrayFromImage(selected_label)
    labels_array = itk.GetArrayFromImage(labels)
    
    check_boolean_vector = np.where(labels_array == value, selected_label_array == 1, selected_label_array == 0)
    
    assert np.all( np.logical_or( selected_label_array == 0, selected_label_array == 1) )
    assert np.all(check_boolean_vector)
