import pytest
import hypothesis.strategies as st
from hypothesis import given, settings, assume
from hypothesis import HealthCheck as HC


from Neuroradiomics.registration import *
from Neuroradiomics.skull_stripping import *
from Neuroradiomics.normalization import *
from Neuroradiomics.segmentation import *
from Neuroradiomics.evaluation_utilities import *


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
    This function generates an itk image with a 3D white cube of random side. It has to be used as a mask.
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
def label_image_strategy(draw):
    '''
    This function creates a cubic black image with two not overlapping white cubes with random sides in it.
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
    

@st.composite
def cubic_image_strategy(draw):
    '''
    This function generates an itk cubic black image with a random white 3D cube.
    '''
    
    x_max = draw(st.integers(150,200))
    y_max = draw(st.integers(150,200))
    z_max = draw(st.integers(150,200))
    image = np.zeros([x_max, y_max, z_max], np.float32)
    
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
                image[x,y]=100
                
    image = itk.image_view_from_array(image)
    return image



@st.composite
def probability_mask_strategy(draw):
    '''
    This function generates an itk image with a 3D cube with random values in each pixel etween 0 and 1. It has to be used as a probability map.
    '''
    
    x_max = 5
    y_max = 5
    z_max = 5
    image = np.zeros([x_max, y_max, z_max], np.float32)
    
    for x in range (x_max):
        for y in range (y_max):
            for z in range (z_max):
                image[x,y,z] = (draw (st.integers(1, 1000 )))/1000
                
    image = itk.image_view_from_array(image)
    
    return image




#########################################
######## NOT STRATEGY FUNCTIONS #########
#########################################

def binary_uniform_cube():
    '''
    This function create a 3D image with every pixels of value 1.
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

    

def masks_generator():
    '''
    This function 3 cubic black images with a poligon of pixel value = 1. 
    If superimposed on a 4th image the 3 poligons are not overlapping. And fullify the whole image.
    The first two poligons are equals between each other in size and the third one has double the size.
    '''
    
    x_max = 200
    y_max = 200
    z_max = 200
    wm = np.zeros([x_max, y_max, z_max], np.short)
    gm = np.zeros([x_max, y_max, z_max], np.short)
    csf = np.zeros([x_max, y_max, z_max], np.short)

    
    
    x_wm = 50
    x_gm = 100
    
    for x in range (x_wm):
        for y in range (y_max):
            for z in range (z_max):
                wm[x,y,z] = 1
    
    for x in range (x_wm, x_gm):
        for y in range (y_max):
            for z in range (z_max):
                gm[x,y,z] = 1
                
    for x in range (x_gm, x_max):
        for y in range (y_max):
            for z in range (z_max):
                csf[x,y,z] = 1
                
    return itk.image_view_from_array(wm), itk.image_view_from_array(gm), itk.image_view_from_array(csf)




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
@settings(max_examples=5, deadline = None, suppress_health_check = (HC.too_slow, HC.large_base_example, HC.data_too_large))
def test_indexing_len (image, mask):
    '''
    This function index an image and tests if the obtained image_array has the same length of the index_array.
    '''
    
    image_array, index_array = indexing(image, mask)
        
    assert len(image_array) == len(index_array)

    
    
@given (image = masking_random_image_strategy(), mask = masking_cube_mask_strategy())
@settings(max_examples=5, deadline = None, suppress_health_check = (HC.too_slow, HC.large_base_example, HC.data_too_large))
def test_indexing_shape (image, mask):
    '''
    This function index an image and tests if the obtained index_array has the same shape of the index_array and is proper for a 3D image.
    '''
    
    image_array, index_array = indexing(image, mask)
        
    assert np.shape(image_array)[0] == np.shape(index_array)[0]
    assert np.shape(index_array)[1] == 3 #because it is a 3D image

    
@given (image = masking_random_image_strategy(), mask = masking_cube_mask_strategy())
@settings(max_examples=5, deadline = None, suppress_health_check = (HC.too_slow, HC.large_base_example, HC.data_too_large))
def test_indexing_non_zero (image, mask):
    '''
    This function index an image and tests if the obtained image_array has the same number of non-zero pixels of an image masked with the same mask.
    '''
    
    image_array, index_array = indexing(image, mask)
    
    masked_image = masked_image = negative_3d_masking (image, mask)

    assert np.count_nonzero(itk.GetArrayFromImage(masked_image)) == np.count_nonzero(image_array)
    
    
    
#Testing the de-indexing function

@given (image = masking_random_image_strategy(), mask = masking_cube_mask_strategy())
@settings(max_examples=10, deadline = None, suppress_health_check = (HC.too_slow, HC.large_base_example, HC.data_too_large))
def test_de_indexing (image, mask):
    '''
    This function checks if indexing and then de-indexing an image the final image has the same Size and spacing of the original image.
    '''
    
    image_array, index_array = indexing(image, mask)
    
    de_indexed_image = label_de_indexing (image_array, index_array, image)
    
    masked_image = negative_3d_masking (image, mask)
    
    #The Same Size
    assert np.all(de_indexed_image.GetLargestPossibleRegion().GetSize() == image.GetLargestPossibleRegion().GetSize())
    #Same Spacing
    assert np.all(de_indexed_image.GetSpacing() == image.GetSpacing())
    
    
    
    
#Testing if indexing and de-indexing give the same image

@given (image = cubic_image_strategy())
@settings(max_examples=10, deadline = None, suppress_health_check = (HC.too_slow, HC.large_base_example, HC.data_too_large))
def test_index_de_index_validation (image):
    '''
    This function checks if indexing and then de-indexing an image the result is equal to the original image.
    '''
    
    mask = binary_uniform_cube()
    image_array, index_array = indexing(image, mask) 
    
    de_indexed_image = label_de_indexing (image_array, index_array, image)
    
    
    #same image
    assert np.all( np.isclose( itk.GetArrayFromImage(de_indexed_image), itk.GetArrayFromImage(image), 1e-03, 1e-03))
    
    
    
    
###################
# Means Functions #
###################
    
    
#Testing the Gaussian function

@given (mask = masking_cube_mask_strategy())
@settings(max_examples=5, deadline = None, suppress_health_check = (HC.too_slow, HC.large_base_example, HC.data_too_large))
def test_gaussian_prameters_mean_1 (mask):
    '''
    This function checks if obtaining the mean value of an image with every pixel has value 1 then the resulting mean is 1.
    '''
    
    image1 = binary_uniform_cube
    params1 = gaussian_pixel_distribution_params_evaluation(image1, mask)
        
    assert params1[0] == 1

    
@given (mask = masking_cube_mask_strategy())
@settings(max_examples=5, deadline = None, suppress_health_check = (HC.too_slow, HC.large_base_example, HC.data_too_large))
def test_gaussian_prameters_std_1 (mask):
    '''
    This function checks if obtaining the std dev value of an image with every pixel has value 1 then the resulting std dev is 10
    '''
    
    image1 = binary_uniform_cube
    params1 = gaussian_pixel_distribution_params_evaluation(image1, mask)
        
    assert params1[1] == 0
    
    

@given (image2 = masking_cube_mask_strategy(), mask = masking_cube_mask_strategy())
@settings(max_examples=5, deadline = None, suppress_health_check = (HC.too_slow, HC.large_base_example, HC.data_too_large))
def test_gaussian_prameters_mean_random (image2, mask):
    '''
    This function checks if obtaining the mean value of a random image the mean value is greater or equal to 1 and smaller or equal to 0.
    '''
    
    params2 = gaussian_pixel_distribution_params_evaluation(image2, mask)
    
    
    assert params2[0] <= 1
    assert params2[0] >= 0
    assert params2[1] >= 0
    
    
    
@given (image2 = masking_cube_mask_strategy(), mask = masking_cube_mask_strategy())
@settings(max_examples=5, deadline = None, suppress_health_check = (HC.too_slow, HC.large_base_example, HC.data_too_large))
def test_gaussian_prameters_std_random (image2, mask):
    '''
    This function checks if obtaining the std dev value of a random image the std dev value is greater or equal to 0.
    '''
    
    params2 = gaussian_pixel_distribution_params_evaluation(image2, mask)
    
    assert params2[1] >= 0
    
#####################
# Weights Functions #
#####################


#3 classes weights function    
@given (mask1 = probability_mask_strategy(), mask2 = probability_mask_strategy(), mask3 = probability_mask_strategy() )
@settings(max_examples = 10, deadline = None, suppress_health_check = (HC.too_slow, HC.large_base_example, HC.data_too_large))
def test_weights_function_range (mask1, mask2, mask3):
    '''
    This function tests if all the wheights obtained are in the right range (<= 1 and >= 0 )
    '''
    
    
    weights = find_prob_weights(mask1, mask2, mask3)
    
    #every wheights in range [0;1]
    assert weights[0] <= 1
    assert weights[0] >= 0
    assert weights[1] <= 1
    assert weights[1] >= 0
    assert weights[2] <= 1
    assert weights[2] >= 0

    

@given (mask1 = probability_mask_strategy(), mask2 = probability_mask_strategy(), mask3 = probability_mask_strategy() )
@settings(max_examples = 10, deadline = None, suppress_health_check = (HC.too_slow, HC.large_base_example, HC.data_too_large))
def test_weights_function_sum (mask1, mask2, mask3):
    '''
    This function tests if the sum of al the weights is 1.
    '''
    
    weights = find_prob_weights(mask1, mask2, mask3)

    #Weights sum = 1.  isclose to get around problems with approximation.
    assert np.isclose( (weights[0] + weights[1] + weights[2] ), 1)
    
    

def test_weights_function_value ():
    '''
    This function tests if giving 3 known masks the computed weights are the same of those manually computed.
    '''
    
    mask1, mask2, mask3 = masks_generator()
    weights = find_prob_weights(mask1, mask2, mask3)

    assert weights[0] == 0.25
    assert weights[1] == 0.25
    assert weights[2] == 0.5

    
#4 classes weights function    
@given (mask1 = probability_mask_strategy(), mask2 = probability_mask_strategy(), mask3 = probability_mask_strategy() )
@settings(max_examples = 10, deadline = None, suppress_health_check = (HC.too_slow, HC.large_base_example, HC.data_too_large))
def test_4_weights_function_range (mask1, mask2, mask3):
    '''
    This function tests if all the wheights obtained are in the right range (<= 1 and >= 0 )
    '''
        
    weights = find_prob_4_weights(mask1, mask2, mask3)
    
    
    #every wheights in range [0;1]
    assert weights[0] <= 1
    assert weights[0] >= 0
    assert weights[1] <= 1
    assert weights[1] >= 0
    assert weights[2] <= 1
    assert weights[2] >= 0
    assert weights[3] <= 1
    assert weights[3] >= 0

    
    
    
#4 classes weights function    
@given (mask1 = probability_mask_strategy(), mask2 = probability_mask_strategy(), mask3 = probability_mask_strategy() )
@settings(max_examples = 10, deadline = None, suppress_health_check = (HC.too_slow, HC.large_base_example, HC.data_too_large))
def test_4_weights_function_sum (mask1, mask2, mask3):
    '''
    This function tests if the sum of al the weights is 1.
    '''
    
    weights = find_prob_4_weights(mask1, mask2, mask3)


    #Weights sum = 1.  isclose to get around problems with approximation.
    assert np.isclose( (weights[0] + weights[1] + weights[2] + weights[3]), 1)
    
    
#######################
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

