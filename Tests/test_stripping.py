

import pytest
import hypothesis.strategies as st
from hypothesis import given, settings, assume
from hypothesis import HealthCheck as HC


from Neuroradiomics.registration import *
from Neuroradiomics.skull_stripping import *
from Neuroradiomics.normalization import *


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
def random_image_strategy(draw):
    
    '''
    This function generates a 3D random itk image
    '''
    
    PixelType = itk.F
    ImageType = itk.Image[PixelType, 3]
    Size = ( draw(st.integers(10,100)), draw(st.integers(10,100)) , draw(st.integers(10,100)) )

    rndImage = itk.RandomImageSource[ImageType].New()
    rndImage.SetSize(Size)
    rndImage.SetNumberOfWorkUnits(1)
    rndImage.UpdateLargestPossibleRegion()

    return rndImage.GetOutput()


@st.composite
def cube_random_image_strategy(draw):
    '''
    This function generates an itk image with a 3D cube of random pixel values.
    '''
    
    x_max = 20
    y_max = 20
    z_max = 20
    image = np.zeros([x_max, y_max, z_max])
    
    for x in range (x_max):
        for y in range (y_max):
            for z in range (z_max):
                image[x,y,z] = draw(st.integers(1,10))
                
    image = itk.image_view_from_array(image)
    
    return image



@st.composite
def masking_cube_mask_strategy(draw):
    '''
    This function generates an itk image with a 3D cube of random side. It has to be used as a mask.
    '''
    
    x_max = 200
    y_max = 200
    z_max = 200
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
                image[x,y,z] = 1
                
    image = itk.image_view_from_array(image)
    
    return image




############################################
######### NOT STRATEGY FUNCTIONS ###########
############################################

def binary_uniform_cube_image():
    '''
    This function generates an itk image with a 3D cube full of ones. It has to be used as a mask.
    '''
    
    x_max = 20
    y_max = 20
    z_max = 20
    image = np.zeros([x_max, y_max, z_max])
    
    for x in range (x_max):
        for y in range (y_max):
            for z in range (z_max):
                image[x,y,z] = 1
                
    image = itk.image_view_from_array(image)
    
    return image



def masking_random_image():
    '''
    This function generates a 3D random itk image.
    '''
    
    PixelType = itk.F
    ImageType = itk.Image[PixelType, 3]
    Size = (200, 200, 200)

    rndImage = itk.RandomImageSource[ImageType].New()
    rndImage.SetSize(Size)
    rndImage.SetNumberOfWorkUnits(1)
    rndImage.UpdateLargestPossibleRegion()

    return rndImage.GetOutput()



def four_level_image():
    '''
    This function generates a 3D itk image with 4 pixel levels.
    '''
    
    x_max = 20
    y_max = 20
    z_max = 20
    image = np.zeros([x_max, y_max, z_max])
    
    for x in range (x_max):
        for y in range (y_max):
            for z in range (5):
                image[x,y,z] = 0
            for z in range (5,10):    
                image[x,y,z] = 1
            for z in range (10,15):    
                image[x,y,z] = 2
            for z in range (15,20):    
                image[x,y,z] = 3

    return itk.image_view_from_array(image)



def black_image_one_white_point(x, y, z):
    '''
    This function create a black cubic image with only one white point.
    
    Paramters
    ---------
        x: int number
            first coordinate of the white point
        
        y: int number
            second cordinate of the white point
        
        z: int number
            third cordinate of the white point
            
    Returns
    -------
        image: itk Image object
            The image created.
    '''
    x_max = 10
    y_max = 10
    z_max = 10
    image_array = np.zeros([x_max, y_max, z_max], np.float32)
    
    image_array[x,y,z] = 1
                
    image = itk.image_view_from_array(image_array)
    
    return image



# ████████ ███████ ███████ ████████ ███████ 
#    ██    ██      ██         ██    ██      
#    ██    █████   ███████    ██    ███████
#    ██    ██           ██    ██         ██
#    ██    ███████ ███████    ██    ███████





#negative masking function

@given (mask = masking_cube_mask_strategy())
@settings(max_examples = 10, deadline = None)
def test_negative_masking_attributes(mask):
    '''
    This function tests if an image obtained with the negative_3d_masking function has the same attributes of the original image.
    '''
    
    image = masking_random_image()
    masked_image = negative_3d_masking(image, mask)
    
    index = itk.Index[3]()
    
    
    for index[0] in range( mask.GetLargestPossibleRegion().GetSize()[0] ):
        for index[1] in range( mask.GetLargestPossibleRegion().GetSize()[1] ):
            for index[2] in range( mask.GetLargestPossibleRegion().GetSize()[2] ):  
                if mask.GetPixel(index) < 0.5:
                    assert np.isclose(masked_image.GetPixel(index), 0 )
                
 
    assert np.all(image.GetLargestPossibleRegion().GetSize() == masked_image.GetLargestPossibleRegion().GetSize())
    assert np.all( image.GetSpacing() == masked_image.GetSpacing() )
    assert np.all( image.GetOrigin() == masked_image.GetOrigin() )
    assert np.all( image.GetDirection() == masked_image.GetDirection() )
    
    
@given (mask = masking_cube_mask_strategy())
@settings(max_examples = 10, deadline = None)
def test_negative_masking_bg(mask):
    '''
    This function tests if an image obtained with the negative_3d_masking function has pixels with 0 value where the mask has a value < 0.5.
    '''
    
    image = masking_random_image()
    masked_image = negative_3d_masking(image, mask)
    
    index = itk.Index[3]()
    
    
    for index[0] in range( mask.GetLargestPossibleRegion().GetSize()[0] ):
        for index[1] in range( mask.GetLargestPossibleRegion().GetSize()[1] ):
            for index[2] in range( mask.GetLargestPossibleRegion().GetSize()[2] ):  
                if mask.GetPixel(index) < 0.5:
                    assert np.isclose(masked_image.GetPixel(index), 0 )
    

    
#Test the masking with a full ones mask
@given (image = cube_random_image_strategy())
@settings(max_examples = 10, deadline = None, suppress_health_check = (HC.too_slow, HC.large_base_example, HC.data_too_large))
def test_negative_masking_validation(image):
    '''
    This function tests if masking an image with a total white mask the output image is the same of the one in input.
    '''
    
    mask = binary_uniform_cube_image()
    masked_image = negative_3d_masking(image, mask)

    
    assert np.all(np.isclose(image, masked_image))
    
    
    
#masking function

@given(mask = masking_cube_mask_strategy())
@settings(max_examples=10, deadline = None)
def test_masking_attributes(mask):
    '''
    This function tests if an image obtained with the negative_3d_masking function has the same attributes of the original image.
    '''
    
    image = masking_random_image()
    masked_image = masking(image, mask)
    
    index = itk.Index[3]()           
 
    assert np.all(image.GetLargestPossibleRegion().GetSize() == masked_image.GetLargestPossibleRegion().GetSize())
    assert np.all( image.GetSpacing() == masked_image.GetSpacing() )
    assert np.all( image.GetOrigin() == masked_image.GetOrigin() )
    assert np.all( image.GetDirection() == masked_image.GetDirection() )
    
    
    

@given(mask = masking_cube_mask_strategy())
@settings(max_examples=10, deadline = None)
def test_masking_bg(mask):
    '''
    This function tests if an image obtained with the negative_3d_masking function has pixels with 0 value where the mask has a value > 0.5.
    '''
    
    image = masking_random_image()
    masked_image = masking(image, mask)
    
    index = itk.Index[3]()
    
    
    for index[0] in range( mask.GetLargestPossibleRegion().GetSize()[0] ):
        for index[1] in range( mask.GetLargestPossibleRegion().GetSize()[1] ):
            for index[2] in range( mask.GetLargestPossibleRegion().GetSize()[2] ):
                if mask.GetPixel(index) > 0.5:
                    assert np.isclose(masked_image.GetPixel(index), 0 )

    

    

#Test the masking with a full ones mask
@given (image = cube_random_image_strategy())
@settings(max_examples = 5, deadline = None, suppress_health_check = (HC.too_slow, HC.large_base_example, HC.data_too_large))
def test_masking_validation(image):
    '''
    This function tests if masking an image with a total white mask the output image is totally black.
    '''
    
    mask = binary_uniform_cube_image()
    masked_image = masking(image, mask)

    
    assert np.all(np.isclose(masked_image, 0.))
    
    
    
    
#Binarize low
@given (image = random_image_strategy())
@settings(max_examples=10, deadline = None)
def test_binarize_attributes(image):
    '''
    This function tests if the output of the binary function has the same attributes of the input image
    '''
    
    bin_image = binarize(image)
    
    assert np.all( image.GetSpacing() == bin_image.GetSpacing() )
    assert np.all( image.GetOrigin() == bin_image.GetOrigin() )
    assert np.all( image.GetDirection() == bin_image.GetDirection() )
    
    
#Binarize low
@given (image = random_image_strategy())
@settings(max_examples=10, deadline = None)
def test_binarize_binary(image):
    '''
    This function tests if the output of the binary image is really binary
    '''
    
    bin_image = binarize(image)
    
    assert np.all( (itk.GetArrayFromImage(bin_image) == 0) | (itk.GetArrayFromImage(bin_image) == 1) )


    

def test_binarize_full_white():
    '''
    This function tests if giving in input an image with every pixel =1 the output is the same image as the input.
    '''
    image = binary_uniform_cube_image()
    bin_image = binarize(image)
    
    assert np.all( itk.GetArrayFromImage(bin_image) == itk.GetArrayFromImage(image) )
    
    
    
    
#Binarize hi
@given (image = random_image_strategy())
@settings(max_examples=10, deadline = None)
def test_binarize_double_extremes_binary(image):
    '''
    This function tests if the output of the binary image is really binary even when an high value is inserted.
    '''
    
    bin_image = binarize(image, 1, 100)
    
    assert np.all( (itk.GetArrayFromImage(bin_image) == 0) | (itk.GetArrayFromImage(bin_image) == 1) )

#Binarize hi
@given (image = random_image_strategy())
@settings(max_examples=5, deadline = None)
def test_binarize_double_extremes_attributes(image):
    '''
    This function tests if the output of the binary function has the same attributes of the input image even when an high value is inserted.
    '''
    
    bin_image = binarize(image, 1, 100)
    
    assert np.all( image.GetSpacing() == bin_image.GetSpacing() )
    assert np.all( image.GetOrigin() == bin_image.GetOrigin() )
    assert np.all( image.GetDirection() == bin_image.GetDirection() )
    

#Binarize hi

def test_binarize_double_extremes_one():
    '''
    This function tests if the output of the binarized image is 1 for values inside the range and 0 for values outside.
    '''
    
    image = four_level_image()
    bin_image = binarize(image, 1, 2)
    
    index = itk.Index[3]()

    for index[0] in range( image.GetLargestPossibleRegion().GetSize()[0] ):
        for index[1] in range( image.GetLargestPossibleRegion().GetSize()[1] ):
            for index[2] in range( image.GetLargestPossibleRegion().GetSize()[2] ):
                if image.GetPixel(index) >= 1 and image.GetPixel(index) <= 2 :
                    assert np.isclose(bin_image.GetPixel(index), 1 )
                else :
                    assert np.isclose(bin_image.GetPixel(index), 0 )
    
    
#Binay Dilating
@given (image = random_image_strategy())
@settings(max_examples=10, deadline = None)
def test_binary_dilating_attributes(image):
    '''
    This function tests the opening function gives in output an image with the same attributes of the one in input.
    '''
    
    bin_image = binarize(image)
    
    dilated_image = binary_dilating(bin_image)
    
    assert np.all( image.GetSpacing() == dilated_image.GetSpacing() )
    assert np.all( image.GetOrigin() == dilated_image.GetOrigin() )
    assert np.all( image.GetDirection() == dilated_image.GetDirection() )
    assert np.all( image.GetLargestPossibleRegion().GetSize() == dilated_image.GetLargestPossibleRegion().GetSize())


    
def test_binary_dilating_functionality():
    '''
    This function checks if dilating an image with only one pixel of value 1 with a ball of radius 1 the output is equal to the one manually computed.
    '''
    
    image = black_image_one_white_point(2,2,2)
    
    dilated_image = binary_dilating(image)
    
    assert np.count_nonzero(itk.GetArrayFromImage(dilated_image) ) == 19
    
    
#Binary Eroding    
@given (image = random_image_strategy())
@settings(max_examples=20, deadline = None)
def test_binary_eroding(image):
    '''
    This function tests the opening function gives in output an image with the same attributes of the one in input.
    '''
    
    bin_image = binarize(image)
    
    dilated_image = binary_eroding(bin_image)
    
    assert np.all( image.GetSpacing() == dilated_image.GetSpacing() )
    assert np.all( image.GetOrigin() == dilated_image.GetOrigin() )
    assert np.all( image.GetDirection() == dilated_image.GetDirection() )
    assert np.all( image.GetLargestPossibleRegion().GetSize() == dilated_image.GetLargestPossibleRegion().GetSize())
    

def test_binary_eroding_functionality():
    '''
    This function checks if eroding an image with only one pixel of value 1 with a ball of radius 1 the output is equal to the one manually computed.
    '''
    
    image = black_image_one_white_point(2,2,2)
    
    dilated_image = binary_eroding(image)
    
    assert np.count_nonzero(itk.GetArrayFromImage(dilated_image) ) == 0
    
    

#Largest Connected Region  
@given (image = random_image_strategy())
@settings(max_examples=20, deadline = None)
def test_largest_connected_region(image):
    '''
    This function tests the opening function.
    '''
    
    bin_image = binarize(image)
    
    final_image = find_largest_connected_region(bin_image)
    
    assert np.all( (itk.GetArrayFromImage(final_image) == 0) | (itk.GetArrayFromImage(final_image) == 1) )
    assert np.all( image.GetSpacing() == final_image.GetSpacing() )
    assert np.all( image.GetOrigin() == final_image.GetOrigin() )
    assert np.all( image.GetDirection() == final_image.GetDirection() )
    assert np.all( image.GetLargestPossibleRegion().GetSize() == final_image.GetLargestPossibleRegion().GetSize())
    assert np.any( itk.GetArrayFromImage(final_image) == 1 )
    