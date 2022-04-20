

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
def masking_random_image_strategy(draw):
    
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




# ████████ ███████ ███████ ████████ ███████ 
#    ██    ██      ██         ██    ██      
#    ██    █████   ███████    ██    ███████
#    ██    ██           ██    ██         ██
#    ██    ███████ ███████    ██    ███████





#negative masking function

@given (image = masking_random_image_strategy(), mask = masking_cube_mask_strategy())
@settings(max_examples=20, deadline = None)
def test_negative_masking(image, mask):
    '''
    This function tests the negative_3d_masking function
    '''
    
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
    
    
#masking function

@given (image = masking_random_image_strategy(), mask = masking_cube_mask_strategy())
@settings(max_examples=20, deadline = None)
def test_masking(image, mask):
    '''
    This function tests the negative_3d_masking function
    '''
    
    masked_image = masking(image, mask)
    
    index = itk.Index[3]()
    
    
    for index[0] in range( mask.GetLargestPossibleRegion().GetSize()[0] ):
    
        for index[1] in range( mask.GetLargestPossibleRegion().GetSize()[1] ):
        
            for index[2] in range( mask.GetLargestPossibleRegion().GetSize()[2] ):
                
                if mask.GetPixel(index) > 0.5:
                    assert np.isclose(masked_image.GetPixel(index), 0 )
                
 
    assert np.all(image.GetLargestPossibleRegion().GetSize() == masked_image.GetLargestPossibleRegion().GetSize())
    assert np.all( image.GetSpacing() == masked_image.GetSpacing() )
    assert np.all( image.GetOrigin() == masked_image.GetOrigin() )
    assert np.all( image.GetDirection() == masked_image.GetDirection() )
    
    

#Binarize low
@given (image = random_image_strategy())
@settings(max_examples=20, deadline = None)
def test_binarize(image):
    '''
    This function tests the binarize funtion
    '''
    
    bin_image = binarize(image)
    
    assert np.all( (itk.GetArrayFromImage(bin_image) == 0) | (itk.GetArrayFromImage(bin_image) == 1) )
    assert np.all( image.GetSpacing() == bin_image.GetSpacing() )
    assert np.all( image.GetOrigin() == bin_image.GetOrigin() )
    assert np.all( image.GetDirection() == bin_image.GetDirection() )
    
    
#Binarize hi
@given (image = random_image_strategy())
@settings(max_examples=20, deadline = None)
def test_binarize_double_extremes(image):
    '''
    This function tests the binarize funtion
    '''
    
    bin_image = binarize(image, 1, 100)
    
    assert np.all( (itk.GetArrayFromImage(bin_image) == 0) | (itk.GetArrayFromImage(bin_image) == 1) )
    assert np.all( image.GetSpacing() == bin_image.GetSpacing() )
    assert np.all( image.GetOrigin() == bin_image.GetOrigin() )
    assert np.all( image.GetDirection() == bin_image.GetDirection() )

    
#Thresholding
@given (image = random_image_strategy(), value = st.floats(0,10))
@settings(max_examples=20, deadline = None)
def test_normal_threshold(image, value):
    '''
    This function tests the thresholding funtion
    '''
    
    
    final_image = normal_threshold(image, value)
    
    index = itk.Index[3]()
    
    
    for index[0] in range( image.GetLargestPossibleRegion().GetSize()[0] ):
    
        for index[1] in range( image.GetLargestPossibleRegion().GetSize()[1] ):
        
            for index[2] in range( image.GetLargestPossibleRegion().GetSize()[2] ):
                
                if (image.GetPixel(index) > value or image.GetPixel(index) < -value):
                    assert np.isclose(final_image.GetPixel(index), 0 )
                else: assert np.isclose(final_image.GetPixel(index), 1 )
    
    assert np.all( (itk.GetArrayFromImage(final_image) == 0) | (itk.GetArrayFromImage(final_image) == 1) )
    assert np.all( image.GetSpacing() == final_image.GetSpacing() )
    assert np.all( image.GetOrigin() == final_image.GetOrigin() )
    assert np.all( image.GetDirection() == final_image.GetDirection() )
    

#Binay Opening
@given (image = random_image_strategy())
@settings(max_examples=20, deadline = None)
def test_binary_opening(image):
    '''
    This function tests the opening function.
    '''
    
    bin_image = binarize(image)
    
    dilated_image = binary_opening(bin_image)
    
    assert np.all( (itk.GetArrayFromImage(dilated_image) == 0) | (itk.GetArrayFromImage(dilated_image) == 1) )
    assert np.all( image.GetSpacing() == dilated_image.GetSpacing() )
    assert np.all( image.GetOrigin() == dilated_image.GetOrigin() )
    assert np.all( image.GetDirection() == dilated_image.GetDirection() )
    assert np.all( image.GetLargestPossibleRegion().GetSize() == dilated_image.GetLargestPossibleRegion().GetSize())

    
#Binary Eroding    
@given (image = random_image_strategy())
@settings(max_examples=20, deadline = None)
def test_binary_eroding(image):
    '''
    This function tests the opening function.
    '''
    
    bin_image = binarize(image)
    
    eroded_image = binary_eroding(bin_image)
    
    assert np.all( (itk.GetArrayFromImage(eroded_image) == 0) | (itk.GetArrayFromImage(eroded_image) == 1) )
    assert np.all( image.GetSpacing() == eroded_image.GetSpacing() )
    assert np.all( image.GetOrigin() == eroded_image.GetOrigin() )
    assert np.all( image.GetDirection() == eroded_image.GetDirection() )
    assert np.all( image.GetLargestPossibleRegion().GetSize() == eroded_image.GetLargestPossibleRegion().GetSize())
    
    

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
    