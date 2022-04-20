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