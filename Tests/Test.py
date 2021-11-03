#!/usr/bin/env python



import pytest
import hypothesis.strategies as st
from hypothesis import given, settings, assume
from hypothesis import HealthCheck as HC




from Neuroradiomics.registration import registration_reader
from Neuroradiomics.registration import elastix_registration
from Neuroradiomics.registration import registration_writer


import itk



################################################################################
##                          Define Test strategies                            ##
################################################################################


@st.composite
def elastix_registration_strategy(draw):
    
    '''
    This function generates two 2 or 3D blank itk images
    '''
    
    Dimension = draw(st.integer(2,3))
    PixelType = itk.F
    ImageType = itk.Image[PixelType, Dimension]
        
        
    Origin_fixed = draw(st.tuples(*[st.floats(0., 100.)] * Dimension))
    Size_fixed = (draw(st.integers(10, 100)), draw(st.integers(10, 100)), draw(st.integers(10, 100)))
    
    Origin_moving = draw(st.tuples(*[st.floats(0., 100.)] * Dimension))
    Size_moving = (draw(st.integers(10, 100)), draw(st.integers(10, 100)), draw(st.integers(10, 100)))
    

    fixed_image = ImageType.New()
    moving_image = ImageType.New()

    fixed_region = itk.ImageRegion[Dimension]()
    fixed_region.SetSize(Size_fixed)
    fixed_region.SetIndex(Origin_fixed)
    
    moving_region = itk.ImageRegion[Dimension]()
    moving_region.SetSize(Size_moving)
    moving_region.SetIndex(Origin_moving)

    fixed_image.SetRegions(fixed_region)
    fixed_image.Allocate()
    
    moving_image.SetRegions(moving_region)
    moving_image.Allocate()
    
    return fixed_image, moving_image



################################################################################
###                                                                          ###
###                                 TESTING                                  ###
###                                                                          ###
################################################################################

@given(elastix_registration_strategy())
def elastix_registration_dimension_test(fixed_image, moving_image):
    
    '''
    This function tests if the final registered image has the same size of the initial fixed image
    '''
    registered_image, registration_parameters = elastix_registration(fixed_image, moving_image)
    assert (registered_image.GetSize() == fixed_image.GetSize())
    
    









