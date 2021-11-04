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
    This function generates a 3D blank itk image
    '''
    
    Dimension = 3
    PixelType = itk.F
    ImageType = itk.Image[PixelType, Dimension]
        
        
    Origin = draw(st.tuples(*[st.integers(0., 100.)] * Dimension))
    Size = (draw(st.integers(10, 100)), draw(st.integers(10, 100)), draw(st.integers(10, 100)))
    
    
    image = ImageType.New()
    image = ImageType.New()

    region = itk.ImageRegion[Dimension]()
    region.SetSize(Size)
    region.SetIndex(Origin)

    image.SetRegions(region)
    image.Allocate()
    
    
    return image



################################################################################
###                                 TESTING                                  ###
################################################################################

@given(fixed_image = elastix_registration_strategy(), moving_image = elastix_registration_strategy())
def test_elastix_registration_dimension(fixed_image, moving_image):
    
    '''
    This function tests if the final registered image has the same size of the initial fixed image
    '''
    ImageType = itk.Image[itk.UC, 3]
    registered_image = ImageType.New()
    registered_image, registration_parameters = elastix_registration(fixed_image, moving_image)
    assert (registered_image.GetSize() == fixed_image.GetSize())
    

