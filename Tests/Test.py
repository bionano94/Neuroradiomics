#!/usr/bin/env python


import pytest
import hypothesis.strategies as st
from hypothesis import given, settings, assume
from hypothesis import HealthCheck as HC


from Neuroradiomics.registration import registration_reader
from Neuroradiomics.registration import elastix_registration
from Neuroradiomics.registration import registration_writer


import itk
import numpy as np
import os



################################################################################
##                          Define Test strategies                            ##
################################################################################


@st.composite
def elastix_IO_strategy(draw):
    
    '''
    This function generates a 3D random itk image
    '''
    
    Dimension = 3
    PixelType = itk.F
    ImageType = itk.Image[PixelType, Dimension]
        
        
    Origin = draw(st.tuples(*[st.integers(0, 100)] * Dimension))
    Size = ( draw(st.integers(10, 100)), draw(st.integers(10, 100)), draw(st.integers(10, 100)) )
    
    region = itk.ImageRegion[Dimension]()
    region.SetSize(Size)
    region.SetIndex(Origin)
    
    RandomSourceType = itk.RandomImageSource[ImageType]
    randomImageSource = RandomSourceType.New()
    randomImageSource.SetSize(Size)
    randomImageSource.GetOutput().SetRequestedRegion(region)
    

    image = randomImageSource.GetOutput()
    

    image.SetRegions(region)
    image.Allocate() 
    
    
    return image

@st.composite
def random_square_strategy(draw):
    
    '''
    This function generates a 3D random square itk image
    '''
    
    Dimension = 3
    PixelType = itk.F
    ImageType = itk.Image[PixelType, Dimension]
        
    
    axis_size = draw (st.integers(10,100))
    Origin = draw(st.tuples(*[st.integers(0, 100)] * Dimension))
    Size = ( axis_size, axis_size, axis_size )
    
    region = itk.ImageRegion[Dimension]()
    region.SetSize(Size)
    region.SetIndex(Origin)
    
    RandomSourceType = itk.RandomImageSource[ImageType]
    randomImageSource = RandomSourceType.New()
    randomImageSource.SetSize(Size)
    randomImageSource.GetOutput().SetRequestedRegion(region)
    

    image = randomImageSource.GetOutput()
    

    image.SetRegions(region)
    image.Allocate() 
    
    
    return image



################################################################################
###                                 TESTING                                  ###
################################################################################

def test_pytest_properly_works():
    '''
    This function just checks the proper work of pytest
    '''
    assert 1 == 1
    
    

@given(image = elastix_IO_strategy())
@settings(deadline = None)
def test_pytest_itk(image):
    '''
    This function just checks if the test strategy is right
    '''
    itk.imwrite(image, "./output_image.nii")
    image1 = itk.imread("./output_image.nii", itk.F)
    image2 = itk.imread("./output_image.nii", itk.F)
    os.remove("./output_image.nii")
    
    
    #Test if the images are the same:
    
    #The Same Size
    assert np.all(image1.GetLargestPossibleRegion().GetSize() == image2.GetLargestPossibleRegion().GetSize())
    #Same Spacing
    assert np.all(image1.GetSpacing() == image2.GetSpacing())
    #Same Image
    assert np.all( np.isclose( itk.GetArrayFromImage(image1), itk.GetArrayFromImage(image2)) )
    
    
    
    
@given(image = elastix_IO_strategy())
@settings(deadline = None)
def test_elastix_registration_reader(image):
    '''
    This function tests if the reader works properly
    '''
    itk.imwrite(image, "./output_image.nii")
    read_im1, read_im2 = registration_reader ('./output_image.nii', './output_image.nii')
    normal_read_image = itk.imread("./output_image.nii", itk.F)
    os.remove("./output_image.nii")

    
    #Test for the 1st image read:
    #Size
    assert np.all(read_im1.GetLargestPossibleRegion().GetSize() == normal_read_image.GetLargestPossibleRegion().GetSize())
    #Same Spacing
    assert np.all(read_im1.GetSpacing() == normal_read_image.GetSpacing())
    #Same Image
    assert np.all( np.isclose( itk.GetArrayFromImage(read_im1), itk.GetArrayFromImage(normal_read_image)) )
    
    #Test for the 2nd image read:
    #Size
    assert np.all(read_im2.GetLargestPossibleRegion().GetSize() == normal_read_image.GetLargestPossibleRegion().GetSize())
    #Same Spacing
    assert np.all(read_im2.GetSpacing() == normal_read_image.GetSpacing())
    #Same Image
    assert np.all( np.isclose( itk.GetArrayFromImage(read_im2), itk.GetArrayFromImage(normal_read_image)) )
    
    
    
    
@given(image= elastix_IO_strategy())
@settings(deadline = None)
def test_elastix_registration_writer(image):
    '''
    This function tests if the writer works properly
    '''
    registration_writer(image)
    itk.imwrite(image, "itk_written_image.nii")
    read_image = itk.imread("./output_image.nii")
    itk_written_image = itk.imread("./itk_written_image.nii")
    os.remove("./output_image.nii")
    os.remove("./itk_written_image.nii")


    
    #Tests
    #Size
    assert np.all(read_image.GetLargestPossibleRegion().GetSize() == itk_written_image.GetLargestPossibleRegion().GetSize())
    #Same Spacing
    assert np.all(read_image.GetSpacing() == itk_written_image.GetSpacing())
    #Same Image
    assert np.all( np.isclose( itk.GetArrayFromImage(read_image), itk.GetArrayFromImage(itk_written_image)) )
    
    


@given(fixed_image = elastix_IO_strategy(), moving_image = random_square_strategy())
@settings(deadline = None)
def test_elastix_registration_dimension(fixed_image, moving_image):
    
    '''
    This function tests if the final registered image has the same size of the initial fixed image
    '''
    itk.imwrite(fixed_image, "./f_image.nii")
    itk.imwrite(moving_image, "./m_image.nii")
    f_image, m_image = registration_reader('./f_image.nii','./m_image.nii')
    os.remove("./f_image.nii")
    os.remove("./m_image.nii")
    
    ImageType = itk.Image[itk.UC, 3]
    final_image = ImageType.New()
    
    final_image, registration_parameters = elastix_registration(f_image, m_image, True)
    LogToConsoleOn()
    assert np.all(final_image.GetLargestPossibleRegion().GetSize() == f_image.GetLargestPossibleRegion().GetSize())
    
