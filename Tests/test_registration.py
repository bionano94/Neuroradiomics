#!/usr/bin/env python


import pytest
import hypothesis.strategies as st
from hypothesis import given, settings, assume
from hypothesis import HealthCheck as HC


from Neuroradiomics.registration import registration_reader
from Neuroradiomics.registration import elastix_multimap_registration
from Neuroradiomics.registration import registration_writer
from Neuroradiomics.registration import apply_transform_from_files
from Neuroradiomics.registration import read_transform_from_files
from Neuroradiomics.registration import elastix_rigid_registration
from Neuroradiomics.registration import Set_sampler_parameters_as_image
from Neuroradiomics.registration import transform_parameters_writer
from Neuroradiomics.evaluation_utilities import evaluate_registration_mse




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
def square_image_strategy(draw):
    '''
    This function generates a black itk image of random size with a random 2D white square.
    '''
   
    x_max = draw(st.integers(100,200))
    y_max = draw(st.integers(100,200))
    image = np.zeros([x_max, y_max], np.float32)

    
    side = draw(st.integers(40,60))
    x1 = draw(st.integers(10,40))
    y1 = draw (st.integers(10,40))
    x2 = x1 + side
    y2 = y1 + side
    
    for x in range (x1, x2):
        for y in range (y1, y2):
            image[x,y]=100
    image = itk.image_view_from_array(image)
    return image


@st.composite
def rectangular_image_strategy(draw):
    '''
    This function generates a black itk image of random size with a random 2D white rectangle.
    '''
    
    x_max = draw(st.integers(100,200))
    y_max = draw(st.integers(100,200))
    image = np.zeros([x_max, y_max], np.float32)
    
    x1 = draw (st.integers(10,40))
    y1 = draw (st.integers(10,40))
    side_x = draw (st.integers(40,60))
    side_y = draw (st.integers(40,60))
    
    for x in range (x1, x1+side_x):
        for y in range (y1, y1+side_y):
            image[x,y]=100
    image = itk.image_view_from_array(image)
    return image

@st.composite
def cubic_image_strategy(draw):
    '''
    This function generates a black itk image of random size with a random 3D white cube.
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
def poligon_image_strategy(draw):
    '''
    This function generates an itk image with a random 3D poligon.
    '''
    
    x_max = draw(st.integers(150,200))
    y_max = draw(st.integers(150,200))
    z_max = draw(st.integers(150,200))
    image = np.zeros([x_max, y_max, z_max], np.float32)
    
    x1 = draw (st.integers(10,40))
    y1 = draw (st.integers(10,40))
    z1 = draw (st.integers(10,40))
    side_x = draw (st.integers(40,60))
    side_y = draw (st.integers(40,60))
    side_z = draw (st.integers(40,60))
    
    
    for x in range (x1, x1+side_x):
        for y in range (y1, y1+side_y):
            for z in range (z1, z1+side_z):
                image[x,y]=100
                
    image = itk.image_view_from_array(image)
    return image





@st.composite
def rigid_square_image_strategy(draw):
    '''
    This function generates an itk image with a random 2D square of fixed size.
    '''
    
    #I create a square image
    image = np.zeros([200, 200], np.float32)
    
    side = 50
    x1 = draw(st.integers(50,100))
    y1 = draw (st.integers(50,100))
    x2 = x1 + side
    y2 = y1 + side
    
    for x in range (x1, x2):
        for y in range (y1, y2):
            image[x,y]=100
    image = itk.image_view_from_array(image)
    
    
    return image
                            
                
                
    


# ████████ ███████ ███████ ████████ ███████ 
#    ██    ██      ██         ██    ██      
#    ██    █████   ███████    ██    ███████
#    ██    ██           ██    ██         ██
#    ██    ███████ ███████    ██    ███████



#######################################
#####     AM I TESTING RIGHT?     #####
#######################################





# Is Pytest working?

def test_pytest_properly_works():
    '''
    This function just checks the proper work of pytest. 
    '''
    assert 1 == 1
    
    

#Are tests working with ITK?

@given(image = random_image_strategy())
@settings(deadline = None)
def test_pytest_itk(image):
    '''
    This function just checks if the test strategy is right by writing a random image and then reading it two times and comparing if the two images are the same. To check the equality of the images we chek that they have the same size, the same spacing an the same pixel values.
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
    
    
    

    
#######################################
#####     I/O Functions Tests     #####
#######################################



#Registration Reader Test

@given(image = random_image_strategy())
@settings(deadline = None)
def test_elastix_registration_reader(image):
    '''
    This function tests if the implemented reader function works properly writing a random image and then reading it two times and checking if all the two objects read with the function are equal as the written image. To check the equality of the images we chek that they have the same size, the same spacing an the same pixel values.
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
    
    

    
    
########## Transformation Parameters Reader Test

# 2D Transformation
    
@given(fixed_image = square_image_strategy(), moving_image = rectangular_image_strategy())
@settings(max_examples=5, deadline = None)
def test_read_transformation(fixed_image, moving_image):
    
    '''
    This function register two images, write the transformation parameters and then read them again and check if the read transform has the same spacing.
    The read transform is then applied to the image originally registered.
    It is then checked if the registered image has the same spacing and the same size of the transformed image and it is checked if the read transformation has the same number of parameters as the initial one.
    '''
    
    #run a registration
    elastix_object = elastix_multimap_registration(fixed_image, moving_image)
    
    registered_image = elastix_object.GetOutput()
    final_transf_params = elastix_object.GetTransformParameterObject()
    
    #write the registration and its parameters
    dir_path = registration_writer(elastix_object)
    
    
    #read the Parameters
    transform_params = read_transform_from_files(dir_path)
    
    #apply the transformation over the same image with the 2 parameter objects
    direct_transf = itk.transformix_filter(moving_image, final_transf_params)
    my_transf = itk.transformix_filter(moving_image, transform_params)
    
    #delete everything was created
    for f in os.listdir(dir_path):
        os.remove(os.path.join(dir_path, f))
        
    os.rmdir(dir_path)
    
    #Size
    assert np.all(direct_transf.GetLargestPossibleRegion().GetSize() == my_transf.GetLargestPossibleRegion().GetSize())
    #Same Spacing
    assert np.all(direct_transf.GetSpacing() == my_transf.GetSpacing())        
    
    
@given(fixed_image = square_image_strategy(), moving_image = rectangular_image_strategy())
@settings(max_examples=5, deadline = None)
def test_read_transformation_parameters_map(fixed_image, moving_image):
    
    '''
    This function register two images, write the transformation parameters and then read them again.
    The is checked if the two object have the same number of parameter maps, and for each parameter map if they have:
    same transform
    same resampler
    same resamplerinterpolator
    '''
    
    #run a registration
    elastix_object = elastix_multimap_registration(fixed_image, moving_image)
    
    registered_image = elastix_object.GetOutput()
    final_transf_params = elastix_object.GetTransformParameterObject()
    
    #write the registration and its parameters
    dir_path = registration_writer(elastix_object)
    
    
    #read the Parameters
    transform_params = read_transform_from_files(dir_path)
    
    
    #delete everything was created
    for f in os.listdir(dir_path):
        os.remove(os.path.join(dir_path, f))
        
    os.rmdir(dir_path)
 
    assert transform_params.GetNumberOfParameterMaps() == final_transf_params.GetNumberOfParameterMaps()
    
    for i in range(0, transform_params.GetNumberOfParameterMaps()):
        assert transform_params.GetParameter(i, 'Transform') == final_transf_params.GetParameter(i, 'Transform')
        assert transform_params.GetParameter(i, 'Resampler') == final_transf_params.GetParameter(i, 'Resampler')
        assert transform_params.GetParameter(i, 'ResampleInterpolator') == final_transf_params.GetParameter(i, 'ResampleInterpolator')

    

####### Registration Writer Test

@given(fixed_image = rigid_square_image_strategy(), moving_image = rigid_square_image_strategy())
@settings(max_examples=10, deadline = None)
def test_elastix_registration_writer_exists_parameters(fixed_image, moving_image):
    '''
    This function tests if the writer save some parameter files and check if the names are correct.
    This is done registering an image and then writing the parameter files down and then checking their existence.
    '''
    
    #create the elastix object for the registration
    parameter_object = itk.ParameterObject.New()
    default_affine_parameter_map = parameter_object.GetDefaultParameterMap('affine')
    parameter_object.AddParameterMap(default_affine_parameter_map)

    #set and run the registration
    elastix_object = itk.ElastixRegistrationMethod.New(fixed_image, moving_image)
    elastix_object.SetParameterObject(parameter_object)
    elastix_object.UpdateLargestPossibleRegion()

    #go with the function we want to test
    dir_path = registration_writer(elastix_object)
    
    #go inside our new directory
    os.chdir(dir_path)
    
    
    #read our transformation files
    
    num_of_transf = len(os.listdir('./')) -1 #-1 because we also have 1 images in the directory
    
    for i in range(0, num_of_transf):
        with open('TransformParameters.' + str(i) + '.txt') as f:
            lines = f.readlines()
    f.close()
    
    
    for j in range(0, num_of_transf):
        path = 'TransformParameters.' + str(j) + '.txt'
        assert os.path.isfile(path)
    
    #delete everything was created
    os.chdir('..')
    for f in os.listdir(dir_path):
        os.remove(os.path.join(dir_path, f))
        
    os.rmdir(dir_path)

    assert lines
    
    

@given(fixed_image = rigid_square_image_strategy(), moving_image = rigid_square_image_strategy())
@settings(max_examples=10, deadline = None)
def test_elastix_registration_writer_exists_image(fixed_image, moving_image):
    '''
    This function tests if the writer save a nii image.
    This is done registering an image and then writing the image file and then checking its existence.
    '''
    
    #create the elastix object for the registration
    parameter_object = itk.ParameterObject.New()
    default_affine_parameter_map = parameter_object.GetDefaultParameterMap('affine')
    parameter_object.AddParameterMap(default_affine_parameter_map)

    #set and run the registration
    elastix_object = itk.ElastixRegistrationMethod.New(fixed_image, moving_image)
    elastix_object.SetParameterObject(parameter_object)
    elastix_object.UpdateLargestPossibleRegion()

    #go with the function we want to test
    dir_path = registration_writer(elastix_object)
    
    #go inside our new directory
    os.chdir(dir_path)
    
    
    check_nii_existence = False
    for fname in os.listdir('.'): 
        check_nii_existence = check_nii_existence or fname.endswith('.nii')
        
    
    #delete everything was created
    os.chdir('..')
    for f in os.listdir(dir_path):
        os.remove(os.path.join(dir_path, f))
        
    os.rmdir(dir_path)
    
    assert check_nii_existence

    
@given(fixed_image = rigid_square_image_strategy(), moving_image = rigid_square_image_strategy())
@settings(max_examples=10, deadline = None)
def test_elastix_registration_writer_proper_transformation(fixed_image, moving_image):
    '''
    This function tests if the writer save properly the parameter object. 
    It writes the transformation in a file then read it again and check if are the same of the registration.
    '''
    
     #create the elastix object for the registration
    parameter_object = itk.ParameterObject.New()
    default_affine_parameter_map = parameter_object.GetDefaultParameterMap('affine')
    parameter_object.AddParameterMap(default_affine_parameter_map)

    #set and run the registration
    elastix_object = itk.ElastixRegistrationMethod.New(fixed_image, moving_image)
    elastix_object.SetParameterObject(parameter_object)
    elastix_object.UpdateLargestPossibleRegion()

    #go with the function we want to test
    dir_path = registration_writer(elastix_object)
    
    #read the parameters written
    
    read_param_object = read_transform_from_files(dir_path)
    
    
    #delete everything was created
    for f in os.listdir(dir_path):
        os.remove(os.path.join(dir_path, f))
        
    os.rmdir(dir_path)

    for i in range(0, read_param_object.GetNumberOfParameterMaps()):
        assert read_param_object.GetParameter(i, 'Transform') == parameter_object.GetParameter(i, 'Transform')
        assert read_param_object.GetParameter(i, 'Resampler') == parameter_object.GetParameter(i, 'Resampler')
        assert read_param_object.GetParameter(i, 'ResampleInterpolator') == parameter_object.GetParameter(i, 'ResampleInterpolator')    


        
    
#############################################
#####     Operative Functions Tests     #####
#############################################

# 2D Rigid Registration

@given(fixed_image = rigid_square_image_strategy(), moving_image = rigid_square_image_strategy())
@settings(max_examples=20, deadline = None, suppress_health_check = (HC.too_slow,))
def test_2D_elastix_rigid_registration(fixed_image, moving_image):
    
    '''
    This function tests if the final registered image has the same size and the same spacing of the initial fixed image.
    This is for 2D images.
    '''
    
    elastix_object = elastix_rigid_registration(fixed_image, moving_image, True)
    image = elastix_object.GetOutput()
    
    itk.imwrite(image, "./final_image.nii")
    reg_image = itk.imread('./final_image.nii')
    
    os.remove('./final_image.nii')              
                
    
    assert np.all(reg_image.GetLargestPossibleRegion().GetSize() == fixed_image.GetLargestPossibleRegion().GetSize())
    assert np.all(reg_image.GetSpacing() == fixed_image.GetSpacing())
    
    
    
# 2D Multimap Registration

@given(fixed_image = square_image_strategy(), moving_image = rectangular_image_strategy())
@settings(max_examples=10, deadline = None, suppress_health_check = (HC.too_slow,))
def test_2D_elastix_multimap_registration(fixed_image, moving_image):
    
    '''
    This function tests if the final registered image has the same size and the same spacing of the initial fixed image.
    This is for 2D images.
    '''
    
    elastix_object = elastix_multimap_registration(fixed_image, moving_image)
    image = elastix_object.GetOutput()
    
    itk.imwrite(image, "./final_image.nii")
    reg_image = itk.imread('./final_image.nii')
    
    os.remove('./final_image.nii')              
                
    
    assert np.all(reg_image.GetLargestPossibleRegion().GetSize() == fixed_image.GetLargestPossibleRegion().GetSize())
    assert np.all(reg_image.GetSpacing() == fixed_image.GetSpacing())
    

    
# 3D Multmap Registration

@given(fixed_image = cubic_image_strategy(), moving_image = poligon_image_strategy())
@settings(max_examples=10, deadline = None, suppress_health_check = (HC.too_slow,))
def test_3D_elastix_registration(fixed_image, moving_image):
    
    '''
    This function tests if the final registered image has the same size and the same spacing of the initial fixed image.
    This is for 3D images.
    '''
  
    elastix_object = elastix_multimap_registration(fixed_image, moving_image)
    image = elastix_object.GetOutput()
    
    itk.imwrite(image, "./final_image.nii")
    reg_image = itk.imread('./final_image.nii')
    
    os.remove('./final_image.nii')              
    
    assert np.all(reg_image.GetLargestPossibleRegion().GetSize() == fixed_image.GetLargestPossibleRegion().GetSize())
    assert np.all(reg_image.GetSpacing() == fixed_image.GetSpacing())
    

    
# 2D Transformation
    
@given(fixed_image = square_image_strategy(), moving_image = rectangular_image_strategy())
@settings(max_examples=10, deadline = None, suppress_health_check = (HC.too_slow,))
def test_2D_elastix_transform(fixed_image, moving_image):
    
    '''
    This function tests if the transformation function operate properly.
    It registers two images and then apply the obtained transformation parameters on the original moving image.
    The it checks if the image registered and the one transformed are the same image.
    This is for 2D images.
    '''
    
    #run a registration
    elastix_object = elastix_multimap_registration(fixed_image, moving_image)
    
    registered_image = elastix_object.GetOutput()
    
    #write the registration and it parameters
    dir_path = registration_writer(elastix_object)
    
    
    #apply the transformation with the parameters obtained on the moving_image
    transformed_image = apply_transform_from_files(moving_image, dir_path)
    
    #delete everything was created
    for f in os.listdir(dir_path):
        os.remove(os.path.join(dir_path, f))
        
    os.rmdir(dir_path)
    
    #Test if the 2 images obtained are the same one
    #The Same Size
    assert np.all(transformed_image.GetLargestPossibleRegion().GetSize() == registered_image.GetLargestPossibleRegion().GetSize())
    #Same Spacing
    assert np.all(transformed_image.GetSpacing() == registered_image.GetSpacing())
    #Same image
    assert np.all( np.isclose( itk.GetArrayFromImage(transformed_image), itk.GetArrayFromImage(registered_image), 0.01, 0.01) )

    
    
    
# 3D Transformation
    
@given(fixed_image = cubic_image_strategy(), moving_image = poligon_image_strategy())
@settings(max_examples=10, deadline = None, suppress_health_check = (HC.too_slow,))
def test_3D_elastix_transform(fixed_image, moving_image):
    
    '''
    This function tests if the transformation function operate properly.
    It registers two images and then apply the obtained transformation parameters on the original moving image.
    The it checks if the image registered and the one transformed are the same image.
    This is for 3D images.
    '''
    
    #run a registration
    elastix_object = elastix_multimap_registration(fixed_image, moving_image)
    
    registered_image = elastix_object.GetOutput()
    
    #write the registration and it parameters
    dir_path = registration_writer(elastix_object)
    
    
    #apply the transformation with the parameters obtained on the moving_image
    transformed_image = apply_transform_from_files(moving_image, dir_path)
    
    #delete everything was created
    for f in os.listdir(dir_path):
        os.remove(os.path.join(dir_path, f))
        
    os.rmdir(dir_path)
    
    #Test if the 2 images obtained are the same one
    #The Same Size
    assert np.all(transformed_image.GetLargestPossibleRegion().GetSize() == registered_image.GetLargestPossibleRegion().GetSize())
    #Same Spacing
    assert np.all(transformed_image.GetSpacing() == registered_image.GetSpacing())
    #Same image
    assert np.all( np.isclose( itk.GetArrayFromImage(transformed_image), itk.GetArrayFromImage(registered_image), 0.01, 0.01) )
    

    
@given(fixed_image = cubic_image_strategy(), moving_image = poligon_image_strategy())
@settings(max_examples=10, deadline = None, suppress_health_check = (HC.too_slow,))
def test_mse(fixed_image, moving_image):
    
    '''
    This function tests if the evaluation function works properly
    '''
    
    elastix_object = elastix_multimap_registration(fixed_image, moving_image)
    image = elastix_object.GetOutput()
    
    assert evaluate_registration_mse(fixed_image, fixed_image) == 0
    assert evaluate_registration_mse(fixed_image, image) > 0
    
    
    
    
# Set Parameters

@given(fixed_image = cubic_image_strategy(), moving_image = poligon_image_strategy())
@settings(max_examples=10, deadline = None, suppress_health_check = (HC.too_slow,))
def test_set_parameters(fixed_image, moving_image):
    
    '''
    This function tests if the Set Parameters function works properly
    '''
  
    elastix_object = elastix_multimap_registration(fixed_image, moving_image)
    params = elastix_object.GetTransformParameterObject()
    
    new_params = Set_sampler_parameters_as_image(params, moving_image)
    
    transformed_image = itk.transformix_filter(moving_image, new_params)
    
    
    assert np.all(transformed_image.GetLargestPossibleRegion().GetSize() == moving_image.GetLargestPossibleRegion().GetSize())
    assert np.all(transformed_image.GetSpacing() == moving_image.GetSpacing())
    
