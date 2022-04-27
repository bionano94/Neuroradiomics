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
from Neuroradiomics.registration import registration_transform_parameters_writer
from Neuroradiomics.registration import evaluate_registration_mse




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
    This function generates an itk image with a random 2D square.
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
    This function generates an itk image with a random 2D rectangle.
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
    This function generates an itk image with a random 3D cube.
    '''
    
    x_max = draw(st.integers(100,200))
    y_max = draw(st.integers(100,200))
    z_max = draw(st.integers(100,200))
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
    
    x_max = draw(st.integers(100,200))
    y_max = draw(st.integers(100,200))
    z_max = draw(st.integers(100,200))
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
    This function generates an itk image with a random 2D square.
    '''
    
    #I create a square image
    image = np.zeros([200, 200], np.float32)
    
    side = 50
    x1 = draw(st.integers(1,140))
    y1 = draw (st.integers(1,140))
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
    This function just checks the proper work of pytest
    '''
    assert 1 == 1
    
    

#Are tests working with ITK?

@given(image = random_image_strategy())
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
    
    
    

    
#######################################
#####     I/O Functions Tests     #####
#######################################



#Registration Reader Test

@given(image = random_image_strategy())
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
    
    

    
    
########## Transformation Parameters Reader Test

# 2D Transformation
    
@given(fixed_image = square_image_strategy(), moving_image = rectangular_image_strategy())
@settings(max_examples=10, deadline = None)
def test_read_transformation(fixed_image, moving_image):
    
    '''
    This function tests if the transformation reader operates properly
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
    
    
    assert np.all(direct_transf.GetLargestPossibleRegion().GetSize() == my_transf.GetLargestPossibleRegion().GetSize())
    #Same Spacing
    assert np.all(direct_transf.GetSpacing() == my_transf.GetSpacing())
    #Same Image
    assert np.all(np.isclose( itk.GetArrayFromImage(direct_transf), itk.GetArrayFromImage(my_transf), 1e-4, 1e-2 ) )
    
    
    
    

####### Registration Writer Test

@given(fixed_image = square_image_strategy(), moving_image = rectangular_image_strategy())
@settings(max_examples=20, deadline = None)
def test_elastix_registration_writer(fixed_image, moving_image):
    '''
    This function tests if the writer works properly
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
    
    #write the same result file with already tested ITK function
    itk.imwrite(elastix_object.GetOutput(), "itk_written_image.nii")
    
    #read our 2 images
    read_image = itk.imread("./registered_image.nii")
    itk_written_image = itk.imread("./itk_written_image.nii")
    
    #read our transformation files
    
    num_of_transf = len(os.listdir('./')) - 2 #-2 because we also have 2 images in the directory
    
    for i in range(0, num_of_transf):
        with open('TransformParameters.' + str(i) + '.txt') as f:
            lines = f.readlines()
    f.close()
    
    #delete everything was created
    os.chdir('..')
    for f in os.listdir(dir_path):
        os.remove(os.path.join(dir_path, f))
        
    os.rmdir(dir_path)


    
    #Tests
    #Size
    assert np.all(read_image.GetLargestPossibleRegion().GetSize() == itk_written_image.GetLargestPossibleRegion().GetSize())
    #Same Spacing
    assert np.all(read_image.GetSpacing() == itk_written_image.GetSpacing())
    #Same Image
    assert np.all( np.isclose( itk.GetArrayFromImage(read_image), itk.GetArrayFromImage(itk_written_image)) )
    #Is the transformation saved?
    assert lines
    
    

    

##### Registration parameters writer Test


@given(fixed_image = square_image_strategy(), moving_image = rectangular_image_strategy())
@settings(max_examples=20, deadline = None)
def test_elastix_transform_registration_writer(fixed_image, moving_image):
    '''
    This function tests if the writer works properly
    '''
    
    #create the elastix object for the registration
    parameter_object = itk.ParameterObject.New()
    default_rigid_parameter_map = parameter_object.GetDefaultParameterMap('rigid')
    default_affine_parameter_map = parameter_object.GetDefaultParameterMap('affine')
    
    parameter_object.AddParameterMap(default_rigid_parameter_map)
    parameter_object.AddParameterMap(default_affine_parameter_map)

    #set and run the registration
    elastix_object = itk.ElastixRegistrationMethod.New(fixed_image, moving_image)
    elastix_object.SetParameterObject(parameter_object)
    elastix_object.UpdateLargestPossibleRegion()

    
    
    #go with the function we want to test
    dir_path = registration_transform_parameters_writer(elastix_object)
    
    
    #read our transformation files
    
    num_of_transf = len(glob.glob1(dir_path,"TransformParameters.*"))
    
    parameter_files = [dir_path + "/TransformParameters.{0}.txt".format(i) for i in range(0, num_of_transf)]
    transform = itk.ParameterObject.New()
    transform.ReadParameterFile(parameter_files)

    
    transformed_mov = itk.transformix_filter(moving_image, transform)
    
    #go inside the directory
    os.chdir(dir_path)
    
    itk.imwrite(transformed_mov, './transformed_mov.nii')
    itk.imwrite(elastix_object.GetOutput(), './registered.nii')
    
    new_mov = itk.imread('./transformed_mov.nii', itk.F)
    reg_im = itk.imread('./registered.nii', itk.F)
    
    #delete everything was created
    os.chdir('..')
    for f in os.listdir(dir_path):
        os.remove(os.path.join(dir_path, f))
        
    os.rmdir(dir_path)

    
    #Tests
    assert np.all( np.isclose( itk.GetArrayFromImage(new_mov), itk.GetArrayFromImage(reg_im),1e-04,1e-02 ))    
    
#############################################
#####     Operative Functions Tests     #####
#############################################

# 2D Rigid Registration

@given(fixed_image = rigid_square_image_strategy(), moving_image = rigid_square_image_strategy())
@settings(max_examples=30, deadline = None)
def test_2D_elastix_rigid_registration(fixed_image, moving_image):
    
    '''
    This function tests if the final registered image has the same size and the same spaging of the initial fixed image.
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
@settings(max_examples=20, deadline = None)
def test_2D_elastix_multimap_registration(fixed_image, moving_image):
    
    '''
    This function tests if the final registered image has the same size and the same spaging of the initial fixed image.
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
@settings(max_examples=20, deadline = None, suppress_health_check = (HC.too_slow,))
def test_3D_elastix_registration(fixed_image, moving_image):
    
    '''
    This function tests if the final registered image has the same size and the same spaging of the initial fixed image.
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
@settings(max_examples=20, deadline = None)
def test_2D_elastix_transform(fixed_image, moving_image):
    
    '''
    This function tests if the transformation function operate properly
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
    #Same Image
    assert np.all(np.isclose( itk.GetArrayFromImage(transformed_image), itk.GetArrayFromImage(registered_image), 1e-04, 1e-02 ) )
    
    
    
    
# 3D Transformation
    
@given(fixed_image = cubic_image_strategy(), moving_image = poligon_image_strategy())
@settings(max_examples=20, deadline = None)
def test_3D_elastix_transform(fixed_image, moving_image):
    
    '''
    This function tests if the transformation function operate properly
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
    #Same Image
    assert np.all(np.isclose( itk.GetArrayFromImage(transformed_image), itk.GetArrayFromImage(registered_image), 1e-04, 1e-02) )
    
    

    
@given(fixed_image = cubic_image_strategy(), moving_image = poligon_image_strategy())
@settings(max_examples=20, deadline = None, suppress_health_check = (HC.too_slow,))
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
@settings(max_examples=20, deadline = None, suppress_health_check = (HC.too_slow,))
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
    
