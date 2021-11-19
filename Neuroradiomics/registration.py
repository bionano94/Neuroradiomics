#!/usr/bin/env python



import itk
import os
from datetime import datetime
import glob


###############################################################################

#IMAGE READER

def registration_reader (fixed_image_filename, moving_image_filename):
    """This function load 2 images from their path and return 2 ITK objects.
        
        Args:
        
            fixed_image_filename : string
                        Filename of the fixed image
                                
            moving_image_filename : string
                        Filename of the moving image
            
        Returns:
        
            fixed_image : itk.F object
                        The fixed image
                        
            moving_image : itk.F object
                        The moving image
    """
    
    
    #We load the images as float (itk.F) because is necessary for elastix
    fixed_image = itk.imread(fixed_image_filename, itk.F)
    moving_image = itk.imread(moving_image_filename, itk.F)
    
    return fixed_image, moving_image;
    
#####
#REGISTRATION FUNCTION

def elastix_registration(fixed_image, moving_image, clog_value = False):
    """This function do the registration of a moving image over a fixed image.
    
    Args:
    
        fixed_image : itk.F object 
                    Image over the registration have to be done
                    
        moving_image : itk.F object
                    Image that has to be registered
                    
        clog_value : boolean object.
                    Default is False. If true it can be seen the Log_To_Console of the Elastix Registration.
        
    Returns:
        elastix_object : elastix object
                    Resulting elastix object after the registration   
                   
    
    """
    
    #This registration will be done using 3 different type of transformation in subsequent order
    
    # Setting our number of resolutions
    parameter_object = itk.ParameterObject.New()
    resolutions = 4
    
    #Import RIGID parameter map
    parameter_map_rigid = parameter_object.GetDefaultParameterMap('rigid', resolutions)
    parameter_map_rigid['Metric']       = ['AdvancedMattesMutualInformation']
    parameter_map_rigid['Interpolator'] = ['BSplineInterpolatorFloat']
    
    parameter_object.AddParameterMap(parameter_map_rigid)
    
    
    #Adding an AFFINE parameter map
    parameter_map_affine = parameter_object.GetDefaultParameterMap("affine", resolutions)
    parameter_map_affine['Metric']       = ['AdvancedMattesMutualInformation']
    parameter_map_affine['Interpolator'] = ['BSplineInterpolatorFloat']

    parameter_object.AddParameterMap(parameter_map_affine)
    
    
    #Adding a NON-RIGID parameter map
    parameter_map_bspline = parameter_object.GetDefaultParameterMap("bspline", resolutions)
    parameter_map_bspline['Interpolator'] = ['BSplineInterpolatorFloat']
    parameter_object.AddParameterMap(parameter_map_bspline)
    
    
    #Now we start the registration
    elastix_object = itk.ElastixRegistrationMethod.New(fixed_image, moving_image)
    elastix_object.SetParameterObject(parameter_object)
    elastix_object.SetLogToConsole(clog_value)
    elastix_object.UpdateLargestPossibleRegion()

    
#    registered_image, result_transform_parameters = itk.elastix_registration_method(
#    fixed_image, moving_image,
#    parameter_object = parameter_object,
#    log_to_console = clog_value)
    
    print("The Registration is done!")
    
    return elastix_object


#####
#IMAGE WRITER

def registration_writer(elastix_object, path = './'):
    
    """This creates a directory and save in it an itk image as a nifti image as "registered_image.nii" and 3 txt files with the final_transformation_parameters.
        
        Args:
        
            elstix_object : elastix object
                The final elastix object with the registration image and the final transformation parameters in it.
                
            file_path : string
                Path where will be the directory in which the image will be saved
         
         
         
        Returns:
        
            dir_path : string
                Path of the created directory
    """
    #find the actual date and time
    now = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
    #name of the directory
    dir_name = 'Registration_'+now
    
    dir_path = path + dir_name
    
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
        
    elastix_object.SetOutputDirectory(dir_path)
    elastix_object.UpdateLargestPossibleRegion()
    
    image = elastix_object.GetOutput()
        
    itk.imwrite(image, dir_path + "/registered_image.nii")
    
    print("Your files are written")
   
    
    return dir_path


    
#####
#APPLY THE SAME TRANSFORM ON OTHER IMAGES
def apply_transform_from_files(image, transform_path):
    '''
    This function  apply the transformation from files in a directory to other images
    
    Args:
    
        image : itk object
            The image you want to transform
            
        transform_path : string
            The path to the directory with the Transformation files you want to apply
            
    Returns:
    
        result_image : itk object
            The transformed image
    '''
    
    
    num_of_transf_file = len(glob.glob1(transform_path,"TransformParameters.*"))
    parameter_files = [transform_path + "/TransformParameters.{0}.txt".format(i) for i in range(num_of_transf_file)]
    transform = itk.ParameterObject.New()
    transform.ReadParameterFile(parameter_files)

    result_image = itk.transformix_filter(image, transform)
    
    return result_image

#####
#APPLY THE SAME TRANSFORM ON OTHER IMAGES
def read_transform_from_files(transform_path):
    '''
    This function simply read a transformation from files in a directory
    
    Args:
            
        transform_path : string
            The path to the directory with the Transformation files you want to apply
            
    Returns:
    
        transform : Elastix ParameterMap
            The transormation parameters
    '''
    
    
    num_of_transf_file = len(glob.glob1(transform_path,"TransformParameters.*"))
    parameter_files = [transform_path + "/TransformParameters.{0}.txt".format(i) for i in range(num_of_transf_file)]
    transform = itk.ParameterObject.New()
    transform.ReadParameterFile(parameter_files)
    
    return transform