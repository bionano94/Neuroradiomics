#!/usr/bin/env python



import itk
import os
from datetime import datetime


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
        result_image : elastix object
                    Resulting registered image   
                   
        result_transform_parameters : elastix object
                    Resulting transform parameters 
    
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
    registered_image, result_transform_parameters = itk.elastix_registration_method(
    fixed_image, moving_image,
    parameter_object = parameter_object,
    log_to_console = clog_value)
    
    print("The Registration is done!")
    
    return registered_image, result_transform_parameters


#####
#IMAGE WRITER

def registration_writer(image, path = './'):
    
    """This creates a directory and save in it an itk image as a nifti image as "registered_image.nii".
        
        Args:
        
            image : itk object
                The image that has to be written
                
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
    
    if not os.path.exists('./'+dir_name):
        os.mkdir('./'+dir_name)
    
    dir_path = path + dir_name
    itk.imwrite(image, dir_path + "/registered_image.nii")
   
    
    return dir_path


    
#####
#APPLY THE SAME TRANSFORM ON OTHER IMAGES
def registration_transform(image, transform):
    '''
    This function simply apply the transformation obtained to other images
    
    Args:
    
        image : itk object
            The image you want to transform
            
        transform_params : Elastix parameter map
            The transformation you want to apply on the image
            
    Returns:
    
        result_image : itk object
            The image transformed
    '''

    result_image = itk.transformix_filter(image, transform)
    
    return result_image