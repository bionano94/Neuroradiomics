import itk
import numpy as np
from sys import argv


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
    


def elastix_registration(fixed_image, moving_image):
    """This function do the registration of a moving image over a fixed image.
    
    Args:
    
        fixed_image : itk.F object 
                    Image over the registration have to be done
                    
        moving_image : itk.F object
                    Image that has to be registered
        
    Returns:
        result_image : elastix object
                    Resulting registered image   
                   
        result_transform_parameters : elastix object
                    Resulting transform parameters 
    
    """
    
    #This registration will be done using 3 different typer of transformation in subsequent order
    
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
    parameter_map_affine['Metric']      = ['AdvancedMattesMutualInformation']
    parameter_map_affine['Interpolator'] = ['BSplineInterpolatorFloat']

    parameter_object.AddParameterMap(parameter_map_affine)
    
    
    #Adding a NON-RIGID parameter map
    parameter_map_bspline = parameter_object.GetDefaultParameterMap("bspline", resolutions)
    parameter_map_bspline['Metric']      = ['AdvancedMattesMutualInformation']
    parameter_map_bspline['Interpolator'] = ['BSplineInterpolatorFloat']
    parameter_object.AddParameterMap(parameter_map_bspline)
    
    
    #Now we start the registration
    registered_image, result_transform_parameters = itk.elastix_registration_method(
    fixed_image, moving_image,
    parameter_object = parameter_object,
    log_to_console = False)
    
    print("The Registration is done!")
    
    #Beacuse the registration is done on a downsampled image, we apply the final transformation on the original moving image
    result_moving_image = itk.transformix_filter(moving_image, result_transform_parameters)
    
    return result_moving_image, result_transform_parameters



def registration_writer(image, file_path):
    
    """This save an itk image as a nifti image as "output_image.nii".
        
        Args:
        
            image : itk object
                The image that has to be written
                
            file_path : string
                Path where the image will be saved
    """
    itk.imwrite(image, file_path+"/output_image.nii")
    print("Your file is written!")





