#!/usr/bin/env python



import itk
import numpy as np
import os
from datetime import datetime
import glob


###############################################################################

#IMAGE READER

def registration_reader (fixed_image_filename, moving_image_filename):
    """This function load 2 images from their path and return 2 ITK objects.
        
        Parameters
        ----------
        
            fixed_image_filename : string
                        Filename of the fixed image
                                
            moving_image_filename : string
                        Filename of the moving image
            
        Return
        ------
        
            fixed_image : itk.F object
                        The fixed image
                        
            moving_image : itk.F object
                        The moving image
    """
    
    
    #We load the images as float (itk.F) because is necessary for elastix
    fixed_image = itk.imread(fixed_image_filename, itk.F)
    moving_image = itk.imread(moving_image_filename, itk.F)
    
    return fixed_image, moving_image;
    
    
    
#############################
# Rigid REGISTRATION FUNCTION

def elastix_rigid_registration(fixed_image, moving_image, clog_value = False):
    """This function do the registration of a moving image over a fixed image using a RIGID parameter map.
    
    Parameters
    ----------
    
        fixed_image : itk.F object 
                    Image over the registration have to be done
                    
        moving_image : itk.F object
                    Image that has to be registered
                    
        clog_value : boolean object.
                    Default is False. If true it can be seen the Log_To_Console of the Elastix Registration.
                    
    Returns
    -------
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
  #  parameter_map_rigid['ShowExactMetricValue'] = ['false', 'false', 'false', 'true']
    
    parameter_object.AddParameterMap(parameter_map_rigid)
    
    
    #Now we start the registration
    elastix_object = itk.ElastixRegistrationMethod.New(fixed_image, moving_image)
    elastix_object.SetParameterObject(parameter_object)
    elastix_object.SetLogToConsole(clog_value)
    
    
    elastix_object.UpdateLargestPossibleRegion()
    
    print("The Rigid Registration is done!")
    
    return elastix_object

################################
# Multimap REGISTRATION FUNCTION

def elastix_multimap_registration(fixed_image, moving_image, clog_value = False):
    """This function do the registration of a moving image over a fixed image using 3 sets of parameters map: Rigid, Affine, BSpline.
    
    Parameters
    ----------
    
        fixed_image : itk.F object 
                    Image over the registration have to be done
                    
        moving_image : itk.F object
                    Image that has to be registered
                    
        clog_value : boolean object.
                    Default is False. If true it can be seen the Log_To_Console of the Elastix Registration.
        
    Returns
    -------
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
    # parameter_map_bspline['WriteIterationInfo'] = ['true']

    
    
    parameter_object.AddParameterMap(parameter_map_bspline)
    
    
    #Now we start the registration
    elastix_object = itk.ElastixRegistrationMethod.New(fixed_image, moving_image)
    elastix_object.SetParameterObject(parameter_object)
    elastix_object.SetLogToConsole(clog_value)
    
    
    elastix_object.UpdateLargestPossibleRegion()
    
    print("The Multimap Registration is done!")
    
    return elastix_object


#####
#IMAGE WRITER

def registration_writer(elastix_object, path = './', image_name = 'registered_image'):
    
    """This creates a directory and save in it an itk image as a nifti image and the txt file(s) with the final_transformation_parameters.
        
        Parameters
        ----------
        
        elstix_object : elastix object
                The final elastix object with the registration image and the final transformation parameters in it.
            
        file_path : string
                Path where will be the directory in which the image will be saved. Default is "./"
                
                
        image_name : string
                The name of the registered file without the extension. Default is "registered_image"
         
         
         
        Returns
        -------
        
            dir_path : string
                Path of the created directory
    """
    #find the actual date and time
    now = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
    #name of the directory
    dir_name = 'Registration_'+now
    
    dir_path = path + "/"  + dir_name
    
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
        
            
    for index in range(elastix_object.GetTransformParameterObject().GetNumberOfParameterMaps()):
        parameter_map = elastix_object.GetTransformParameterObject().GetParameterMap(index)
        elastix_object.GetTransformParameterObject().WriteParameterFile(parameter_map, dir_path + "/TransformParameters.{0}.txt".format(index))
    
    image = elastix_object.GetOutput()
        
    itk.imwrite(image, dir_path + "/"+ image_name +".nii")
    
    
    print("All your files are written")
   
    
    return dir_path




def registration_transform_parameters_writer(elastix_object, path ='./'):
    
    """This creates a directory and save in it the txt file(s) with the final_transformation_parameters.
        
        Parameters
        ----------
        
        elstix_object : elastix object
                The final elastix object with the registration image and the final transformation parameters in it.
            
        file_path : string
                Path where will be the directory in which the image will be saved. Default is "./"
         
         
         
        Returns
        -------
        
            dir_path : string
                Path of the created directory
    """
    
    #find the actual date and time
    now = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
    #name of the directory
    dir_name = 'Registration_'+now
    
    dir_path = path + "/"  + dir_name
    
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
        
    
    for index in range(elastix_object.GetTransformParameterObject().GetNumberOfParameterMaps()):
        parameter_map = elastix_object.GetTransformParameterObject().GetParameterMap(index)
        elastix_object.GetTransformParameterObject().WriteParameterFile(parameter_map, dir_path + "/TransformParameters.{0}.txt".format(index))
        
        print("Your Transform Files are written")

    
    return dir_path



def transform_parameters_writer(params_obj, path ='./' ):
    
    """This creates a directory and save in it the txt file(s) of the parameter object.
        
        Parameters
        ----------
        
        params_obj : elastix (or transformix) parameter object
                The parameter object you want to write
            
        file_path : string
                Path where will be the directory in which the image will be saved. Default is "./"
         
         
         
        Returns
        -------
        
        dir_path : string
            Path of the created directory
    """
    
    #find the actual date and time
    now = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
    #name of the directory
    dir_name = 'Saved_transform_'+now
    
    dir_path = path + "/"  + dir_name
    
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
        
    
    for index in range(params_obj.GetNumberOfParameterMaps()):
        parameter_map = params_obj.GetParameterMap(index)
        params_obj.WriteParameterFile(parameter_map, dir_path + "/TransformParameters.{0}.txt".format(index))
        
    print("Your Transform Files are written")

    
    return dir_path
    

    
#####
#APPLY THE SAME TRANSFORM ON OTHER IMAGES
def apply_transform_from_files(image, transform_path):
    '''
    This function  apply the transformation from files in a directory to other images
    
    Parameters
    ----------
    image : itk object
            The image you want to transform
            
    transform_path : string
            The path to the directory with the Transformation files you want to apply
            
    Returns
    -------
    
    result_image : itk object
            The transformed image
    '''
    
    
    num_of_transf_file = len(glob.glob1(transform_path,"TransformParameters.*"))
    parameter_files = [transform_path + "/TransformParameters.{0}.txt".format(i) for i in range(num_of_transf_file)]
    transform = itk.ParameterObject.New()
    transform.ReadParameterFile(parameter_files)

    result_image = itk.transformix_filter(image, transform)
    
    print("The Transformation is done!")
    
    return result_image

#####
#READ A TRANSFORM FROM A FILE
def read_transform_from_files(transform_path):
    '''
    This function simply read a transformation from files in a directory
    
    Parameters
    ----------
            
    transform_path : string
        The path to the directory with the Transformation files you want to apply
            
    Returns
    -------
    
    transform : Elastix ParameterMap
        The transormation parameters
    '''
    
    
    num_of_transf_file = len(glob.glob1(transform_path,"TransformParameters.*"))
    parameter_files = [transform_path + "/TransformParameters.{0}.txt".format(i) for i in range(num_of_transf_file)]
    transform = itk.ParameterObject.New()
    transform.ReadParameterFile(parameter_files)
    
    return transform



#CHANGE THE SIZE AND THE SPACING OF A TRASFORM
def Set_sampler_parameters_as_image(params_file, image):
    """
    This function sets the Size and the Spacing saved in a parameters file as the ones of an other image.
    
    Parameters
    ----------
    
    params_file : elastix parameter object
        The parameters file you want to change
        
    image : itk image object
        The image you want to get the Size and the Spacing from
        
    
    
    Returns
    -------
    
    params_file : elastix parameter object
        The parameters file changed
    
    """
    
    
    size = np.array(image.GetLargestPossibleRegion().GetSize())
    spacing = np.array(image.GetSpacing())
    
    for index in range(params_file.GetNumberOfParameterMaps()):
        params_file.SetParameter(index, "Size", size.astype(str))
        params_file.SetParameter(index, "Spacing", spacing.astype(str))
    
    
    return params_file


#CHANGE AN ATTRIBUTE VALUE OF A TRASFORM
def Set_parameters_map_attribute(params_file, attribute, value):
    """
    This function sets the Size and the Spacing saved in a parameters file as the ones of an other image.
    
    Parameters
    ----------
    
    params_file : elastix parameter object
        The parameters file you want to change
        
    attribute : string object
        The attribute you want to change
        
    value : string object
        the final value you want for the attribute
        
    
    
    Returns
    -------
    
    params_file : elastix parameter object
        The parameters file changed
    
    """
    
    
   
    
    for index in range(params_file.GetNumberOfParameterMaps()):
        params_file.SetParameter(index, attribute, value)
    
    
    return params_file


#EVALUATION
def evaluate_registration_mse(fixed_image, deformed_image, ax = None):
    """
    This function find the MSE between the 2 images. It's useful to evaluate the registration
    
    Parameters
    ----------
    
    fixed_image : itk image object
        The fixed image of your registration
        
    
    deformed_image : itk image object
        The result of your registration
        
        
    ax : boolean or int
        The axis you want to compute the mean of the squares on: 0 on columns, 1 on rows.
        Default is None: the mean of the flattered array
        
        
    Returns
    -------
    
    mse : float object
        The calcuated mse
    """
    
    fix_im_array = itk.GetArrayFromImage(fixed_image)
    def_im_array = itk.GetArrayFromImage(deformed_image)
    
    
    mse = np.mean( np.square( fix_im_array - def_im_array ), axis = ax)
    
    
    return mse

