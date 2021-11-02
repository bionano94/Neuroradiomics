#!/usr/bin/env python
# coding: utf-8

# In[2]:


import itk
import numpy as np
from sys import argv


# In[3]:


def registration_reader (fixed_image_filename, moving_image_filename):
    """This function load 2 images from their path and return 2 ITK objects.
        
        Args:
            fixed_image_filename
            moving_image_filename
            
        Returns:
            fixed_image
            moving_image
    """
    
    
    #We load the images as float (itk.F) because is necessary for elastix
    fixed_image = itk.imread(fixed_image_filename, itk.F)
    moving_image = itk.imread(moving_image_filename, itk.F)
    
    return fixed_image, moving_image;
    


# In[7]:


def elastix_registration(fixed_image, moving_image):
    """This function do the registration of a moving image over a fixed image.
    
    Args:
        fixed_image
        moving_image
        
    Returns:
        registered_moving_image
        final_transform_parameters
    
    """
    
    parameter_object = itk.ParameterObject.New()
    default_rigid_parameter_map = parameter_object.GetDefaultParameterMap('rigid')
    parameter_object.AddParameterMap(default_rigid_parameter_map)
    
    result_image, result_transform_parameters = itk.elastix_registration_method(
    fixed_image, moving_image,
    parameter_object=parameter_object,
    log_to_console=False)
    
    print("The Registration is done!")
    
    return result_image, result_transform_parameters


# In[5]:


def registration_writer(image, file_path):
    itk.imwrite(image, file_path+"/registered_image.nii")
    print("Your file is written!")


# In[9]:


import tkinter as tk #serve per apire una finestra di dialogo per parire un determinato file
from tkinter import filedialog


#questi comandi inizializzano tkinter
root = tk.Tk()
root.withdraw()

#Fixed Image
#questo comando apre la finestra di dialogo e mi permette di scegliere il percorso del file
fixed_image_filename = filedialog.askopenfilename(title = "Select the fixed image")

#Moving Image
moving_image_filename = filedialog.askopenfilename(title = "Select the moving image")

#Where to save the registered image
output_filepath = filedialog.askdirectory(title = "Select where do you want to save the new image")


# In[11]:


fixed_image, moving_image = registration_reader(fixed_image_filename, moving_image_filename)
registered_image, registration_parameters = elastix_registration(fixed_image, moving_image)


# In[12]:


registration_writer(registered_image, output_filepath)


# In[ ]:




