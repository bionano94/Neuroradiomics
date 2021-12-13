import itk
import numpy
import tkinter as tk
from tkinter import filedialog

from Neuroradiomics.registration import registration_reader
from Neuroradiomics.registration import elastix_multimap_registration
from Neuroradiomics.registration import registration_writer
from Neuroradiomics.registration import apply_transform_from_files
from Neuroradiomics.registration import read_transform_from_files
from Neuroradiomics.registration import elastix_rigid_registration
from Neuroradiomics.graphical_setter import graphical_setter
from Neuroradiomics.skull_stripping import skull_stripper

def main():
    root = tk.Tk()
    root.withdraw()
    
    mask_file = filedialog.askopenfilename(title = "Select the mask image")
    transform_path = filedialog.askdirectory(title = "Select where are the transform files")
    mask = itk.imread(mask_file, itk.F)
    reg_mask = apply_transform_from_files(mask, transform_path)
    registered_brain_file = filedialog.askopenfilename(title = "Select the image")
    registered_image = itk.imread(registered_brain_file)
    
    brain = skull_stripper(registered_image, reg_mask)
    
    output_path = filedialog.askdirectory(title = "Select where you want to save the image")
    
    itk.imwrite(brain, output_path)
    

if __name__ == '__main__':

    main()