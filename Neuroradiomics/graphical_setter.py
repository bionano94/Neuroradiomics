

import tkinter as tk
from tkinter import filedialog


#GRAPHIC INTERFACE TO SET THE READER AND THE WRITER

def graphical_setter():
    """This function opens some dialog windows to let choose the user the fixed image,
     the moving image and the path of the output file.
        
        Args: None
        
        
        Returns:
        
            fixed_image : string
                The complete path to the fixed image
                
            moving_image : string 
                The complete path to the moving image
                
            output_filepath : string
                The path to the directory where the output image will be written
    """
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
    
    return fixed_image_filename, moving_image_filename, output_filepath