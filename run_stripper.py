import itk
import numpy
import argparse
from datetime import datetime



from Neuroradiomics.skull_stripping import *



def parse_args():
    description = 'Automated MRI Brain Image Extraction'
    parser = argparse.ArgumentParser(description = description)

    parser.add_argument('--image',
                        dest='image',
                        required=True,
                        type=str,
                        action='store',
                        help='Head_Image filename')
    parser.add_argument('--atlas',
                        dest='atlas',
                        required=True,
                        type=str,
                        action='store',
                        help='Atlas_Image filename')
    parser.add_argument('--mask',
                        dest='mask',
                        required=True,
                        type=str,
                        action='store',
                        help='brain mask image filename')
    parser.add_argument('--output',
                        dest='output',
                        required=False,
                        type=str,
                        default='./',
                        action='store',
                        help='Output filepath. Default is ./')
    
    args = parser.parse_args()
    return args
    
    
def main():
    
    # parse the arguments
    args = parse_args()
    
    image = itk.imread(args.image, itk.F)
    atlas = itk.imread(args.atlas, itk.F)
    mask = itk.imread(args.mask, itk.F)
    
    brain = skull_stripper(image, atlas, mask)
    
    #find the actual date and time
    now = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
    
    itk.imwrite(brain, args.output+'/'+now+'_extracted_brain.nii')
    

if __name__ == '__main__':

    main()