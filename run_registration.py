import itk
import numpy as np
import argparse

from Neuroradiomics.registration import registration_reader
from Neuroradiomics.registration import elastix_multimap_registration
from Neuroradiomics.registration import registration_writer
from Neuroradiomics.registration import elastix_rigid_registration



def parse_args():
    description = 'Automated Elastix MRI Brain Image Registration'
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('--t1',
                        dest='t1_image',
                        required=True,
                        type=str,
                        action='store',
                        help='t1_Image filename')
    parser.add_argument('--flair',
                        dest='flair_image',
                        required=True,
                        type=str,
                        action='store',
                        help='Flair_Image filename')
    parser.add_argument('--atlas',
                        dest='atlas_image',
                        required=True,
                        type=str,
                        action='store',
                        help='Atlas_Image filename')
    parser.add_argument('--log_to_console',
                        dest='clog',
                        required=False,
                        type=bool,
                        default=False,
                        action='store',
                        help='If True the console log is activated. Default is False')
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

    
    #read the fixed and moving image
    flair_image, t1_image = registration_reader(args.flair_image, args.t1_image)
    
    #Registrate the t1 image over the flair image
    
    t1_moved_image = elastix_rigid_registration(flair_image, t1_image, args.clog)
    
    atlas = itk.imread(args.atlas_image, itk.F)
    
    #do the registration and write it
    directory = registration_writer( elastix_multimap_registration(t1_moved_image, atlas, args.clog), args.output )

    itk.imwrite(t1_moved_image, directory+'/registered_t1.nii')

if __name__ == '__main__':

    main()