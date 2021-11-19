import itk
import numpy as np
import argparse

from Neuroradiomics.registration import registration_reader
from Neuroradiomics.registration import elastix_registration
from Neuroradiomics.registration import registration_writer


def parse_args():
    description = 'Automated Elastix MRI Brain Image Registration'
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('--fixed',
                        dest='fixed_image',
                        required=True,
                        type=str,
                        action='store',
                        help='Fixed_Image filename')
    parser.add_argument('--moving',
                        dest='moving_image',
                        required=True,
                        type=str,
                        action='store',
                        help='Moving_Image filename')
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

    # parse the arguments: Fixed Image, Moving Image, Output file path (optional)
    args = parse_args()

    #read the fixed and moving image
    
    f_image, m_image = registration_reader(args.fixed_image, args.moving_image)
    
    
    #do the registration and write it
    registration_writer( elastix_registration(f_image, m_image, args.clog), args.output )


if __name__ == '__main__':

    main()