import itk
import numpy as np
import argparse


from Neuroradiomics.registration import apply_transform_from_files


def parse_args():
    description = 'Automated Transformix Image Tranformation'
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('--image',
                        dest='image_path',
                        required=True,
                        type=str,
                        action='store',
                        help='Image filename')
    parser.add_argument('--transform',
                        dest='transform_path',
                        required=True,
                        type=str,
                        action='store',
                        help='Path to the folder with the Transformation files')
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

    # parse the arguments: Image Path, Tranformation_Path, Output file path (optional)
    args = parse_args()

    #read the image
    
    image = itk.imread(args.image_path)

    result_image = apply_transform_from_files(image, args.transform_path)
    
    itk.imwrite(result_image, args.output + 'Result_image.nii')


if __name__ == '__main__':

    main()