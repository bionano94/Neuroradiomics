import itk
import numpy


def negative_3d_masking (image, mask):
    '''
    This function apply the negative of the image loaded as "mask" to a 3D image.
    
    Parameters
    ----------
        image: 3D itk object.
                The image you want to apply the mask on.
        
        mask: 3D itk object.
                The binary image of the mask you want to apply.
    
    Returns
    -------
        masked_image: 3D itk object.
                        The masked image.
    '''
    
    Dimension = 3
    
    masked_image = itk.Image[itk.F, Dimension].New()
    
    masked_image.SetRegions(image.GetLargestPossibleRegion())
    masked_image.Allocate()
    
    index = itk.Index[Dimension]()
    
    for index[0] in range( mask.GetLargestPossibleRegion().GetSize()[0] ):
    
        for index[1] in range( mask.GetLargestPossibleRegion().GetSize()[1] ):
        
            for index[2] in range( mask.GetLargestPossibleRegion().GetSize()[2] ):
            
                if mask.GetPixel(index) < 1e-01:
                    masked_image.SetPixel(index, 0)
                else: 
                    masked_image.SetPixel(index, image.GetPixel(index))
                
    
    return masked_image