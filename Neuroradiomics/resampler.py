__author__ = [ 'Riccado Biondi' ]
__email__  = [ 'riccardo.biondi4@studio.unibo.it' ]

import itk

def match_physical_spaces(image, reference):
    '''
    This function matches the physical spaces of an image with another without modifying the image.
    
    Parameters
    ----------
            image: itk Image object.
                   The image for which the physical spaces is intended to be change.
                   
            reference: itk Image object.
                    The image which physical space will be matched.
                    
    Return
    ------
            resampler: itk Filter object.
                    The filter used to match the physical space. 
                    In order to obtain the matched image call the GetOutput() function to the output of this function.
                    (e.g. resampler = match_physical_spaces(image, reference)
                          new_image = resampler.GetOutput()
                          )
    '''

    NNInterpolatorType = itk.NearestNeighborInterpolateImageFunction[type(image),itk.D]
    
    interpolator = NNInterpolatorType.New()

    TransformType = itk.IdentityTransform[itk.D, 3]
    transformer = TransformType.New()
    _ = transformer.SetIdentity()

    resampler = itk.ResampleImageFilter[type(image), type(image)].New()
    _ = resampler.SetInterpolator(interpolator)
    _ = resampler.SetTransform(transformer)
    _ = resampler.SetUseReferenceImage(True)
    _ = resampler.SetReferenceImage(reference)
    _ = resampler.SetInput(image)
    _ = resampler.Update()

    return resampler