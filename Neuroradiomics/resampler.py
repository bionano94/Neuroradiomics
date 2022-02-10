__author__ = [ 'Riccado Biondi' ]
__email__  = [ 'riccardo.biondi4@studio.unibo.it' ]

import itk

def match_physical_spaces(image, reference):

    NNInterpolatorType = itk.NearestNeighborInterpolateImageFunction[type(image),
                                                                     itk.D]
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

    return resampler