__author__ = [ 'Riccado Biondi' ]
__email__  = [ 'riccardo.biondi4@studio.unibo.it' ]

import itk
import numpy as np

def itk_label_shape_statistics( image, labelmap ):
    '''
    Given an intensity image and a label map, compute min, max, variance and
    mean of the pixels associated with each label or segment.

    Parameters
    ----------
    image: itk.Image
        intensity image
    labelmap: itk.LabelMap
        label map

    Return
    ------
    filter_ : itk.LabelStatisticsImageFilter
        itk.LabelStatisticsImageFilter instance. The instance is not updated
    '''

    filter_ = itk.LabelStatisticsImageFilter[ type(image), type(labelmap) ].New()
    _ = filter_.SetLabelInput( labelmap )
    _ = filter_.SetInput( image )

    return filter_


def itk_shift_scale( image, shift = 0., scale = 1. ):
    '''
    Shift and scale the pixels in an image.

    Parameters
    ----------
    image: itk.Image
        image to apply filter to
    shift: float
        shift factor
    scale: float
        scale factor

    Return
    ------
    filter_ : itk.ShiftScaleImageFilter
        itk.ShiftScaleImageFilter instance. The instance is not updated
    '''
    #InputType = infer_itk_image_type(image, None)

    filter_ = itk.ShiftScaleImageFilter[type(image), type(image)].New()
    _ = filter_.SetInput( image )
    _ = filter_.SetScale( scale )
    _ = filter_.SetShift( shift )

    return filter_


def itk_gaussian_normalization( image, mask, label = 1 ):
    '''
    Normalize the data according to mean and standard deviation of the
    voxels inside the mask image

    Parameters
    ----------
    image: itk.Image
        image to normalize
    mask: itk.Image
        ROI mask
    label: int
        label value to determine the ROI
    '''

    stats = itk_label_shape_statistics( image, mask )
    _ = stats.Update()
    shift = -stats.GetMean( label )
    scale = 1. / abs( stats.GetSigma( label ) )
    normalized = itk_shift_scale( image, shift = shift, scale = scale )

    return normalized