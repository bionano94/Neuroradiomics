
from Neuroradiomics.registration import *
from Neuroradiomics.skull_stripping import *
from Neuroradiomics.normalization import *


import itk
import numpy as np
import os
from datetime import datetime
import glob


def match_atlases (changing_img, reference_img, displacement = [0, 0, 0]):
    '''
    This function matches the size and the origin of 2 images.
    Its useful when two atlases are radiomacally comparable but the origin are slightly changed.
    
    Parameters
    ----------
        changing_img: ITK Image object.
                      The atlas that would be transformed to match the reference_img.
        
        reference_img: ITK Image object.
                       The atlas that has to be matched.
                       
        displacement: Array of int.
                      The coordinates of the vector with which the changed Origin will be moved.
                      Default is [0, 0, 0].
                      
    Returns
    -------
        changin_img: ITK Image object.
                     The changing image matched with the reference_img.
                    
    
    '''
    
    changing_img.SetOrigin(reference_img.GetOrigin() + displacement)
    changing_img.SetDirection(reference_img.GetDirection())
    changing_img.GetLargestPossibleRegion().SetSize(reference_img.GetLargestPossibleRegion().GetSize())
    
    return changing_img


def find_connected_regions (image):
    '''
    This function finds the connected region in a binary image.
    
    Parameters
    ----------
        image: ITK image object.
               The image that has to be labeled.
               
    Returns
    -------
        connected_filter.GetOutput(): ITK image object.
                                      An image with each connected region labeled.
    
    '''
    
    #eseguo il cast per essere sicuro che la funzione venga applicata
    OutputType = itk.Image[itk.SS, 3]
    
    if type(image) != OutputType :
        
        cast_filter = itk.CastImageFilter[type(image), OutputType].New()
        cast_filter.SetInput(image)
        cast_filter.Update()
        image = cast_filter.GetOutput()

    #differentiating labeled damages
    connected_filter = itk.ConnectedComponentImageFilter[OutputType, OutputType].New()
    connected_filter.SetInput(image)
    connected_filter.Update()

    return connected_filter.GetOutput()


def scoring (label_img, pos_mask, neg_mask, pos_val = 1, neg_val = 1):
    '''
    This function evaluates the score for each labeled element.
    It gives a positive score for each pixel overlapping the potivie mask and a negative score for each pixel overlapping the negative mask.
    The score for each label is then divided by the total pixel number of that label.
    
    Parameters
    ----------
        label_img: ITK Image object.
                   The image containig the labeled elements.
        
        pos_mask: ITK Image object.
                  Mask which overlaps gives positive score.
        
        neg_mask: ITK Image object.
                  Mask which overlaps gives negative score.
        
        pos_val: Int number.
                 Score added for each pixel overlapping the pos_mask.
        
        neg_val: Int number.
                 Score subtracted for each pixel overlapping the pos_mask.
                 
                 
    Return
    ------
        pounded_score: list of float.
                       The averaged score for each label.
    
    '''
    #FINDING NUMBER OF LABELS
    maximum_filter = itk.MinimumMaximumImageCalculator[type(label_img)].New()
    maximum_filter.SetImage(label_img)
    maximum_filter.ComputeMaximum()

    index = itk.Index[3]()
    score = [0] * maximum_filter.GetMaximum() #score for each lesion
    count = [0] * maximum_filter.GetMaximum() #total number of pixels for lesion
    
    for index[0] in range( label_img.GetLargestPossibleRegion().GetSize()[0] ):

        for index[1] in range( label_img.GetLargestPossibleRegion().GetSize()[1] ):

            for index[2] in range( label_img.GetLargestPossibleRegion().GetSize()[2] ):

                if label_img.GetPixel(index) != 0 :
                    count[label_img.GetPixel(index) - 1] += 1

                    if neg_mask.GetPixel(index) == 1 :
                        score[label_img.GetPixel(index) - 1] += - neg_val # subtract points for neg masked pixel
                        
                    if  pos_mask.GetPixel(index) == 1 : 
                        score[label_img.GetPixel(index) - 1] += pos_val # adding points for pos masked pixel

    pounded_score = [x/y for x,y in zip(score, count)]
    
    return pounded_score