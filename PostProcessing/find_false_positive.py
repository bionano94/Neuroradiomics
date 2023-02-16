
from Neuroradiomics.registration import *
from Neuroradiomics.skull_stripping import *
from Neuroradiomics.normalization import *


import itk
import numpy as np
import os
from datetime import datetime
import glob


def match_atlases (changing_img, reference_img, displacement = [0, 0, 0], registration = False):
    '''
    This function matches the size and the origin of 2 images.
    
    Parameters
    ----------
        changing_img: ITK Image object.
                      The atlas that would be transformed to match the reference_img.
        
        reference_img: ITK Image object.
                       The atlas that has to be matched.
                       
        displacement: Array of int.
                      The coordinates of the vector with which the changed Origin will be moved.
                      Default is [0, 0, 0].
                      
        registration: Boolean value
                      If true also a Rigid Registration will be used to match the two atlases.
                      
    Returns
    -------
        changin_img: ITK Image object.
                     The changing image matched with the reference_img.
                    
    
    '''
    
    changing_img.SetOrigin(reference_img.GetOrigin() + displacement)
    changing_img.SetDirection(reference_img.GetDirection())
    changing_img.GetLargestPossibleRegion().SetSize(reference_img.GetLargestPossibleRegion().GetSize())
    
    
    if registration:
        elastix_obj = elastix_rigid_registration (changing_img, reference_img)
        changing_img = itk.transformix_filter(changing_img, 
                                 Set_parameters_map_attribute(elastix_obj.GetTransformParameterObject(), 
                                                              'ResampleInterpolator', 
                                                              'FinalNearestNeighborInterpolator'
                                                             )
                                )
    
    return changing_img


#################################
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




#############
# SCORING 
#############
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


##################
def feature_scoring (label_img, masks_list):
    '''
    This function evaluates the score for each feature of each labeled element (with a value greater than 0.5).
    It returns 2 dimensional array. For every labeled element it contains n-masks + 2 elements.
    The first feature is the average of the pixel's value for each label, the second value is the label volume.
    
    Parameters
    ----------
        label_img: ITK Image object.
                   The image containig the labeled elements.
        
        masks_list: list of ITK Image.
                    The masks you wanna use to compute some features.
                 
                 
    Return
    ------
        pounded_score: Array of float.
                       The score for each features of each label.
                       
        counted_label: ITK Image object.
                   The image containig the labeled elements differentiated one from another.
    
    '''
    #binarize the label
    bin_label = binarize (label_img, 0.5)
    
    #differentiating the labels
    counted_label = find_connected_regions (bin_label)
    
    voxel_volume = label_img.GetSpacing()[0] * label_img.GetSpacing()[1] * label_img.GetSpacing()[2] 
    
    #FINDING NUMBER OF LABELS
    maximum_filter = itk.MinimumMaximumImageCalculator[type(counted_label)].New()
    maximum_filter.SetImage(counted_label)
    maximum_filter.ComputeMaximum()

    index = itk.Index[3]()
    pounded_score = np.array( [[0.] * (len(masks_list) + 2)] * maximum_filter.GetMaximum() ) #score for lesions divided pixels per lesion
    score = np.array( [[0.] * (len(masks_list) + 2)] * maximum_filter.GetMaximum() ) #total scores for each lesion
    count = np.array( [0.] * maximum_filter.GetMaximum() ) #total number of pixels for lesion
    
    for index[0] in range( label_img.GetLargestPossibleRegion().GetSize()[0] ):

        for index[1] in range( label_img.GetLargestPossibleRegion().GetSize()[1] ):

            for index[2] in range( label_img.GetLargestPossibleRegion().GetSize()[2] ):

                if label_img.GetPixel(index) != 0 :
                    count[counted_label.GetPixel(index) - 1] += 1
                    
                    score[counted_label.GetPixel(index) - 1][0] += label_img.GetPixel(index)
                    
                    for i in range(len(masks_list)):
                        score[counted_label.GetPixel(index) - 1][i + 2] += masks_list[i].GetPixel(index)
               
                    
    
    for i in range(len(score)):
        pounded_score[i][0] = score[i][0] / count[i]
        pounded_score[i][1] = voxel_volume * count[i]
        pounded_score[i][2] = score[i][2] / count[i]
        pounded_score[i][3] = score[i][3] / count[i]
    
    return pounded_score, counted_label


#############################
# FIND TRUTH VALUE
#############################
def find_Jaccard_truth_value (counted_label, gnd_label, threshold = 0.25):
    
    '''
    This function is meant to be used for finding the classification of the training set.
    It gives an array of bools of the size of the number of labels.
    A label will be considered true if the modified Jaccard score is greater than the treshold.
    
    Parameters
    ----------
        counted_label: ITK Image object.
                       The image containing the supposed labels. Every not connected object must have a different int value.
        
        
        gnd_label: ITK Image object.
                    The image containing the labels considered as Ground Truth. It must be binary.
                    
        threshold: Float number.
                   The threshold for the modified Jaccard score to consider a label True.
    
    
    Return
    ------
        value_array: Array of bool.
                     A list with the value of truth for every supposed label. Default is 0.25
                     
        jaccard_index: Array of float.
                       An array with the modified Jaccard score for each label
    
    
    '''
    
    #FINDING NUMBER OF LABELS
    maximum_filter = itk.MinimumMaximumImageCalculator[type(counted_label)].New()
    maximum_filter.SetImage(counted_label)
    maximum_filter.ComputeMaximum()
    num_labels = maximum_filter.GetMaximum()
    
    count_labels = np.array([0] * num_labels) #total number of pixels for lesion
    value_array = np.array( [False] * num_labels ) #array that reports what label is true
    
    
    
    #finding number of GND Truth label
    counted_gnd = find_connected_regions(gnd_label)
    #FINDING NUMBER OF GND LABELS
    maximum_filter = itk.MinimumMaximumImageCalculator[type(counted_gnd)].New()
    maximum_filter.SetImage(counted_gnd)
    maximum_filter.ComputeMaximum()
    num_gnd = maximum_filter.GetMaximum()
    
    count_gnd = np.array( [0] * num_gnd ) #total number of pixels for lesion
    
    
    overlapping_matrix = np.array( [[0]*num_gnd]*num_labels)
    
    jaccard_index = np.array([0.]*num_labels)
    
    
    index = itk.Index[3]()

    for index[0] in range( counted_label.GetLargestPossibleRegion().GetSize()[0] ):

            for index[1] in range( counted_label.GetLargestPossibleRegion().GetSize()[1] ):

                for index[2] in range( counted_label.GetLargestPossibleRegion().GetSize()[2] ):
                    
                    if counted_gnd.GetPixel(index) != 0 :
                        count_gnd[ counted_gnd.GetPixel(index) - 1 ] +=1
                        
                    if counted_label.GetPixel(index) != 0 :
                        count_labels[ counted_label.GetPixel(index) - 1 ] +=1
                        
                        
                        if counted_gnd.GetPixel(index) != 0 :
                            overlapping_matrix[counted_label.GetPixel(index) - 1][counted_gnd.GetPixel(index) - 1] += 1
                            
    
    
    
    num = np.array([0]*num_labels)
    den = np.array([0]*num_labels)
    
    
    for i in range(num_labels):
        for j in range(num_gnd):
            num[i] += overlapping_matrix[i][j]
            if overlapping_matrix[i][j]!= 0 : den[i] += count_gnd[j]
        den[i] += count_labels[i] - num[i]
        
        jaccard_index[i] = num[i]/den[i]
        
        value_array[i] = (jaccard_index[i] >= threshold)
        
    
    return value_array, jaccard_index


################################

def find_simple_truth_value (counted_label, true_label):
    
    '''
    This function is meant to be used for finding the classification of the training set.
    It gives an array of bools of the size of the number of labels.
    A label will be considered true if at least one pixel overlaps the ground truth.
    
    Parameters
    ----------
        counted_label: ITK Image object.
                       The image containing the supposed labels. Every not connected object must have a different int value.
        
        
        true_label: ITK Image object.
                    The image containing the labels considered as Ground Truth. It must be binary.
    
    
    Return
    ------
        value_array: List of bool.
                     A list with the value of truth for every supposed label.
    
    
    '''
    
    #FINDING NUMBER OF LABELS
    maximum_filter = itk.MinimumMaximumImageCalculator[type(counted_label)].New()
    maximum_filter.SetImage(counted_label)
    maximum_filter.ComputeMaximum()
    
    index = itk.Index[3]()
    value_array = np.array( [False] * maximum_filter.GetMaximum() ) #total number of pixels for lesion
    

    for index[0] in range( counted_label.GetLargestPossibleRegion().GetSize()[0] ):

            for index[1] in range( counted_label.GetLargestPossibleRegion().GetSize()[1] ):

                for index[2] in range( counted_label.GetLargestPossibleRegion().GetSize()[2] ):

                    if counted_label.GetPixel(index) != 0 :
                        value_array[counted_label.GetPixel(index) - 1] = value_array[counted_label.GetPixel(index) - 1] or (true_label.GetPixel(index) == 1)
    
    return value_array



##################
# Selecting Labels
##################

def label_killer (counted_label, surviving_array):
    
    '''
    This function "kills" every label that is not considered TRUE.
    
    Parameters
    ----------
        counted_label: ITK Image object.
                       The image containing the labels. Every not connected object must have a different int value.

        
        surviving_array: List of bool.
                         A list with the value of truth for every label. It must have the size of the total number of labels
        
        
    Return
    ------
        final_label: ITK Image object.
                     The image containing only the labels considered 'TRUE'.
        
    '''
    
    #FINDING NUMBER OF LABELS
    maximum_filter = itk.MinimumMaximumImageCalculator[type(counted_label)].New()
    maximum_filter.SetImage(counted_label)
    maximum_filter.ComputeMaximum()
    
    Dimension = 3
    ImageType = itk.template(counted_label)[1]
    final_label = itk.Image[ImageType].New()
    
    final_label.SetRegions( counted_label.GetLargestPossibleRegion() )
    final_label.SetSpacing( counted_label.GetSpacing() )
    final_label.SetOrigin( counted_label.GetOrigin() )
    final_label.SetDirection( counted_label.GetDirection() )
    final_label.Allocate()
    
    index = itk.Index[3]()

    for index[0] in range( counted_label.GetLargestPossibleRegion().GetSize()[0] ):

            for index[1] in range( counted_label.GetLargestPossibleRegion().GetSize()[1] ):

                for index[2] in range( counted_label.GetLargestPossibleRegion().GetSize()[2] ):

                    if counted_label.GetPixel(index) != 0 :
                        
                        if (surviving_array[ int(counted_label.GetPixel(index)) - 1 ]) : final_label.SetPixel(index, counted_label.GetPixel(index))
                        else: final_label.SetPixel(index, 0)
                        
                    else: final_label.SetPixel(index, 0)
                        
    return final_label