from Neuroradiomics.segmentation import *
from Neuroradiomics.evaluation_utilities import *
from PostProcessing.find_false_positive import *

import itk
import numpy as np
from sklearn.mixture import GaussianMixture
import os
import sys

import pytest
import hypothesis.strategies as st
from hypothesis import given, settings, assume
from hypothesis import HealthCheck as HC



# ███████ ████████ ██████   █████  ████████ ███████  ██████  ██ ███████ ███████
# ██         ██    ██   ██ ██   ██    ██    ██      ██       ██ ██      ██
# ███████    ██    ██████  ███████    ██    █████   ██   ███ ██ █████   ███████
#      ██    ██    ██   ██ ██   ██    ██    ██      ██    ██ ██ ██           ██
# ███████    ██    ██   ██ ██   ██    ██    ███████  ██████  ██ ███████ ███████


@st.composite
def random_image_strategy(draw):
    '''
    This function generates a random image.
    '''
    
    PixelType = itk.F
    ImageType = itk.Image[PixelType, 3]
    Size = ( draw(st.integers(90,100)), draw(st.integers(90,100)) , draw(st.integers(90,100)) )
    Origin = ( draw(st.integers(0,90)), draw(st.integers(0,90)) , draw(st.integers(0,90)) )
    
    rndImage = itk.RandomImageSource[ImageType].New()
    rndImage.SetSize(Size)
    rndImage.SetOrigin(Origin)
    rndImage.SetNumberOfWorkUnits(1)
    rndImage.UpdateLargestPossibleRegion()

    return rndImage.GetOutput()



    

                                      
@st.composite
def label_image_strategy(draw):
    '''
    This function generates an image with 8 different object of value 1.
    '''
    
    x_size = 200
    y_size = 200
    z_size = 200
    image = np.zeros([x_size, y_size, z_size], dtype = np.float32) 
    
    x1 = 10
    y1 = 10
    z1 = 10
    side1 = draw(st.integers(1, 30))
    
    x2 = 60
    y2 = 60
    z2 = 60
    side2 = draw(st.integers(1, 30))
    
    side3 = draw(st.integers(1, 30))
    
    side4 = draw(st.integers(1, 30))
    
    
    for x in range (x1, x1+side1):
        for y in range (y1, y1+side1):
            for z in range (z1, z1+side1):
                image[x,y,z]=1
                
    for x in range (x1, x1+side1):
        for y in range (y1, y1+side1):
            for z in range (z2, z2+side3):
                image[x,y,z]=1
                
    
    for x in range (x2, x2+side2):
        for y in range (y2, y2+side2):
            for z in range (z2, z2+side2):
                image[x,y,z]=1
    
    for x in range (x2, x2+side2):
        for y in range (y2, y2+side2):
            for z in range (z1, z1+side4):
                image[x,y,z]=1
                
    
    for x in range (x1, x1+side1):
        for y in range (y2, y2+side1):
            for z in range (z1, z1+side1):
                image[x,y,z]=1
    

    for x in range (x2, x2+side2):
        for y in range (y1, y1+side2):
            for z in range (z2, z2+side2):
                image[x,y,z]=1

                
    for x in range (x2, x2+side1):
        for y in range (y1, y1+side2):
            for z in range (z1, z1+side3):
                image[x,y,z]=1
                
                
    for x in range (x1, x1+side2):
        for y in range (y2, y2+side3):
            for z in range (z2, z2+side4):
                image[x,y,z]=1
                
    
    image = itk.image_view_from_array(image)
    
    return image
              
    
    
#############################################
########## NON STRATEGY FUNCTIONS ###########
#############################################

def white_image():
    '''
    This function generates a 3D image with every pixel of value 1.
    '''
    
    x_size = 200
    y_size = 200
    z_size = 200
    image = itk.image_view_from_array( np.full((x_size, y_size, z_size), 1, dtype = np.float32 ) )
    
    
    return image

def black_image():
    '''
    This function generates a cubic black image.
    '''
    
    x_size = 200
    y_size = 200
    z_size = 200
    image = itk.image_view_from_array( np.zeros([x_size, y_size, z_size], dtype = np.float32) )
    
    return image



# ████████ ███████ ███████ ████████ ███████ 
#    ██    ██      ██         ██    ██      
#    ██    █████   ███████    ██    ███████
#    ██    ██           ██    ██         ██
#    ██    ███████ ███████    ██    ███████



# Is Pytest working?

def test_pytest_properly_works():
    '''
    This function just checks the proper work of pytest
    '''
    assert 1 == 1
    


#Testing the matching alias function
@given(changing = random_image_strategy(), ref = random_image_strategy() )
@settings(max_examples = 10, deadline = None)
def test_matching_atlases_no_displacement (changing, ref):
    '''
    This function generates two random images and then apply the "match_atlases" function without the displacement vector and then checks if the two images have the same direction and if the origin is equal to the reference origin.
    '''
    matched = match_atlases(changing, ref)
    
    assert matched.GetOrigin() == ref.GetOrigin()
    assert matched.GetDirection() == ref.GetDirection()
    

@given(changing = random_image_strategy(), ref = random_image_strategy() )
@settings(max_examples = 10, deadline = None)
def test_matching_atlases_with_displacement (changing, ref):
    '''
    This function generates two random images and then apply the "match_atlases" function with the displacement vector and then check
    if the two images have the same direction and if the origin is equal to the reference origin + dicplacement.
    '''
    
    displacement = [np.random.randint(0,10),np.random.randint(0,10),np.random.randint(0,10)]
    matched = match_atlases(changing, ref, displacement)
    
    assert matched.GetOrigin() - ref.GetOrigin() == displacement
    assert matched.GetDirection() == ref.GetDirection()
    
    
    
    
#testing the function to relabel the label image
@given(image = label_image_strategy())
@settings(max_examples = 10, deadline = None)
def test_find_connected_regions(image):
    '''
    This function take as input an image with 8 disconnected regions of pixel values = 1 and check if the find_connected_regions
    relabel every region just findind that the final max pixel value is = 8.
    '''
    
    relabeled_img = find_connected_regions(image)
    
    #FINDING NUMBER OF LABELS
    maximum_filter = itk.MinimumMaximumImageCalculator[type(relabeled_img)].New()
    maximum_filter.SetImage(relabeled_img)
    maximum_filter.ComputeMaximum()
    
    assert maximum_filter.GetMaximum() == 8
    

    
#Testing Score Function
@given(label = label_image_strategy() )
@settings(max_examples = 10, deadline = None)
def test_all_positive_score (label):
    '''
    This function checks if givin as masks for the scoring function an image with every pixel =1 (pos_mask) and a black image (neg_mask) the score is 1.
    '''
    
    pos_mask = white_image()
    neg_mask = black_image()
    relabelled = find_connected_regions(label) #relabel the image to differentiate every connected region
    
    score = scoring(relabelled, pos_mask, neg_mask)
    
    maximum_filter = itk.MinimumMaximumImageCalculator[type(relabelled)].New()
    maximum_filter.SetImage(relabelled)
    maximum_filter.ComputeMaximum()
    
    
    assert np.size(score) == maximum_filter.GetMaximum()
    assert score == [1] * len(score)
    
#Testing Score Function
@given(label = label_image_strategy())
@settings(max_examples = 10, deadline = None)
def test_all_negative_score (label):
    '''
    This function checks if giving as masks for the scoring function an image with every pixel = 0 (pos_mask) and an image with every pixel =1 (neg_mask) the score is -1 for every label.
    '''
    
    pos_mask = black_image()
    neg_mask = white_image() 
    
    relabelled = find_connected_regions(label) #relabel the image to differentiate every connected region
    
    score = scoring(relabelled, pos_mask, neg_mask)
    
    maximum_filter = itk.MinimumMaximumImageCalculator[type(relabelled)].New()
    maximum_filter.SetImage(relabelled)
    maximum_filter.ComputeMaximum()
    
    
    assert np.size(score) == maximum_filter.GetMaximum()
    assert score == [-1] * len(score)
    
#Testing Score Function
@given(label = label_image_strategy(),)
@settings(max_examples = 10, deadline = None)
def test_all_overlapping_score (label):
    '''
    This function checks if giving as masks for the scoring function two images with every pixel = 1 (pos_mask and neg_mask) the final score for every label is 0.
    '''
    
    pos_mask = white_image()
    neg_mask = white_image() 
    
    relabelled = find_connected_regions(label) #relabel the image to differentiate every connected region
    
    score = scoring(relabelled, pos_mask, neg_mask)
    
    maximum_filter = itk.MinimumMaximumImageCalculator[type(relabelled)].New()
    maximum_filter.SetImage(relabelled)
    maximum_filter.ComputeMaximum()
    
    
    assert np.size(score) == maximum_filter.GetMaximum()
    assert score == [0] * len(score)
    
    
    

#Testing Features Score Function
@given(label = label_image_strategy())
@settings(max_examples = 10, deadline = None)
def test_feature_scoring (label):
    '''
    This function checks the feature score function giving two white masks at the feature scoring function for a label image where every pixel has value 1 and tests if the score associated with those features is 1.
    '''
    pos_mask = white_image()
    neg_mask = white_image() 
    masks_list = [pos_mask, neg_mask]
    
    score, relabelled = feature_scoring(label, masks_list)
    
    maximum_filter = itk.MinimumMaximumImageCalculator[type(relabelled)].New()
    maximum_filter.SetImage(relabelled)
    maximum_filter.ComputeMaximum()
    
    adjusted_score = np.delete(score, 1, 1)
    
    assert len(score) == maximum_filter.GetMaximum()
    assert np.all( adjusted_score == [[1, 1, 1]] * len(score) )
    
    
    
    
    #Testing Features Score Function
@given(label = label_image_strategy())
@settings(max_examples = 10, deadline = None)
def test_feature_scoring2 (label):
    '''
    This function checks the feature score function giving two black masks at the feature scoring function for a label image where every pixel has value 1 and tests if the score associated with those features is 1 for the average pixel value and 0 for the masks overlapping.
    '''
    pos_mask = black_image()
    neg_mask = black_image() 
    masks_list = [pos_mask, neg_mask]
    
    score, relabelled = feature_scoring(label, masks_list)
    
    relabelled_test = find_connected_regions(label)
    maximum_filter = itk.MinimumMaximumImageCalculator[type(relabelled)].New()
    maximum_filter.SetImage(relabelled)
    maximum_filter.ComputeMaximum()
    
    adjusted_score = np.delete(score, 1, 1)
    
    assert np.all(np.isclose( itk.GetArrayFromImage(relabelled_test), itk.GetArrayFromImage(relabelled), 1e-3, 1e-2 ) ) 
    assert len(score) == maximum_filter.GetMaximum()
    assert np.all( adjusted_score == [[1, 0, 0]] * len(score) )
    
    
    
    
    
#Testing the FINDING SIMPLE TRUTH Function
@given(label = label_image_strategy())
@settings(max_examples = 10, deadline = None)
def test_find_simple_truth_all_false (label):
    '''
    This function checks the simple_truth function giving a black image as a mask and then checking that each label is considered false.
    '''
     
    gnd_truth = black_image()
    relabelled = find_connected_regions(label)
    maximum_filter = itk.MinimumMaximumImageCalculator[type(relabelled)].New()
    maximum_filter.SetImage(relabelled)
    maximum_filter.ComputeMaximum()
    
    value_array = find_simple_truth_value(relabelled, gnd_truth)
    
    
    assert len(value_array) == maximum_filter.GetMaximum()
    assert np.all( value_array == [False] * len(value_array) )
    

#Testing the FINDING TRUTH Function
@given(label = label_image_strategy())
@settings(max_examples = 10, deadline = None)
def test_find_simple_truth_all_true (label):
    '''
    This function checks the simple_truth function giving an image with every pixel of value 1 as a mask and then checking that each label is considered true.
    '''
     
    gnd_truth = white_image()
    relabelled = find_connected_regions(label)
    maximum_filter = itk.MinimumMaximumImageCalculator[type(relabelled)].New()
    maximum_filter.SetImage(relabelled)
    maximum_filter.ComputeMaximum()
    
    value_array = find_simple_truth_value(relabelled, gnd_truth)
    
    
    assert len(value_array) == maximum_filter.GetMaximum()
    assert np.all( value_array == [True] * len(value_array) )
    
    
    
    
    
#Testing the FINDING TRUTH Function
@given(label = label_image_strategy())
@settings(max_examples = 10, deadline = None)
def test_find_Jaccard_truth_all_false(label):
    '''
    This function checks the Jaccard_truth function giving a black image as a mask and then checking that each label is considered false and the modified Jaccard index for each label is 0.
    '''
    gnd_truth = black_image()

    relabelled = find_connected_regions(label)
    maximum_filter = itk.MinimumMaximumImageCalculator[type(relabelled)].New()
    maximum_filter.SetImage(relabelled)
    maximum_filter.ComputeMaximum()
    
    value_array, jaccard_index = find_Jaccard_truth_value(relabelled, gnd_truth)
    
    
    assert len(value_array) == maximum_filter.GetMaximum()
    assert np.all( value_array == [False] * len(value_array) )
    assert np.all( jaccard_index == [0] * len(value_array) )
    
    
#Testing the FINDING TRUTH Function
@given(label = label_image_strategy() )
@settings(max_examples = 10, deadline = None)
def test_find_Jaccard_truth_all_true(label):
    '''
    This function checks the Jaccard_truth function giving an image with every pixel of value 1 as a mask and then checking that each label is considered true and the modified Jaccard index for each label is 1.
    '''
        
     
    relabelled = find_connected_regions(label)
    maximum_filter = itk.MinimumMaximumImageCalculator[type(relabelled)].New()
    maximum_filter.SetImage(relabelled)
    maximum_filter.ComputeMaximum()
    
    value_array, jaccard_index = find_Jaccard_truth_value(relabelled, relabelled)
    
    
    assert len(value_array) == maximum_filter.GetMaximum()
    assert np.all( value_array == [True] * len(value_array) )
    assert np.all( jaccard_index == [1] * len(value_array) )
    
    
    
#Testing the LABEL KILLER function
@given(label = label_image_strategy() )
@settings(max_examples = 10, deadline = None)
def test_label_killer_all_dead(label):
    '''
    This function checks the proper funcionlity of the label_kiler function setting all labels as False and then checking that every label was deleted giving in output a black image with the same attributes of the one given as input.
    '''
    
    relabelled = find_connected_regions(label)
    maximum_filter = itk.MinimumMaximumImageCalculator[type(relabelled)].New()
    maximum_filter.SetImage(relabelled)
    maximum_filter.ComputeMaximum()
    
    killer_array = np.array([False]*8)
    
    no_survivors = label_killer(relabelled, killer_array)
    
    maximum_filter = itk.MinimumMaximumImageCalculator[type(no_survivors)].New()
    maximum_filter.SetImage(no_survivors)
    maximum_filter.ComputeMaximum()
    
    assert maximum_filter.GetMaximum() == 0
    assert no_survivors.GetLargestPossibleRegion() == relabelled.GetLargestPossibleRegion()
    
    
#Testing the LABEL KILLER function
@given(label = label_image_strategy() )
@settings(max_examples = 10, deadline = None)
def test_label_killer_all_survived(label):
    '''
    This function checks the proper funcionlity of the label_kiler function setting all labels as True and then checking that every label was keeped giving in output the same image given in input with also the same attributes.
    '''
    
    relabelled = find_connected_regions(label)
    maximum_filter = itk.MinimumMaximumImageCalculator[type(relabelled)].New()
    maximum_filter.SetImage(relabelled)
    maximum_filter.ComputeMaximum()
    
    killer_array = np.array([True]*8)
    
    survivors = label_killer(relabelled, killer_array)
    
    post_maximum_filter = itk.MinimumMaximumImageCalculator[type(survivors)].New()
    post_maximum_filter.SetImage(survivors)
    post_maximum_filter.ComputeMaximum()
    
    assert maximum_filter.GetMaximum() == post_maximum_filter.GetMaximum()
    assert survivors.GetLargestPossibleRegion() == relabelled.GetLargestPossibleRegion()
    
    
    
#Testing the SMALL LABEL REMOVER function
@given(label = label_image_strategy() )
@settings(max_examples = 10, deadline = None)
def test_remove_small_labels_by_dimension_survivor(label):
    '''
    This function checks if applying the small-labels-remover function with minimum pixel dimension = 1 to an image the output image is equal to the input ones.
    '''
    
    all_survived_label = remove_small_labels_by_dimension(label)
 
    
    
    assert np.all(np.isclose( itk.GetArrayFromImage(label), itk.GetArrayFromImage(all_survived_label), 1e-3, 1e-2 ) )
    
    
@given(label = label_image_strategy() )
@settings(max_examples = 10, deadline = None)
def test_remove_small_labels_by_dimension_killed(label):
    '''
    This function checks if applying the small-labels-remover function with minimum pixel dimension grater than every possible object in the image the output image is totally black.
    '''
    
    all_killed_label = remove_small_labels_by_dimension(label, 27001)
    
    relabeled_img = find_connected_regions(all_killed_label)
    
    #FINDING NUMBER OF LABELS
    maximum_filter = itk.MinimumMaximumImageCalculator[type(relabeled_img)].New()
    maximum_filter.SetImage(relabeled_img)
    maximum_filter.ComputeMaximum()
    
    
    assert maximum_filter.GetMaximum() == 0