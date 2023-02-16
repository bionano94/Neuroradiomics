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
def white_image_strategy(draw):
    '''
    This function generates a white image
    '''
    
    x_size = 200
    y_size = 200
    z_size = 200
    image = itk.image_view_from_array( np.full((x_size, y_size, z_size), 1, dtype = np.float32 ) )
    
    
    return image

@st.composite
def black_image_strategy(draw):
    '''
    This function generates a black image
    '''
    
    x_size = 200
    y_size = 200
    z_size = 200
    image = itk.image_view_from_array( np.zeros([x_size, y_size, z_size], dtype = np.float32) )
    
    return image
    

                                      
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
@settings(max_examples = 20, deadline = None)
def test_matching_atlases_no_displacement (changing, ref):
    
    matched = match_atlases(changing, ref)
    
    assert matched.GetOrigin() == ref.GetOrigin()
    assert matched.GetDirection() == ref.GetDirection()
    

@given(changing = random_image_strategy(), ref = random_image_strategy() )
@settings(max_examples = 20, deadline = None)
def test_matching_atlases_with_displacement (changing, ref):
    
    displacement = [np.random.randint(0,10),np.random.randint(0,10),np.random.randint(0,10)]
    matched = match_atlases(changing, ref, displacement)
    
    assert matched.GetOrigin() - ref.GetOrigin() == displacement
    assert matched.GetDirection() == ref.GetDirection()
    
    
    
    
#testing the function to relabel the label image
@given(image = label_image_strategy())
@settings(max_examples = 20, deadline = None)
def test_find_connected_regions(image):
    
    relabeled_img = find_connected_regions(image)
    
    #FINDING NUMBER OF LABELS
    maximum_filter = itk.MinimumMaximumImageCalculator[type(relabeled_img)].New()
    maximum_filter.SetImage(relabeled_img)
    maximum_filter.ComputeMaximum()
    
    assert maximum_filter.GetMaximum() == 8
    

    
#Testing Score Function
@given(label = label_image_strategy(), pos_mask = white_image_strategy(), neg_mask = black_image_strategy() )
@settings(max_examples = 20, deadline = None)
def test_all_positive_score (label, pos_mask, neg_mask):
    
    relabelled = find_connected_regions(label) #relabel the image to differentiate every connected region
    
    score = scoring(relabelled, pos_mask, neg_mask)
    
    maximum_filter = itk.MinimumMaximumImageCalculator[type(relabelled)].New()
    maximum_filter.SetImage(relabelled)
    maximum_filter.ComputeMaximum()
    
    
    assert np.size(score) == maximum_filter.GetMaximum()
    assert score == [1] * len(score)
    
#Testing Score Function
@given(label = label_image_strategy(), pos_mask = black_image_strategy(), neg_mask = white_image_strategy() )
@settings(max_examples = 20, deadline = None)
def test_all_negative_score (label, pos_mask, neg_mask):
    
    relabelled = find_connected_regions(label) #relabel the image to differentiate every connected region
    
    score = scoring(relabelled, pos_mask, neg_mask)
    
    maximum_filter = itk.MinimumMaximumImageCalculator[type(relabelled)].New()
    maximum_filter.SetImage(relabelled)
    maximum_filter.ComputeMaximum()
    
    
    assert np.size(score) == maximum_filter.GetMaximum()
    assert score == [-1] * len(score)
    
#Testing Score Function
@given(label = label_image_strategy(), pos_mask = white_image_strategy(), neg_mask = white_image_strategy() )
@settings(max_examples = 20, deadline = None)
def test_all_overlapping_score (label, pos_mask, neg_mask):
    
    relabelled = find_connected_regions(label) #relabel the image to differentiate every connected region
    
    score = scoring(relabelled, pos_mask, neg_mask)
    
    maximum_filter = itk.MinimumMaximumImageCalculator[type(relabelled)].New()
    maximum_filter.SetImage(relabelled)
    maximum_filter.ComputeMaximum()
    
    
    assert np.size(score) == maximum_filter.GetMaximum()
    assert score == [0] * len(score)
    
    
    

#Testing Features Score Function
@given(label = label_image_strategy(), pos_mask = white_image_strategy(), neg_mask = white_image_strategy() )
@settings(max_examples = 20, deadline = None)
def test_feature_scoring (label, pos_mask, neg_mask):
    
    masks_list = [pos_mask, neg_mask]
    
    score, relabelled = feature_scoring(label, masks_list)
    
    maximum_filter = itk.MinimumMaximumImageCalculator[type(relabelled)].New()
    maximum_filter.SetImage(relabelled)
    maximum_filter.ComputeMaximum()
    
    adjusted_score = np.delete(score, 1, 1)
    
    assert len(score) == maximum_filter.GetMaximum()
    assert np.all( adjusted_score == [[1, 1, 1]] * len(score) )
    
    
    
    
    #Testing Features Score Function
@given(label = label_image_strategy(), pos_mask = black_image_strategy(), neg_mask = black_image_strategy() )
@settings(max_examples = 20, deadline = None)
def test_feature_scoring2 (label, pos_mask, neg_mask):
    
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
@given(label = label_image_strategy(), gnd_truth = black_image_strategy() )
@settings(max_examples = 20, deadline = None)
def test_find_simple_truth_all_false (label, gnd_truth):
        
     
    relabelled = find_connected_regions(label)
    maximum_filter = itk.MinimumMaximumImageCalculator[type(relabelled)].New()
    maximum_filter.SetImage(relabelled)
    maximum_filter.ComputeMaximum()
    
    value_array = find_simple_truth_value(relabelled, gnd_truth)
    
    
    assert len(value_array) == maximum_filter.GetMaximum()
    assert np.all( value_array == [False] * len(value_array) )
    

#Testing the FINDING TRUTH Function
@given(label = label_image_strategy(), gnd_truth = white_image_strategy() )
@settings(max_examples = 20, deadline = None)
def test_find_simple_truth_all_true (label, gnd_truth):
     
    relabelled = find_connected_regions(label)
    maximum_filter = itk.MinimumMaximumImageCalculator[type(relabelled)].New()
    maximum_filter.SetImage(relabelled)
    maximum_filter.ComputeMaximum()
    
    value_array = find_simple_truth_value(relabelled, gnd_truth)
    
    
    assert len(value_array) == maximum_filter.GetMaximum()
    assert np.all( value_array == [True] * len(value_array) )
    
    
    
    
    
#Testing the FINDING TRUTH Function
@given(label = label_image_strategy(), gnd_truth = black_image_strategy() )
@settings(max_examples = 20, deadline = None)
def test_find_Jaccard_truth_all_false(label, gnd_truth):
    
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
@settings(max_examples = 20, deadline = None)
def test_find_Jaccard_truth_all_true(label):
        
     
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
@settings(max_examples = 20, deadline = None)
def test_label_killer_all_dead(label):
    
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
@settings(max_examples = 20, deadline = None)
def test_label_killer_all_survived(label):
    
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