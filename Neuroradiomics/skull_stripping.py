import itk
import numpy

def skull_stripper(image, atlas, mask):
    """
    
    """
    InputType = itk.Image[itk.F, 3]
    OutputType = itk.Image[itk.SS, 3]
    cast_filter = itk.CastImageFilter[InputType, OutputType].New()
    
    cast_filter.SetInput(image)
    cast_image = cast_filter.GetOutput()
    
    cast_filter.SetInput(atlas)
    cast_atlas = cast_filter.GetOutput()
    
    
    OutputType = itk.Image[itk.UC, 3]
    cast_filter = itk.CastImageFilter[InputType, OutputType].New()
    
    cast_filter.SetInput(mask)
    cast_mask = cast_filter.GetOutput()
    
    strip_filter = itk.StripTsImageFilter.New()
    strip_filter.SetInput(cast_image)
    strip_filter.SetAtlasImage(cast_atlas)
    strip_filter.SetAtlasBrainMask(cast_mask)
    strip_filter.UpdateLargestPossibleRegion()
    
    print('Strip Filter Updated')
    
    mask_filter = itk.MaskImageFilter.New()    
    mask_filter.SetInput1(image)
    mask_filter.SetInput2(strip_filter.GetOutput())
    mask_filter.Update()
    
    print('Stripping Ready')
    
    brain = mask_filter.GetOutput()
    
    print('Stripping Done')
    
    return brain