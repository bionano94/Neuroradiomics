# Neuroradiomics

This project is intended to be a bunch of python's functions useful for the brain tissue segmentation of a brain MRI scan.

| **Authors**  | **Project** |  **Build Status** | **License** |
|:------------:|:-----------:|:-----------------:|:-----------:|
|[**N. Biondini**](https://github.com/bionano94) <br/> [**R.Biondi**](https://github.com/RiccardoBiondi)| **Neuroradiomics** | [![Ubuntu CI](https://github.com/bionano94/Neuroradiomics/action/workflows/Neuroradiomics_python_CI.yml/badge.svg?branch=master)](https://github.com/bionano94/Neuroradiomics/action/workflows/Neuroradiomics_python_CI.yml) | ![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg) |


## Table of Contents
  1. [Prerequisites](#Prerequisites)
  2. [Installation](#Installation)
  3. [References](#References)

## Prerequisites

Supported python versions: ![Python version](https://img.shields.io/badge/python-3.6.*|3.7.*|3.8.*|3.9.*|3.10.*|3.11.*-blue.svg)

Supported [ITK](https://itk.org/) versions: 5.1.0 or above

Supported [ITK-elastix version](https://github.com/InsightSoftwareConsortium/ITKElastix): 0.13 or above

Supported [Scikit Learn](https://scikit-learn.org/stable/) versions: 1.1.2 or above


## Installation

Clone the repository:

```console
git clone https://github.com/bionano94/Neuroradiomics
cd Neuroradiomics
```

Using 'pip' install the required packages:

```console
pip install --upgrade pip
python -m pip install -r requirements.txt
```

Now you're ready to build the package:

```console
python setup.py develop --user
```

## Usage
The repository main functions are divided in 3 modules. In each one is possible to find useful functions to the related applications.

In the drectory "Examples" can be found a jupyter notebook file in which are demonstrated the main functions here collected.

Next a brief explaining of the main modules.

### Registration
In registration module there are some useful functions that applies elastix[1] and transformix filters.
It aims to rapidly and automaticly apply a predetrmined Rigid transformation (suited for co-registering two scans of the same patient taken with different modalities) or to apply a Multimap (Rigid -> Affine -> BSpline) transformatione (suited for registering an Atlas over a patient scan).

There are also functions that permits to easily apply write on file the transformations applier or to modify them.

### Skull Stripping
The skulls_stripping module contains functions that permits the brain extraction from a head MRI scan.
There are two main functions, both requires the usage of an atlas with itk brain mask to be used.

1. skull_stripping_mask that permits to obtain a brain mask for the head image as well as the transformation prameters used to register the atlas over the stripped image;
2. skull_stripper that returns only the extracted brain

### Segmentation
In segmentation module there are funtions that permits an automatic segmentation of white mater, grey matter, cerebrospinal fluid and background from a brain image.
The main segmentations functions necessities for a brain already extracted as well as the brain mask.
To be used they necessity of an atlas with probability maps for the tissues.

## References

<a id="1">[1]</a>
S. Klein, M. Staring, K. Murphy, M.A. Viergever, J.P.W. Pluim, "elastix: a toolbox for intensity based medical image registration," IEEE Transactions on Medical Imaging, vol. 29, no. 1, pp. 196 - 205, January 2010.

<a id="2">[2]</a>
D.P. Shamonin, E.E. Bron, B.P.F. Lelieveldt, M. Smits, S. Klein and M. Staring, "Fast Parallel Image Registration on CPU and GPU for Diagnostic Classification of Alzheimer's Disease", Frontiers in Neuroinformatics, vol. 7, no. 50, pp. 1-15, January 2014.
