# Neuroradiomics

A pipeline for the skull stripping and the segmentation of the MRI of a human head.

| **Authors**  | **Project** |  **Build Status** | **License** |
|:------------:|:-----------:|:-----------------:|:-----------:|
|[**N. Biondini**](https://github.com/bionano94) <br/> [**R.Biondi**](https://github.com/RiccardoBiondi)| **Neuroradiomics** | [![Ubuntu CI](https://github.com/bionano94/Neuroradiomics/workflows/Neuroradiomics%CI.yml/badge.svg)] | ![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg) |


## Table of Contents
  1. [Prerequisites](#Prerequisites)
  2. [Installation](#Installation)
  3. [References](#References)

## Prerequisites

Supported python versions: ![Python version](https://img.shields.io/badge/python-3.6.*|3.7.*|3.8.*|3.9.*-blue.svg)

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
python -m pip install -r requirements.txt
```

Now you're ready to build the package:

```console
python setup.py develop --user
```

## References

<a id="1">[1]</a>

S. Klein, M. Staring, K. Murphy, M.A. Viergever, J.P.W. Pluim, "elastix: a toolbox for intensity based medical image registration," IEEE Transactions on Medical Imaging, vol. 29, no. 1, pp. 196 - 205, January 2010.

<a id="2">[2]</a>

D.P. Shamonin, E.E. Bron, B.P.F. Lelieveldt, M. Smits, S. Klein and M. Staring, "Fast Parallel Image Registration on CPU and GPU for Diagnostic Classification of Alzheimer's Disease", Frontiers in Neuroinformatics, vol. 7, no. 50, pp. 1-15, January 2014.
