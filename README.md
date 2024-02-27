[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
# Singularity Strength Recalibration of Fully Convolutional Neural Networks for Biomedical Imaging
Singularity Stregth Recalibration (SSR) is a lightweight, plug-in module for CNNs that leverages multifractal analysis in an end-to-end fashion for semantic segmentation.  

This repository provides:
  - Source code for the TensorFlow 2.14 implementation of Squeeze-Excite and SSR modules
  - Source code for a vanilla U-Net and a U-Net leveraging SSR
  - Experiments using 10-fold cross-validation for the ISIC-2018 and KvasirSeg semantic segmentation datasets. 


### Recommended requirements
1. Use anaconda/miniconda to create a __python 3.10.12__ virtual environment:
    ```zsh
    $ conda create --name env_name python=3.10.12
    ```
2. Activate environment and update pip:
    ```zsh
    $ (env_name) python3 -m pip install --upgrade pip
    ```
4. Use pip to install packages in `requirements.txt` file:
    ```zsh
    $ (env_name) pip install -r /path/to/project/requirements.txt
    ```
Note that this code was developed for TensorFlow 2.14.1.

### Datasets
#### ISIC-2018
We use the following [code](https://github.com/NITR098/Awesome-U-Net/blob/main/datasets/prepare_isic.ipynb) from [1] to generate the numpy files: `X_tr_224x224.npy` and `Y_tr_224x224.npy`. Their file path should be specificated in the ISIC-18 experiments. 
#### Kvasir-Seg
The directory structure is expected for Kvasir-Seg. The dataset can be downloaded [here](https://datasets.simula.no/kvasir-seg/): 

  ```bash
   .
   ├── Kvasir-SEG
   │   ├── images
   │   │    └── *.jpg
   │   └── masks
   │        └── *.jpg
   └──
   ```

### References
[1] Azad, Reza, et al. "Medical image segmentation review: The success of u-net." arXiv preprint arXiv:2211.14830 (2022).
