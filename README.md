[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
# Singularity Strength Recalibration of Fully Convolutional Neural Networks for Biomedical Imaging
Singularity Strength Recalibration of Fully Convolutional Neural Networks for Biomedical Imaging

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
