# MEDPL
This is a code demo for the paper "Momentum-Enhanced Dual-Prototype Learning Framework for Robust Few-Shot Hyperspectral Image Classification".

## Requirements

- CUDA = 11.3

- python = 3.8 

- torch = 1.12.1+cu113 

- numpy = 1.26.0

## Datasets

- source domain dataset
  - Chikusei

- target domain datasets
  - Indian Pines
  - Salinas
  - University of UP
  - Houston
  
  

You can download the source and target datasets mentioned above at  https://pan.baidu.com/s/1erafsAzPYCdomqBziJpy-w?pwd=ddf1, and move to folder `datasets`.  In particular, for the source dataset Chikusei, you can choose to download it in mat format, and then use the utils/chikusei_imdb_128.py file to process it to get the patch size you want, or directly use the preprocessed source dataset Chikusei_imdb_128_7_7.pickle with a patch size of 7 $\times$ 7. 

An example datasets folder has the following structure:

```
datasets
├── Chikusei_imdb_128_7_7.pickle
├── Chikusei_raw_mat
│   ├── HyperspecVNIR_Chikusei_20140729.mat
│   └── HyperspecVNIR_Chikusei_20140729_Ground_Truth.mat
├── IP
│   ├── indian_pines_corrected.mat
│   └── indian_pines_gt.mat
├── Houston
│   ├── Houston.mat
│   └── Houston_gt.mat
├── salinas
│   ├── salinas_corrected.mat
│   └── salinas_gt.mat
└── paviaU
    ├── paviaU_gt.mat
    └── paviaU.mat
```
