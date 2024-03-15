## *Semantics-Augmented Quantization-Aware Training for Point Cloud Classification*

### Introduction

This project is the official implementation of our paper *Semantics-Augmented Quantization-Aware Training for Point Cloud Classification*. 
### Installation
The latest codes are tested on Ubuntu 22.04, CUDA11.7, PyTorch 1.13.1 and Python 3.8:
```shell script
# create new conda environment
conda create -n saqat python=3.8 -y
conda activate saqat

# install pytorch
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia

# install pytorch3d
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install pytorch3d -c pytorch3d

# install other dependencies
pip install brevitas
pip install h5py
pip install opencv-python
```
### Dataset
##### ModelNet40
Note in this experiment, we do not use any re-sampled version of ModelNet40 (more than 2K points) or any normal information.  The dataset utilized here is: `modelnet40_ply_hdf5_2048`. ModelNet40 dataset will be downloaded automatically.

##### ScanObjectNN
The hardest variant is available for download via the provided link. Please cite their paper if you use the link to download the data.
```bash
mkdir data
cd data
gdown https://drive.google.com/uc?id=1iM3mhMJ_N0x5pytcP831l3ZFwbLmbwzi
tar -xvf ScanObjectNN.tar
```
Organize the ScanObjectNN dataset as follows:
```
data
 |--- h5_files
        |--- main_split
                |--- training_objectdataset_augmentedrot_scale75.h5
                |--- test_objectdataset_augmentedrot_scale75.h5
```
### Train
```shell script
conda activate saqat
python train_classification.py --gpu 0 --model pointnet_quant_cls --dataset scanobjectnn --log_dir SONN_cls --quant_bit 8 --knn_k 30 --max_probability 0.1 --loss2_weight 1 --learning_rate 0.0005
```

### Test
```shell script
conda activate saqat
python test_classification.py --gpu 1 --model pointnet_quant_cls --dataset scanobjectnn --log_dir SONN_cls
```

