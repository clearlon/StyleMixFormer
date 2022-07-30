# StyleFormer for Instagram Filter Removal

# Description
The official implementation of the User clearlon On the [AIM 2022 Instagram Filter Removal Challenge](https://codalab.lisn.upsaclay.fr/competitions/5081#results). We propose a method for removing Instagram filters from the images by assuming the affects of filters as the style information.

# Installation
```
python setup.py develop
```

# Dataset
[IFFI dataset](https://codalab.lisn.upsaclay.fr/competitions/5081#participate) contains high-resolution (1080×1080) 600 images and with 16 different filtered versions for each. In particular, we have picked mostly-used 16 filters: 1977, Amaro, Brannan, Clarendon, Gingham, He-Fe, Hudson, Lo-Fi, Mayfair, Nashville, Perpetua, Sutro, Toaster, Valencia, ~~Willow~~, X-Pro II. 

# Training
## Single GPU Training
```
CUDA_VISIBLE_DEVICES=0 \
python basicsr/train.py -opt options/train_StyleFormer.yml
```
## Distributed Training
### Two GPU
```
CUDA_VISIBLE_DEVICES=0,1 \
python -m torch.distributed.launch --nproc_per_node=2 --master_port=4321 basicsr/train.py -opt options/train_StyleFormer.yml --launcher pytorch
```

# Evaluation
```
CUDA_VISIBLE_DEVICES=0 python inference/iffi_submit_generate.py
```

# Acknowledgement
This repository is built on [BasicSR](https://github.com/XPixelGroup/BasicSR). Our work is also inspired by [ArcFace](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch), [GFPGAN](https://github.com/TencentARC/GFPGAN), and [StyleGAN2](https://github.com/NVlabs/stylegan2-ada-pytorch).