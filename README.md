# StyleMixFormer for Instagram Filter Removal

# Description
The official implementation of the User clearlon On the [AIM 2022 Instagram Filter Removal Challenge](https://codalab.lisn.upsaclay.fr/competitions/5081#results). We propose a method for removing Instagram filters from the images by assuming the affects of filters as the style information.

# Installation
Clone our repository
```
git clone https://github.com/clearlon/StyleFormer.git

cd StyleFormer 
```

To install requirements:
```
pip install -r requirements.txt
```

Install
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
Download pretrained model from [Google Drive](https://drive.google.com/drive/folders/15ip14nh7vd1v6qxf_CU-axGQmdGDlaT5?usp=sharing) and put it in the `experiments/pretrained` path.
```
CUDA_VISIBLE_DEVICES=0 python inference/iffi_submit_generate.py
```

# Citation

# Acknowledgement
This repository is built on [BasicSR](https://github.com/XPixelGroup/BasicSR). Our work is also inspired by [ArcFace](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch), [Restormer](https://github.com/swz30/Restormer), [NAFNet](https://github.com/megvii-research/NAFNet), [MixFormer](https://arxiv.org/pdf/2204.02557.pdf), and [StyleGAN2](https://github.com/NVlabs/stylegan2-ada-pytorch).
