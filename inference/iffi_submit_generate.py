import os
import numpy as np
import cv2
import torch
import argparse

import sys
sys.path.append('/data/disk2/longshaoyi/project/StyleFormer/')

from styleformer.archs.styleformer_arch import StyleFormerNet
from styleformer.utils.img_util import tensor2img


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda')

def main(args):
    # make dir
    os.makedirs(args.output_path, exist_ok=True)

    img_folders = sorted(os.listdir(args.input_path))
    for index in img_folders:
        output_path_index = os.path.join(args.output_path, index)
        os.makedirs(output_path_index) # mkdir 

        img_folder = os.path.join(args.input_path, index)
        img_fnames = sorted(os.listdir(img_folder))
        for img_fname in img_fnames:
            filtered_img = cv2.imread(os.path.join(img_folder, img_fname), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
            filtered_img = torch.from_numpy(filtered_img.transpose(2, 0, 1)).float()
            filtered_img = filtered_img.unsqueeze(0).to(device)
            
            ## init model
            model = StyleFormerNet(
                enc_blk_nums=[1, 1, 7, 7],
                middle_blk_num=2,
                load_path='experiments/pretrained/resnetarcface.pth'
            )
            # print(model)
            model.load_state_dict(torch.load(args.model_path)['params'], strict=True)
            model.eval()
            model = model.to(device)
            for m in model.modules():
                if hasattr(m, 'switch_to_deploy'):
                    m.switch_to_deploy()
            unfiltered_img = model(filtered_img)
            # save .png
            unfiltered_img = tensor2img(unfiltered_img, rgb2bgr=False, out_type=np.uint8, data_range=255)
            # jpg -> png
            img_fname = img_fname.split('.')[0]
            cv2.imwrite(os.path.join(output_path_index, f'{img_fname}.png'), unfiltered_img)
        
        print(index)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="experiments/StyleFormerNet/net_g_latest.pth")
    parser.add_argument('--input_path', type=str, default="datasets/IFFI/IFFI-dataset-lr-test")
    parser.add_argument('--output_path', type=str, default="results/IFFI/StyleFormerNet-581000")

    args = parser.parse_args()
    main(args)
