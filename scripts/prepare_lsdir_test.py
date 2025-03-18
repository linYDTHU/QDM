#!/usr/bin/env python
# -*- coding:utf-8 -*-

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))

import argparse
from omegaconf import OmegaConf

from dataset import LSDIRFFHQDataset
from utils import util_image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "-o",
            "--outdir",
            type=str,
            default="/data/ydlin718/lsdir_test",
            help="Folder to save the processed image pairs",
            )
    args = parser.parse_args()

    gt_dir = Path(args.outdir) / 'gt'
    if not gt_dir.exists():
        gt_dir.mkdir(parents=True)
    lq_dir = Path(args.outdir) / 'lq'
    if not lq_dir.exists():
        lq_dir.mkdir(parents=True)

    num_imgs = 3000
    configs = OmegaConf.load('./configs/degradation/degradation_testing_realesrgan_vector.yaml')
    opts, opts_degradation = configs.opts, configs.degradation
    dataset = LSDIRFFHQDataset(opts)
    num_imgs = min(num_imgs, len(dataset))
    for ii in range(num_imgs):
        data_dict1 = dataset.__getitem__(ii)
        if (ii + 1) % 10 == 0:
            print(f'Processing: {ii+1}/{num_imgs}')
        data_dict2 = dataset.degrade_fun(
                opts_degradation,
                im_gt=data_dict1['gt'].unsqueeze(0),
                kernel1=data_dict1['kernel1'],
                kernel2=data_dict1['kernel2'],
                sinc_kernel=data_dict1['sinc_kernel'],
                )
        im_lq, im_gt = data_dict2['lq'], data_dict2['gt']
        im_lq, im_gt = util_image.tensor2img([im_lq, im_gt], rgb2bgr=True, min_max=(0,1) ) # uint8

        im_name = data_dict1['img_name']
        im_path_gt = gt_dir / f'{im_name}.png'
        util_image.imwrite(im_gt, im_path_gt, chn='bgr', dtype_in='uint8')

        im_path_lq = lq_dir / f'{im_name}.png'
        util_image.imwrite(im_lq, im_path_lq, chn='bgr', dtype_in='uint8')

if __name__ == "__main__":
    main()
