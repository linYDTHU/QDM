import os
import math
import torch
import random
import argparse
import numpy as np
from multiprocessing import Pool, cpu_count
from PIL import Image
from PIL.Image import BILINEAR

import torch.nn.functional as F
from torchvision import transforms

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from basicsr.utils import DiffJPEG
from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
from basicsr.data.transforms import augment
from basicsr.utils.img_process_util import filter2D
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt

device = torch.device('cpu')

opt = {
    'blur_kernel_size':13,
    'kernel_list': ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso'],
    'kernel_prob': [0.60, 0.40, 0.0, 0.0, 0.0, 0.0], 
    'blur_sigma': [0.2, 0.8],
    'betag_range': [1.0, 1.5],
    'betap_range': [1, 1.2],
    'sinc_prob': 0.1,
    'blur_kernel_size': 12,
    'resize_prob': [0.2, 0.7, 0.1],
    'resize_range': [0.5, 1.5],
    'gray_noise_prob': 0.4,
    'gaussian_noise_prob': 0.5,
    'noise_range': [1, 15],
    'poisson_scale_range': [0.05, 0.3],
    'jpeg_range': [70, 95],
}

sf = 4

# blur settings for the first degradation
blur_kernel_size = opt['blur_kernel_size']
kernel_list = opt['kernel_list']
kernel_prob = opt['kernel_prob']  # a list for each kernel probability
blur_sigma = opt['blur_sigma']
betag_range = opt['betag_range']  # betag used in generalized Gaussian blur kernels
betap_range = opt['betap_range']  # betap used in plateau blur kernels
sinc_prob = opt['sinc_prob']  # the probability for sinc filters

kernel_range1 = [x for x in range(3, opt['blur_kernel_size'], 2)]  # kernel size ranges from 7 to 21

# ------------------------ Generate kernels (used in the first degradation) ------------------------ #
kernel_size = random.choice(kernel_range1)
if np.random.uniform() < sinc_prob:
    # this sinc filter setting is for kernels ranging from [7, 21]
    if kernel_size < 13:
        omega_c = np.random.uniform(np.pi / 3, np.pi)
    else:
        omega_c = np.random.uniform(np.pi / 5, np.pi)
    kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
else:
    kernel = random_mixed_kernels(
        kernel_list,
        kernel_prob,
        kernel_size,
        blur_sigma,
        blur_sigma, [-math.pi, math.pi],
        betag_range,
        betap_range,
        noise_range=None)
# pad kernel
pad_size = (blur_kernel_size - kernel_size) // 2
kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

kernel = torch.FloatTensor(kernel).to(device)

def degrade_fun(im_gt):
        assert im_gt.device == kernel.device, 'device mismatch'
        jpeger = DiffJPEG(differentiable=False).to(im_gt.device)  # simulate JPEG compression artifacts

        ori_h, ori_w = im_gt.size()[2:4]

        # ----------------------- The first degradation process ----------------------- #
        # blur
        out = filter2D(im_gt, kernel)
        # random resize
        updown_type = random.choices(
                ['up', 'down', 'keep'],
                opt['resize_prob'],
                )[0]
        if updown_type == 'up':
            scale = random.uniform(1, opt['resize_range'][1])
        elif updown_type == 'down':
            scale = random.uniform(opt['resize_range'][0], 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(out, scale_factor=scale, mode=mode)
        # add noise
        gray_noise_prob = opt['gray_noise_prob']
        if random.random() < opt['gaussian_noise_prob']:
            out = random_add_gaussian_noise_pt(
                out,
                sigma_range=opt['noise_range'],
                clip=True,
                rounds=False,
                gray_prob=gray_noise_prob,
                )
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=opt['poisson_scale_range'],
                gray_prob=gray_noise_prob,
                clip=True,
                rounds=False)
        # JPEG compression
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*opt['jpeg_range'])
        out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
        out = jpeger(out, quality=jpeg_p)

        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(
                out,
                size=(ori_h // sf, ori_w // sf),
                mode=mode,
                )

        # clamp and round
        im_lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.
        
        return im_lq.contiguous()

def process_image(args):
    """parallel process image"""
    input_path, output_dir, scale_gt, scale_lq, degradation = args
    try:
        if degradation == "bicubic":
            with Image.open(input_path) as img:
                base_name = os.path.basename(input_path)

                # generate LQ
                lq_size = (int(img.width*scale_lq), int(img.height*scale_lq))
                lq_img = img.resize(lq_size, BILINEAR)

                # generate GT
                gt_size = (int(img.width*scale_gt), int(img.height*scale_gt)) 
                gt_img = img.resize(gt_size, BILINEAR)

                # save images
                lq_path = os.path.join(output_dir, 'lq', base_name)
                gt_path = os.path.join(output_dir, 'gt', base_name)

                # optimze for different formats
                if base_name.lower().endswith('.png'):
                    lq_img.save(lq_path, optimize=True, compress_level=3)
                    gt_img.save(gt_path, optimize=True, compress_level=3)
                else:  # JPEG and other formats
                    lq_img.save(lq_path, quality=85, optimize=True, progressive=True)
                    gt_img.save(gt_path, quality=90, optimize=True, progressive=True)

                return True
        elif degradation == "degrade_fun":
            with Image.open(input_path) as img:
                img = img.convert('RGB')
                base_name = os.path.basename(input_path)

                # generate GT
                gt_size = (int(img.width*scale_gt), int(img.height*scale_gt)) 
                gt_img = img.resize(gt_size, BILINEAR)

                # degrad image
                transform = transforms.Compose([transforms.ToTensor()])
                gt_tensor = transform(gt_img).unsqueeze(0)
                # print(gt_tensor.shape)
                lq_tensor = degrade_fun(gt_tensor)
                # print(lq_tensor.shape)
                transform = transforms.ToPILImage()
                lq_img = transform(lq_tensor.squeeze(0))

                # save images
                lq_path = os.path.join(output_dir, 'lq', base_name)
                gt_path = os.path.join(output_dir, 'gt', base_name)

                # optimze for different formats
                if base_name.lower().endswith('.png'):
                    lq_img.save(lq_path, optimize=True, compress_level=3)
                    gt_img.save(gt_path, optimize=True, compress_level=3)
                else:  # JPEG and other formats
                    lq_img.save(lq_path, quality=85, optimize=True, progressive=True)
                    gt_img.save(gt_path, quality=90, optimize=True, progressive=True)
            return True
        else:
            print(f"Unsupported degradation: {degradation}")
            return False
    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='parallel prepare med dataset')
    parser.add_argument('-i', '--input_dir', required=True, help='input directory path')
    parser.add_argument('-o', '--output_dir', required=True, help='output directory path')
    parser.add_argument('-j', '--workers', type=int, default=cpu_count()//2, 
                      help=f'parallel jobs (default: {cpu_count()//2})')
    parser.add_argument('-d', '--degradation', type=str, default='bicubic',
                      help='degradation method: bicubic or degrade_fun (default: bicubic)')
    args = parser.parse_args()

    # create output dirs
    os.makedirs(os.path.join(args.output_dir, 'lq'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'gt'), exist_ok=True)

    # generate task args
    task_args = []
    valid_ext = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    for filename in os.listdir(args.input_dir):
        if filename.lower().endswith(valid_ext):
            input_path = os.path.join(args.input_dir, filename)
            task_args.append((
                input_path,
                args.output_dir,
                0.5,   # scale_gt
                0.125,  # scale_lq
                args.degradation
            ))

    # use process pool to process images
    with Pool(processes=args.workers) as pool:
        results = pool.map(process_image, task_args)
        
    success_count = sum(results)
    print(f"Done!Successfully deal with {success_count}/{len(task_args)} images")

if __name__ == '__main__':
    main()