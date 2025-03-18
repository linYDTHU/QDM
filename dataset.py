import os
import math
import random
from PIL import Image
from pathlib import Path
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import torchvision.transforms as transforms
import torch.nn.functional as F

from basicsr.utils import DiffJPEG
from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
from basicsr.data.transforms import augment
from basicsr.utils.img_process_util import filter2D
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt

import utils.util_image as util_image
import utils.util_common as util_common

def create_dataset(dataset_config):
    if dataset_config['type'] == 'lsdirffhq':
        dataset = LSDIRFFHQDataset(dataset_config['params'])
    elif dataset_config['type'] == 'base':
        dataset = BaseData(**dataset_config['params'])
    elif dataset_config['type'] == 'combined':
        dataset = CombinedDataset(dataset_config['params'])
    elif dataset_config['type'] == 'medsrgan':
        dataset = DegradedMedDataset(dataset_config['params'])
    else:
        raise NotImplementedError(dataset_config['type'])

    return dataset

class BaseData(Dataset):
    def __init__(
            self,
            dir_path,
            txt_path=None,
            extra_dir_path=None,
            length=None,
            need_path=False,
            im_exts=['png', 'jpg', 'jpeg', 'JPEG', 'bmp'],
            chn='rgb',
            recursive=False,
            ):
        super().__init__()

        file_paths_all = []
        if dir_path is not None:
            file_paths_all.extend(util_common.scan_files_from_folder(dir_path, im_exts, recursive))
        if txt_path is not None:
            file_paths_all.extend(util_common.readline_txt(txt_path))

        self.file_paths = file_paths_all if length is None else random.sample(file_paths_all, length)
        self.file_paths_all = file_paths_all

        self.length = length
        self.need_path = need_path

        # Transformation
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5)
            ])
        self.chn = chn

        self.extra_dir_path = extra_dir_path
        if extra_dir_path is not None:
            self.extra_transform = self.transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        im_path_base = self.file_paths[index]
        im_base = util_image.imread(im_path_base, chn=self.chn, dtype='float32')

        im_target = self.transform(im_base)
        out = {'lq':im_target}

        if self.extra_dir_path is not None:
            im_path_extra = Path(self.extra_dir_path) / Path(im_path_base).name
            im_extra = util_image.imread(im_path_extra, chn=self.chn, dtype='float32')
            im_extra = self.extra_transform(im_extra)
            out['gt'] = im_extra

        if self.need_path:
            out['path'] = im_path_base

        return out

    def reset_dataset(self):
        self.file_paths = random.sample(self.file_paths_all, self.length)

class LSDIRFFHQDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        # Load the Hugging Face dataset
        self.split = opt["split"]
        self.hf_dataset = load_dataset("danjacobellis/LSDIR", cache_dir=opt["local_lsdir_cache_dir"], split=self.split)
        
        # Set up local FFHQ paths with a limit (first 10,000 images)
        self.include_ffhq = opt["include_ffhq"]
        if self.include_ffhq:
            self.local_ffhq_dir = opt["local_ffhq_dir"]
            self.limit_ffhq = opt["limit_ffhq"]
            self.ffhq_image_paths = [os.path.join(self.local_ffhq_dir, fname) 
                                     for fname in sorted(os.listdir(self.local_ffhq_dir))[:self.limit_ffhq]]       
            # Combined length of both datasets
            self.length = len(self.hf_dataset) + len(self.ffhq_image_paths)
        else:
            self.length = len(self.hf_dataset)
        
        # Transformation
        if opt["transform_type"]=="crop512":
            self.transform = transforms.Compose([
                RandomCropOrResizeCrop(size=512, use_center_crop=opt["center_crop"]),
                transforms.ToTensor()
            ])
        elif opt["transform_type"]=="crop256":
            self.transform = transforms.Compose([
                RandomCropWithResize(256),
                transforms.ToTensor()
            ])
        else:
            raise NotImplementedError(f"Transform type {opt['transform_type']} not implemented")

        # blur settings for the first degradation
        self.blur_kernel_size = opt['blur_kernel_size']
        self.kernel_list = opt['kernel_list']
        self.kernel_prob = opt['kernel_prob']  # a list for each kernel probability
        self.blur_sigma = opt['blur_sigma']
        self.betag_range = opt['betag_range']  # betag used in generalized Gaussian blur kernels
        self.betap_range = opt['betap_range']  # betap used in plateau blur kernels
        self.sinc_prob = opt['sinc_prob']  # the probability for sinc filters

        # blur settings for the second degradation
        self.blur_kernel_size2 = opt['blur_kernel_size2']
        self.kernel_list2 = opt['kernel_list2']
        self.kernel_prob2 = opt['kernel_prob2']
        self.blur_sigma2 = opt['blur_sigma2']
        self.betag_range2 = opt['betag_range2']
        self.betap_range2 = opt['betap_range2']
        self.sinc_prob2 = opt['sinc_prob2']

        # a final sinc filter
        self.final_sinc_prob = opt['final_sinc_prob']

        self.kernel_range1 = [x for x in range(3, opt['blur_kernel_size'], 2)]  # kernel size ranges from 7 to 21
        self.kernel_range2 = [x for x in range(3, opt['blur_kernel_size2'], 2)]  # kernel size ranges from 7 to 21
        # TODO: kernel range is now hard-coded, should be in the configure file
        # convolving with pulse tensor brings no blurry effect
        self.pulse_tensor = torch.zeros(opt['blur_kernel_size2'], opt['blur_kernel_size2']).float()
        self.pulse_tensor[opt['blur_kernel_size2']//2, opt['blur_kernel_size2']//2] = 1

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Determine whether to load from the Hugging Face dataset or FFHQ local directory
        if idx < len(self.hf_dataset):
            # Load from Hugging Face dataset
            img_gt = self.hf_dataset[idx]["image"]
            img_name = Path(self.hf_dataset[idx]["path"]).stem
            img_gt = np.array(img_gt)
        else:
            # Load from FFHQ local images
            ffhq_idx = idx - len(self.hf_dataset)
            img_path = self.ffhq_image_paths[ffhq_idx]
            img_name = Path(img_path).stem
            img_gt = Image.open(img_path).convert("RGB")
            img_gt = np.array(img_gt)

        # -------------------- Do augmentation for training: flip, rotation -------------------- #
        if self.split == 'train':
            img_gt = augment(img_gt, self.opt['use_hflip'], self.opt['use_rot'])

        # ------------------------ Generate kernels (used in the first degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range1)
        if np.random.uniform() < self.opt['sinc_prob']:
            # this sinc filter setting is for kernels ranging from [7, 21]
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel = random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                kernel_size,
                self.blur_sigma,
                self.blur_sigma, [-math.pi, math.pi],
                self.betag_range,
                self.betap_range,
                noise_range=None)
        # pad kernel
        pad_size = (self.blur_kernel_size - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range2)
        if np.random.uniform() < self.opt['sinc_prob2']:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2 = random_mixed_kernels(
                self.kernel_list2,
                self.kernel_prob2,
                kernel_size,
                self.blur_sigma2,
                self.blur_sigma2, [-math.pi, math.pi],
                self.betag_range2,
                self.betap_range2,
                noise_range=None)

        # pad kernel
        pad_size = (self.blur_kernel_size2 - kernel_size) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------------------- the final sinc kernel ------------------------------------- #
        if np.random.uniform() < self.opt['final_sinc_prob']:
            kernel_size = random.choice(self.kernel_range2)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=self.blur_kernel_size2)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            sinc_kernel = self.pulse_tensor

        # numpy to tensor
        kernel = torch.FloatTensor(kernel)
        kernel2 = torch.FloatTensor(kernel2)

        # Apply the transformation if specified
        if self.transform:
            img_gt = Image.fromarray(img_gt).convert("RGB")
            img_gt = self.transform(img_gt)

        return_d = {'gt': img_gt, 'kernel1': kernel, 'kernel2': kernel2, 'sinc_kernel': sinc_kernel, 'img_name': img_name}
        return return_d

    def degrade_fun(self, conf_degradation, im_gt, kernel1, kernel2, sinc_kernel):
        assert im_gt.device == kernel1.device == kernel2.device == sinc_kernel.device, 'device mismatch'
        if not hasattr(self, 'jpeger'):
            self.jpeger = DiffJPEG(differentiable=False).to(im_gt.device)  # simulate JPEG compression artifacts

        ori_h, ori_w = im_gt.size()[2:4]
        sf = conf_degradation.sf

        # ----------------------- The first degradation process ----------------------- #
        # blur
        out = filter2D(im_gt, kernel1)
        # random resize
        updown_type = random.choices(
                ['up', 'down', 'keep'],
                conf_degradation['resize_prob'],
                )[0]
        if updown_type == 'up':
            scale = random.uniform(1, conf_degradation['resize_range'][1])
        elif updown_type == 'down':
            scale = random.uniform(conf_degradation['resize_range'][0], 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(out, scale_factor=scale, mode=mode)
        # add noise
        gray_noise_prob = conf_degradation['gray_noise_prob']
        if random.random() < conf_degradation['gaussian_noise_prob']:
            out = random_add_gaussian_noise_pt(
                out,
                sigma_range=conf_degradation['noise_range'],
                clip=True,
                rounds=False,
                gray_prob=gray_noise_prob,
                )
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=conf_degradation['poisson_scale_range'],
                gray_prob=gray_noise_prob,
                clip=True,
                rounds=False)
        # JPEG compression
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*conf_degradation['jpeg_range'])
        out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
        out = self.jpeger(out, quality=jpeg_p)

        # ----------------------- The second degradation process ----------------------- #
        # blur
        if random.random() < conf_degradation['second_order_prob']:
            if random.random() < conf_degradation['second_blur_prob']:
                out = filter2D(out, kernel2)
            # random resize
            updown_type = random.choices(
                    ['up', 'down', 'keep'],
                    conf_degradation['resize_prob2'],
                    )[0]
            if updown_type == 'up':
                scale = random.uniform(1, conf_degradation['resize_range2'][1])
            elif updown_type == 'down':
                scale = random.uniform(conf_degradation['resize_range2'][0], 1)
            else:
                scale = 1
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(
                    out,
                    size=(int(ori_h / sf * scale), int(ori_w / sf * scale)),
                    mode=mode,
                    )
            # add noise
            gray_noise_prob = conf_degradation['gray_noise_prob2']
            if random.random() < conf_degradation['gaussian_noise_prob2']:
                out = random_add_gaussian_noise_pt(
                    out,
                    sigma_range=conf_degradation['noise_range2'],
                    clip=True,
                    rounds=False,
                    gray_prob=gray_noise_prob,
                    )
            else:
                out = random_add_poisson_noise_pt(
                    out,
                    scale_range=conf_degradation['poisson_scale_range2'],
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False,
                    )

        # JPEG compression + the final sinc filter
        # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
        # as one operation.
        # We consider two orders:
        #   1. [resize back + sinc filter] + JPEG compression
        #   2. JPEG compression + [resize back + sinc filter]
        # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
        if random.random() < 0.5:
            # resize back + the final sinc filter
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(
                    out,
                    size=(ori_h // sf, ori_w // sf),
                    mode=mode,
                    )
            out = filter2D(out, sinc_kernel)
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*conf_degradation['jpeg_range2'])
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)
        else:
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*conf_degradation['jpeg_range2'])
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)
            # resize back + the final sinc filter
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(
                    out,
                    size=(ori_h // sf, ori_w // sf),
                    mode=mode,
                    )
            out = filter2D(out, sinc_kernel)

        # clamp and round
        im_lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.

        return {'lq':im_lq.contiguous(), 'gt':im_gt}

class CombinedDataset(LSDIRFFHQDataset):
    def __init__(self, opt):
        self.opt = opt
        # Load the LSDIR dataset
        self.split = opt["split"]
        self.lsdir_dataset = load_dataset("danjacobellis/LSDIR", cache_dir=opt["local_lsdir_cache_dir"], split=self.split)
        self.lsdir_length = len(self.lsdir_dataset)
        
        # Set up FFHQ paths with a limit number of images (first 10,000 images)
        self.local_ffhq_dir = opt["local_ffhq_dir"]
        self.limit_ffhq = opt["limit_ffhq"]
        self.ffhq_image_paths = [os.path.join(self.local_ffhq_dir, fname) 
                                 for fname in sorted(os.listdir(self.local_ffhq_dir))[:self.limit_ffhq]]       
        self.ffhq_length = self.limit_ffhq

        # Set up local Flicker2K
        self.local_flicker_dir = opt["local_flicker_dir"]
        self.flicker_image_paths = [os.path.join(self.local_flicker_dir, fname) 
                                 for fname in sorted(os.listdir(self.local_flicker_dir))]
        self.flicker_length = len(self.flicker_image_paths)

        # Set up local DIV2K
        self.local_div2k_dir = opt["local_div2k_dir"]
        self.div2k_image_paths = [os.path.join(self.local_div2k_dir, fname) 
                                 for fname in sorted(os.listdir(self.local_div2k_dir))]
        self.div2k_length = len(self.div2k_image_paths)

        # Set up local DIV8K
        self.local_div8k_dir = opt["local_div8k_dir"]
        self.div8k_image_paths = [os.path.join(self.local_div8k_dir, fname) 
                                 for fname in sorted(os.listdir(self.local_div8k_dir)) if fname.lower().endswith('.png')]
        self.div8k_length = len(self.div8k_image_paths)

        # Set up local outdoorscenetrain
        self.local_outdoorscenetrain_dir = opt["local_outdoorscenetrain_dir"]
        self.outdoorscenetrain_image_paths = []
        for subdir in os.listdir(self.local_outdoorscenetrain_dir):
            for image_name in sorted(os.listdir(os.path.join(self.local_outdoorscenetrain_dir, subdir))):
                self.outdoorscenetrain_image_paths.append(os.path.join(self.local_outdoorscenetrain_dir, subdir, image_name))
        self.outdoorscenetrain_length = len(self.outdoorscenetrain_image_paths)

        # Combined length of both datasets
        self.length = self.lsdir_length + self.ffhq_length + self.flicker_length + self.div2k_length + self.div8k_length + self.outdoorscenetrain_length
        self.lengths = [self.lsdir_length, self.ffhq_length, self.flicker_length, self.div2k_length, self.div8k_length, self.outdoorscenetrain_length]
        self.cum_lengths = [sum(self.lengths[:i+1]) for i in range(len(self.lengths))]

        # Transformation
        if opt["transform_type"]=="crop512":
            self.transform = transforms.Compose([
                RandomCropOrResizeCrop(size=512, use_center_crop=opt["center_crop"]),
                transforms.ToTensor()
            ])
        elif opt["transform_type"]=="crop256":
            self.transform = transforms.Compose([
                RandomCropWithResize(256),
                transforms.ToTensor()
            ])
        else:
            raise NotImplementedError(f"Transform type {opt['transform_type']} not implemented")

        # blur settings for the first degradation
        self.blur_kernel_size = opt['blur_kernel_size']
        self.kernel_list = opt['kernel_list']
        self.kernel_prob = opt['kernel_prob']  # a list for each kernel probability
        self.blur_sigma = opt['blur_sigma']
        self.betag_range = opt['betag_range']  # betag used in generalized Gaussian blur kernels
        self.betap_range = opt['betap_range']  # betap used in plateau blur kernels
        self.sinc_prob = opt['sinc_prob']  # the probability for sinc filters

        # blur settings for the second degradation
        self.blur_kernel_size2 = opt['blur_kernel_size2']
        self.kernel_list2 = opt['kernel_list2']
        self.kernel_prob2 = opt['kernel_prob2']
        self.blur_sigma2 = opt['blur_sigma2']
        self.betag_range2 = opt['betag_range2']
        self.betap_range2 = opt['betap_range2']
        self.sinc_prob2 = opt['sinc_prob2']

        # a final sinc filter
        self.final_sinc_prob = opt['final_sinc_prob']

        self.kernel_range1 = [x for x in range(3, opt['blur_kernel_size'], 2)]  # kernel size ranges from 7 to 21
        self.kernel_range2 = [x for x in range(3, opt['blur_kernel_size2'], 2)]  # kernel size ranges from 7 to 21
        # TODO: kernel range is now hard-coded, should be in the configure file
        # convolving with pulse tensor brings no blurry effect
        self.pulse_tensor = torch.zeros(opt['blur_kernel_size2'], opt['blur_kernel_size2']).float()
        self.pulse_tensor[opt['blur_kernel_size2']//2, opt['blur_kernel_size2']//2] = 1

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # Determine to load which dataset
        if idx < self.cum_lengths[0]:
            # Load from LSDIR dataset
            img_gt = self.lsdir_dataset[idx]["image"]
            img_name = Path(self.lsdir_dataset[idx]["path"]).stem
            img_gt = np.array(img_gt)
        elif idx < self.cum_lengths[1]:
            # Load from FFHQ local images
            ffhq_idx = idx - self.cum_lengths[0]
            img_path = self.ffhq_image_paths[ffhq_idx]
            img_name = Path(img_path).stem
            img_gt = Image.open(img_path).convert("RGB")
            img_gt = np.array(img_gt)
        elif idx < self.cum_lengths[2]:
            # Load from Flicker2K local images
            flicker_idx = idx - self.cum_lengths[1]
            img_path = self.flicker_image_paths[flicker_idx]
            img_name = Path(img_path).stem
            img_gt = Image.open(img_path).convert("RGB")
            img_gt = np.array(img_gt)
        elif idx < self.cum_lengths[3]:
            # Load from DIV2K local images
            div2k_idx = idx - self.cum_lengths[2]
            img_path = self.div2k_image_paths[div2k_idx]
            img_name = Path(img_path).stem
            img_gt = Image.open(img_path).convert("RGB")
            img_gt = np.array(img_gt)
        elif idx < self.cum_lengths[4]:
            # Load from DIV8K local images
            div8k_idx = idx - self.cum_lengths[3]
            img_path = self.div8k_image_paths[div8k_idx]
            img_name = Path(img_path).stem
            img_gt = Image.open(img_path).convert("RGB")
            img_gt = np.array(img_gt)
        else:
            # Load from outdoorscenetrain local images
            outdoorscenetrain_idx = idx - self.cum_lengths[4]
            img_path = self.outdoorscenetrain_image_paths[outdoorscenetrain_idx]
            img_name = Path(img_path).stem
            img_gt = Image.open(img_path).convert("RGB")
            img_gt = np.array(img_gt)

        if img_gt is None:
            raise ValueError(f"Image could not be loaded properly for idx: {idx}, path: {img_path if 'img_path' in locals() else 'N/A'}")

        # -------------------- Do augmentation for training: flip, rotation -------------------- #
        if self.split == 'train':
            img_gt = augment(img_gt, self.opt['use_hflip'], self.opt['use_rot'])

        # ------------------------ Generate kernels (used in the first degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range1)
        if np.random.uniform() < self.opt['sinc_prob']:
            # this sinc filter setting is for kernels ranging from [7, 21]
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel = random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                kernel_size,
                self.blur_sigma,
                self.blur_sigma, [-math.pi, math.pi],
                self.betag_range,
                self.betap_range,
                noise_range=None)
        # pad kernel
        pad_size = (self.blur_kernel_size - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range2)
        if np.random.uniform() < self.opt['sinc_prob2']:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2 = random_mixed_kernels(
                self.kernel_list2,
                self.kernel_prob2,
                kernel_size,
                self.blur_sigma2,
                self.blur_sigma2, [-math.pi, math.pi],
                self.betag_range2,
                self.betap_range2,
                noise_range=None)

        # pad kernel
        pad_size = (self.blur_kernel_size2 - kernel_size) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------------------- the final sinc kernel ------------------------------------- #
        if np.random.uniform() < self.opt['final_sinc_prob']:
            kernel_size = random.choice(self.kernel_range2)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=self.blur_kernel_size2)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            sinc_kernel = self.pulse_tensor

        # numpy to tensor
        kernel = torch.FloatTensor(kernel)
        kernel2 = torch.FloatTensor(kernel2)

        # Apply the transformation if specified
        if self.transform:
            img_gt = Image.fromarray(img_gt).convert("RGB")
            img_gt = self.transform(img_gt)

        return_d = {'gt': img_gt, 'kernel1': kernel, 'kernel2': kernel2, 'sinc_kernel': sinc_kernel, 'img_name': img_name}
        return return_d

class DegradedMedDataset(Dataset):
    def __init__(self, opt,):
        self.opt = opt
        self.dir_path = opt["dir_path"]
        self.im_exts = opt["im_exts"]
        file_paths_all = []
        if self.dir_path is not None:
            file_paths_all.extend(util_common.scan_files_from_folder(self.dir_path, self.im_exts, False))

        self.file_paths_all = file_paths_all

        # Transformation
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            ])
        
        # blur settings for the first degradation
        self.blur_kernel_size = opt['blur_kernel_size']
        self.kernel_list = opt['kernel_list']
        self.kernel_prob = opt['kernel_prob']  # a list for each kernel probability
        self.blur_sigma = opt['blur_sigma']
        self.betag_range = opt['betag_range']  # betag used in generalized Gaussian blur kernels
        self.betap_range = opt['betap_range']  # betap used in plateau blur kernels
        self.sinc_prob = opt['sinc_prob']  # the probability for sinc filters

        self.kernel_range1 = [x for x in range(3, opt['blur_kernel_size'], 2)]  # kernel size ranges from 7 to 21

    def __len__(self):
        return len(self.file_paths_all)

    def __getitem__(self, idx):
        im_path = self.file_paths_all[idx]
        img_name = Path(im_path).stem
        img_gt = Image.open(im_path).convert("RGB")
        img_gt = np.array(img_gt)

        # -------------------- Do augmentation for training: flip, rotation -------------------- #
        img_gt = augment(img_gt, self.opt['use_hflip'], self.opt['use_rot'])

        # ------------------------ Generate kernels (used in the first degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range1)
        if np.random.uniform() < self.opt['sinc_prob']:
            # this sinc filter setting is for kernels ranging from [7, 21]
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel = random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                kernel_size,
                self.blur_sigma,
                self.blur_sigma, [-math.pi, math.pi],
                self.betag_range,
                self.betap_range,
                noise_range=None)
        # pad kernel
        pad_size = (self.blur_kernel_size - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        # numpy to tensor
        kernel = torch.FloatTensor(kernel)

        # Apply the transformation if specified
        if self.transform:
            img_gt = Image.fromarray(img_gt).convert("RGB")
            img_gt = self.transform(img_gt)

        return_d = {'gt': img_gt, 'kernel1': kernel, 'img_name': img_name}
        return return_d

    def degrade_fun(self, conf_degradation, im_gt, kernel1):
        assert im_gt.device == kernel1.device, 'device mismatch'
        if not hasattr(self, 'jpeger'):
            self.jpeger = DiffJPEG(differentiable=False).to(im_gt.device)  # simulate JPEG compression artifacts

        ori_h, ori_w = im_gt.size()[2:4]
        sf = conf_degradation.sf

        # ----------------------- The first degradation process ----------------------- #
        # blur
        out = filter2D(im_gt, kernel1)
        # random resize
        updown_type = random.choices(
                ['up', 'down', 'keep'],
                conf_degradation['resize_prob'],
                )[0]
        if updown_type == 'up':
            scale = random.uniform(1, conf_degradation['resize_range'][1])
        elif updown_type == 'down':
            scale = random.uniform(conf_degradation['resize_range'][0], 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(out, scale_factor=scale, mode=mode)
        # add noise
        gray_noise_prob = conf_degradation['gray_noise_prob']
        if random.random() < conf_degradation['gaussian_noise_prob']:
            out = random_add_gaussian_noise_pt(
                out,
                sigma_range=conf_degradation['noise_range'],
                clip=True,
                rounds=False,
                gray_prob=gray_noise_prob,
                )
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=conf_degradation['poisson_scale_range'],
                gray_prob=gray_noise_prob,
                clip=True,
                rounds=False)
        # JPEG compression
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*conf_degradation['jpeg_range'])
        out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
        out = self.jpeger(out, quality=jpeg_p)
        
        out = F.interpolate(
                out,
                size=(ori_h // sf, ori_w // sf),
                mode=mode,
                )

        # clamp and round
        im_lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.

        return {'lq':im_lq.contiguous(), 'gt':im_gt}


# Custom transformation class with two options: direct crop or resize + crop

class RandomCropOrResizeCrop:
    def __init__(self, size=512, use_center_crop=False):
        """
        Initializes the transform with the desired crop size.
        
        Args:
            size (int): Target size for the crop (width and height).
            use_center_crop (bool): If True, applies a center crop instead of random crop.
        """
        self.size = size
        self.use_center_crop = use_center_crop

    def __call__(self, img):
        """
        Applies a random crop, resize-then-crop, or center crop transformation to the image.
        
        If the image has any dimension smaller than `size`, it first resizes
        the image to meet the minimum dimension, then applies the specified crop.
        """
        width, height = img.size
        
        # Check if resizing is necessary
        if width < self.size or height < self.size:
            # Compute new dimensions to maintain aspect ratio
            if width < height:
                new_width = self.size
                new_height = int(height * (self.size / width))
            else:
                new_height = self.size
                new_width = int(width * (self.size / height))

            # Resize to at least `self.size` in the smaller dimension
            img = transforms.Resize((new_height, new_width))(img)

        # Choose the type of crop transformation
        if self.use_center_crop:
            # Apply center crop
            transform = transforms.CenterCrop(self.size)
        else:
            # Randomly decide between direct random crop or resize-then-crop
            if random.random() > 0.5:
                # Direct random crop to size
                transform = transforms.RandomCrop(self.size)
            else:
                # Resize then random crop to size
                resize_then_crop = transforms.Compose([
                    transforms.Resize((self.size, self.size)),
                    transforms.RandomCrop(self.size)
                ])
                transform = resize_then_crop

        return transform(img)
    

class RandomCropWithResize:
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, img):
        # Check dimensions of the image
        width, height = img.size
        
        # If either dimension is smaller than the crop size, resize the shorter side to crop_size
        if min(width, height) < self.crop_size:
            if width < height:
                new_width = self.crop_size
                new_height = int((self.crop_size / width) * height)
            else:
                new_height = self.crop_size
                new_width = int((self.crop_size / height) * width)
            img = img.resize((new_width, new_height), Image.BILINEAR)
        
        # Apply random crop
        transform = transforms.RandomCrop(self.crop_size)
        return transform(img)


if __name__ == "__main__":
    opt = {
        "include_ffhq": True,
        "split": "train",
        "limit_ffhq": 10000,
        "local_lsdir_cache_dir": "/data/datasets/lsdir",
        "local_ffhq_dir": "/data/datasets/FFHQ/FFHQ-1024/",  # Replace with the actual path to FFHQ images

        "blur_kernel_size":21,
        "kernel_list":['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso'],
        "kernel_prob": [0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
        "sinc_prob": 0.1,
        "blur_sigma": [0.2, 3.0],
        "betag_range": [0.5, 4.0],
        "betap_range": [1, 2.0],

        "blur_kernel_size2": 15,
        "kernel_list2": ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso'],
        "kernel_prob2": [0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
        "sinc_prob2": 0.1,
        "blur_sigma2": [0.2, 1.5],
        "betag_range2": [0.5, 4.0],
        "betap_range2": [1, 2.0],

        "final_sinc_prob": 0.8,
        "use_hflip": True,
        "use_rot": False,
        "rescale_gt": True
    }
    # Create an instance of the combined dataset
    combined_dataset = LSDIRFFHQDataset(
        opt,
    )
    print(len(combined_dataset))  # Should print the total number of images in both datasets

    # DataLoader for the combined dataset
    data_loader = DataLoader(combined_dataset, batch_size=32, shuffle=True)

    # Test loading and transforming a batch of images
    for batch in data_loader:
        print(batch['gt'].shape)  # Should print torch.Size([batch_size, 3, 512, 512])
        break  # Display only the first batch for verification
