#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os, math, random

import numpy as np
from pathlib import Path
from loguru import logger
from omegaconf import OmegaConf
from contextlib import nullcontext

from utils import util_net
from utils import util_image
from utils import util_common
from einops import rearrange

import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision.utils as vutils


from dataset import BaseData
from utils.util_image import ImageSpliterTh

class BaseSampler:
    def __init__(
            self,
            configs,
            in_path,
            out_path,
            use_amp=True,
            chop_size=128,
            chop_stride=112,
            chop_bs=1,
            seed=10000,
            distributed=False,
            ):
        '''
        Input:
            configs: config, see the yaml file in folder ./configs/
            sf: int, super-resolution scale
            seed: int, random seed
        '''
        self.configs = configs
        self.in_path = Path(in_path)
        self.out_path = Path(out_path)
        self.use_amp = use_amp
        self.seed = seed
        self.chop_size = chop_size
        self.chop_stride = chop_stride
        self.chop_bs = chop_bs
        self.sf = self.configs.diffusion.params.sf
        self.distributed = distributed

        self.setup_dist()  # setup distributed training: self.num_gpus, self.rank

        self.setup_directory()

        self.setup_seed()

        self.build_model()

    def setup_directory(self):
        if self.rank == 0:
            if not self.out_path.exists():
                self.out_path.mkdir(parents=True)
            # check the input path
            if not self.in_path.exists():
                raise FileNotFoundError(f"Input path {self.in_path} not found!")

    def setup_seed(self, seed=None):
        seed = self.seed if seed is None else seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def setup_dist(self):
        if self.distributed:
            num_gpus = torch.cuda.device_count()
        else:
            num_gpus = 1

        if num_gpus > 1:
            if mp.get_start_method(allow_none=True) is None:
                mp.set_start_method('spawn')
            rank = int(os.environ['LOCAL_RANK'])
            torch.cuda.set_device(rank % num_gpus)
            dist.init_process_group(backend='nccl', init_method='env://')

        self.num_gpus = num_gpus
        if num_gpus > 1:
            self.rank = int(os.environ['LOCAL_RANK'])
        else:
            self.rank = 0

    def write_log(self, log_str):
        if self.rank == 0:
            print(log_str, flush=True)

    def build_model(self):
        # diffusion model
        log_str = f'Building the diffusion model with length: {self.configs.diffusion.params.steps}...'
        self.write_log(log_str)
        self.base_diffusion = util_common.instantiate_from_config(self.configs.diffusion)
        model = util_common.instantiate_from_config(self.configs.model).cuda()
        ckpt_path =self.configs.ckpt_path
        assert ckpt_path is not None
        self.write_log(f'Loading Diffusion model from {ckpt_path}...')
        self.load_model(model, ckpt_path)
        self.freeze_model(model)
        self.model = model.eval()

        # autoencoder model
        if self.configs.autoencoder is not None:
            ckpt_path = self.configs.autoencoder.ckpt_path
            assert ckpt_path is not None
            self.write_log(f'Loading AutoEncoder model from {ckpt_path}...')
            autoencoder = util_common.instantiate_from_config(self.configs.autoencoder).cuda()
            self.load_model(autoencoder, ckpt_path)
            autoencoder.eval()
            self.autoencoder = autoencoder
        else:
            self.autoencoder = None

    def load_model(self, model, ckpt_path=None):
        state = torch.load(ckpt_path, map_location=f"cuda:{self.rank}")
        if 'state_dict' in state:
            state = state['state_dict']
        util_net.reload_model(model, state)

    def freeze_model(self, net):
        for params in net.parameters():
            params.requires_grad = False

class QDMSampler(BaseSampler):
    def sample_func(self, y0, noise_repeat=False, progress=False):
        '''
        Input:
            y0: n x c x h x w torch tensor, low-quality image, [-1, 1], RGB
        Output:
            sample: n x c x h x w, torch tensor, [-1, 1], RGB
        '''
        if noise_repeat:
            self.setup_seed()

        offset = self.chop_size
        ori_h, ori_w = y0.shape[2:]
        if not (ori_h % offset == 0 and ori_w % offset == 0):
            flag_pad = True
            pad_h = (math.ceil(ori_h / offset)) * offset - ori_h
            pad_w = (math.ceil(ori_w / offset)) * offset - ori_w
            y0 = F.pad(y0, pad=(0, pad_w, 0, pad_h), mode='reflect')
        else:
            flag_pad = False

        if self.configs.model.params.cond_lq:
            model_kwargs={'lq':y0,}
        else:
            model_kwargs = None

        if self.configs.autoencoder is not None:
            latent_downsamping_sf = 2**(len(self.configs.autoencoder.params.ddconfig.ch_mult) - 1)
        else:
            latent_downsamping_sf = 1
        latent_h, latent_w = y0.shape[2:]
        latent_h, latent_w = latent_h * self.configs.diffusion.params.sf, latent_w * self.configs.diffusion.params.sf
        latent_h, latent_w = latent_h // latent_downsamping_sf, latent_w // latent_downsamping_sf

        diffusion_masks = util_image.generate_quadtree_masks(
                    y0,
                    self.configs.inference.threshold,
                    latent_h,
                    latent_w
                ).unsqueeze(1)
        model_masks = util_image.generate_quadtree_masks(
                    y0,
                    self.configs.inference.threshold,
                    latent_h//self.configs.model.params.down_patch_size,
                    latent_w//self.configs.model.params.down_patch_size
                )
        model_kwargs["mask"] = model_masks.to(f"cuda:{self.rank}")
        model_kwargs["up_pred"] = False
        model_kwargs['chunk_size'] = self.configs.inference.chunk_size

        if not progress:
            results = self.base_diffusion.p_sample_loop(
                            y=y0,
                            mask=diffusion_masks,
                            model=self.model.forward_w_mask if self.configs.inference.mask_forward else self.model,
                            first_stage_model=self.autoencoder,
                            noise=None,
                            noise_repeat=noise_repeat,
                            clip_denoised=(self.autoencoder is None),
                            denoised_fn=None,
                            device=f"cuda:{self.rank}",
                            model_kwargs=model_kwargs,
                            progress=False,
                        )    # This has included the decoding for latent space
            if flag_pad:
                results = results[:, :, :ori_h*self.sf, :ori_w*self.sf]

            return results.clamp_(-1.0, 1.0)
        else:
            indices = np.linspace(
                    0,
                    self.base_diffusion.num_timesteps,
                    self.base_diffusion.num_timesteps if self.base_diffusion.num_timesteps < 5 else 4,
                    endpoint=False,
                    dtype=np.int64,
                    ).tolist()
            if not (self.base_diffusion.num_timesteps-1) in indices:
                indices.append(self.base_diffusion.num_timesteps-1)
            num_iters = 0
            for sample in self.base_diffusion.p_sample_loop_progressive(
                        y=y0,
                        mask=diffusion_masks,
                        model=self.model.forward_w_mask if self.configs.inference.mask_forward else self.model,
                        first_stage_model=self.autoencoder,
                        noise=None,
                        clip_denoised=(self.autoencoder is None),
                        model_kwargs=model_kwargs,
                        device=f"cuda:{self.rank}",
                        progress=False,
                        ):
                sample_decode = {}
                if num_iters in indices:
                    for key, value in sample.items():
                        if key in ['sample', ]:
                            sample_decode[key] = self.base_diffusion.decode_first_stage(
                                    value,
                                    self.autoencoder,
                                    ).clamp(-1.0, 1.0)
                            if flag_pad:
                                sample_decode['sample'] = sample_decode['sample'][:, :, :ori_h*self.sf, :ori_w*self.sf]
                    im_sr_progress = sample_decode['sample']
                    if num_iters + 1 == 1:
                        im_sr_all = im_sr_progress
                    else:
                        im_sr_all = torch.cat((im_sr_all, im_sr_progress), dim=1)
                num_iters += 1
            return im_sr_all                

    def inference(self):
        '''
        Inference demo.
        '''
        def _process_per_image(im_lq_tensor):
            '''
            Input:
                im_lq_tensor: b x c x h x w, torch tensor, [-1, 1], RGB
            Output:
                im_sr: h x w x c, numpy array, [0,1], RGB
            '''
            context = torch.cuda.amp.autocast if self.use_amp else nullcontext
            if im_lq_tensor.shape[2] > self.chop_size or im_lq_tensor.shape[3] > self.chop_size:
                im_spliter = ImageSpliterTh(
                        im_lq_tensor,
                        self.chop_size,
                        stride=self.chop_stride,
                        sf=self.configs.inference.sf,
                        extra_bs=self.chop_bs,
                        )
                for im_lq_pch, index_infos in im_spliter:
                    with context():
                        im_sr_pch = self.sample_func(
                                im_lq_pch,
                                noise_repeat=self.configs.inference.noise_repeat,
                                )     # 1 x c x h x w, [-1, 1]
                    im_spliter.update(im_sr_pch, index_infos)
                im_sr_tensor = im_spliter.gather()
            else:
                with context():
                    im_sr_tensor = self.sample_func(
                            im_lq_tensor,
                            noise_repeat=self.configs.inference.noise_repeat,
                            )     # 1 x c x h x w, [-1, 1]

            im_sr_tensor = im_sr_tensor * 0.5 + 0.5
            return im_sr_tensor

        in_path = Path(self.in_path)
        out_path = Path(self.out_path)

        # if self.num_gpus > 1:
        #     dist.barrier()

        if in_path.is_dir():
            dataset = BaseData(
                dir_path=str(self.in_path),
                chn=self.configs.data.chn,
                im_exts=self.configs.data.im_exts,
                need_path=True,
            )
            self.write_log(f'Find {len(dataset)} images in {in_path}')

            # Create distributed sampler for multi-GPU
            sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False) if self.num_gpus > 1 else None

            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=1,  # Force batch size to 1
                shuffle=False,
                sampler=sampler,
                drop_last=False,
            )

            for data in dataloader:

                # Directly process each single image
                lq_tensor = data['lq'].cuda()  # 1 x C x H x W
                results = _process_per_image(lq_tensor)  # 1 x H x W x C
                
                # Save results
                im_sr = util_image.tensor2img(results[0], rgb2bgr=True, min_max=(0.0, 1.0))
                im_name = Path(data['path'][0]).stem
                im_path = out_path / f"{im_name}.png"
                util_image.imwrite(im_sr, im_path, chn='bgr', dtype_in='uint8')
        else:
            im_lq = util_image.imread(in_path, chn='rgb', dtype='float32')  # h x w x c
            im_lq_tensor = util_image.img2tensor(im_lq).cuda()              # 1 x c x h x w

            im_sr_tensor = _process_per_image(
                    (im_lq_tensor - 0.5) / 0.5,
                    )

            im_sr = util_image.tensor2img(im_sr_tensor, rgb2bgr=True, min_max=(0.0, 1.0))
            im_path = out_path / f"{in_path.stem}.png"
            util_image.imwrite(im_sr, im_path, chn='bgr', dtype_in='uint8')

        self.write_log(f"Processing done, enjoy the results in {str(out_path)}")

    def inference_process(self):
        '''
        Inference demo and save the diffusion process.
        '''
        def _process_per_image(im_lq_tensor):
            '''
            Input:
                im_lq_tensor: b x c x h x w, torch tensor, [-1, 1], RGB
            Output:
                im_sr: h x w x c, numpy array, [0,1], RGB
            '''
            context = torch.cuda.amp.autocast if self.use_amp else nullcontext
            assert im_lq_tensor.shape[2] == self.chop_size and im_lq_tensor.shape[3] == self.chop_size, "Only support chop_size input for now."
            
            with context():
                process_tensor = self.sample_func(
                        im_lq_tensor,
                        noise_repeat=self.configs.inference.noise_repeat,
                        progress=True
                    )     # 1 x c x h x w, [-1, 1]

            process_tensor = process_tensor * 0.5 + 0.5
            return process_tensor

        in_path = Path(self.in_path)
        out_path = Path(self.out_path)

        if self.num_gpus > 1:
            dist.barrier()

        if in_path.is_dir():
            dataset = BaseData(
                dir_path=str(self.in_path),
                chn=self.configs.data.chn,
                im_exts=self.configs.data.im_exts,
                need_path=True,
            )
            self.write_log(f'Find {len(dataset)} images in {in_path}')

            # Create distributed sampler for multi-GPU
            sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False) if self.num_gpus > 1 else None

            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=1,  # Force batch size to 1
                shuffle=False,
                sampler=sampler,
                drop_last=False,
            )

            for data in dataloader:

                # Directly process each single image
                lq_tensor = data['lq'].cuda()  # 1 x C x H x W
                im_sr_progress = _process_per_image(lq_tensor)  # 1 x H x W x C
                
                im_sr_progress = rearrange(im_sr_progress, 'b (k c) h w -> (b k) c h w', c=lq_tensor.shape[1])

                # Save results
                im_name = Path(data['path'][0]).stem
                im_path = out_path / f"progress_{im_name}.png"
                length = self.base_diffusion.num_timesteps if self.base_diffusion.num_timesteps < 5 else 5
                self.logging_image(
                            im_sr_progress,
                            path=im_path,
                            nrow=length,
                            )
            # if self.num_gpus > 1:
            #     dist.barrier()
        else:
            im_lq = util_image.imread(in_path, chn='rgb', dtype='float32')  # h x w x c
            im_lq_tensor = util_image.img2tensor(im_lq).cuda()              # 1 x c x h x w

            im_sr_progress = _process_per_image(
                    (im_lq_tensor - 0.5) / 0.5,
                    )
            

            im_sr_progress = rearrange(im_sr_progress, 'b (k c) h w -> (b k) c h w', c=im_lq_tensor.shape[1])

            # Save results
            im_name = Path(in_path).stem
            im_path = out_path / f"progress_{im_name}.png"
            length = self.base_diffusion.num_timesteps if self.base_diffusion.num_timesteps < 5 else 5
            self.logging_image(
                        im_sr_progress,
                        path=im_path,
                        nrow=length,
                        )
        if self.num_gpus > 1:
                dist.barrier()

        self.write_log(f"Processing done, enjoy the results in {str(out_path)}")
        if dist.is_initialized():
            dist.destroy_process_group()

    def logging_image(self, im_tensor, path, nrow=8):
        """
        Args:
            im_tensor: b x c x h x w tensor
            im_tag: str
            phase: 'train' or 'val'
            nrow: number of displays in each row
        """
        im_tensor = vutils.make_grid(im_tensor, nrow=nrow, normalize=True, scale_each=True) # c x H x W
        im_np = im_tensor.cpu().permute(1,2,0).numpy()
        util_image.imwrite(im_np, path, chn="rgb" if im_np.shape[-1]==3 else "gray")

if __name__ == '__main__':
    pass

