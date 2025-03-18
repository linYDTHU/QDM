#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Adapted from ResShift official Repository: https://github.com/zsyOAOA/ResShift/blob/journal/trainer.py

import os, sys, math, time, random, datetime, functools
import pyiqa
import lpips
import wandb
import numpy as np
from pathlib import Path
from loguru import logger
from copy import deepcopy
from omegaconf import OmegaConf
from collections import OrderedDict
from einops import rearrange
from contextlib import nullcontext

from dataset import create_dataset

from utils import util_net
from utils import util_common
from utils import util_image

import torch
import torch.nn as nn
import torch.cuda.amp as amp
import torch.nn.functional as F
import torch.utils.data as udata
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP


class TrainerBase:
    def __init__(self, configs):
        self.configs = configs

        # setup distributed training: self.num_gpus, self.rank
        self.setup_dist()

        # setup seed
        self.setup_seed()

    def setup_dist(self):
        num_gpus = torch.cuda.device_count()

        if num_gpus > 1:
            if mp.get_start_method(allow_none=True) is None:
                mp.set_start_method('spawn')
            rank = int(os.environ['LOCAL_RANK'])
            torch.cuda.set_device(rank % num_gpus)
            dist.init_process_group(
                    timeout=datetime.timedelta(seconds=3600),
                    backend='nccl',
                    init_method='env://',
                    )

        self.num_gpus = num_gpus
        self.rank = int(os.environ['LOCAL_RANK']) if num_gpus > 1 else 0

    def setup_seed(self, seed=None, global_seeding=None):
        if seed is None:
            seed = self.configs.train.get('seed', 12345)
        if global_seeding is None:
            global_seeding = self.configs.train.global_seeding
            assert isinstance(global_seeding, bool)
        if not global_seeding:
            seed += self.rank
            torch.cuda.manual_seed(seed)
        else:
            torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def init_logger(self):
        if self.configs.resume:
            assert self.configs.resume.endswith(".pth")
            save_dir = Path(self.configs.resume).parents[1]
            project_id = save_dir.name
        else:
            project_id = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
            save_dir = Path(self.configs.save_dir) / project_id
            if not save_dir.exists() and self.rank == 0:
                save_dir.mkdir(parents=True)

        # setting log counter
        if self.rank == 0:
            self.log_step = {phase: 1 for phase in ['train', 'val']}
            self.log_step_img = {phase: 1 for phase in ['train', 'val']}

        # text logging
        logtxet_path = save_dir / 'training.log'
        if self.rank == 0:
            if logtxet_path.exists():
                assert self.configs.resume
            self.logger = logger
            self.logger.remove()
            self.logger.add(logtxet_path, format="{message}", mode='a', level='INFO')
            self.logger.add(sys.stdout, format="{message}")

        # tensorboard logging
        log_dir = save_dir / 'tf_logs'
        self.tf_logging = self.configs.train.tf_logging
        if self.rank == 0 and self.tf_logging:
            if not log_dir.exists():
                log_dir.mkdir()
            self.writer = SummaryWriter(str(log_dir))

        # checkpoint saving
        ckpt_dir = save_dir / 'ckpts'
        self.ckpt_dir = ckpt_dir
        if self.rank == 0 and (not ckpt_dir.exists()):
            ckpt_dir.mkdir()
        if 'ema_rate' in self.configs.train:
            self.ema_rate = self.configs.train.ema_rate
            assert isinstance(self.ema_rate, float), "Ema rate must be a float number"
            ema_ckpt_dir = save_dir / 'ema_ckpts'
            self.ema_ckpt_dir = ema_ckpt_dir
            if self.rank == 0 and (not ema_ckpt_dir.exists()):
                ema_ckpt_dir.mkdir()

        # save images into local disk
        self.local_logging = self.configs.train.local_logging
        if self.rank == 0 and self.local_logging:
            image_dir = save_dir / 'images'
            if not image_dir.exists():
                (image_dir / 'train').mkdir(parents=True)
                (image_dir / 'val').mkdir(parents=True)
            self.image_dir = image_dir

        # wandb logging
        self.wandb_logging = self.configs.train.wandb_logging
        if self.rank == 0 and self.wandb_logging:
            project_name = "quadtree-diffusion"
            self.wandb_run = wandb.init(
                # set the wandb project where this run will be logged
                project=project_name,
                name=self.configs.train.run_name,
                # track hyperparameters and run metadata
                config=OmegaConf.to_container(self.configs, resolve=True),
            )

        # logging the configurations
        if self.rank == 0:
            self.logger.info(OmegaConf.to_yaml(self.configs))

    def close_logger(self):
        if self.rank == 0 and self.tf_logging:
            self.writer.close()
        wandb.finish()

    def resume_from_ckpt(self):
        def _load_ema_state(ema_state, ckpt):
            for key in ema_state.keys():
                if key not in ckpt and key.startswith('module'):
                    ema_state[key] = deepcopy(ckpt[7:].detach().data)
                elif key not in ckpt and (not key.startswith('module')):
                    ema_state[key] = deepcopy(ckpt['module.'+key].detach().data)
                else:
                    ema_state[key] = deepcopy(ckpt[key].detach().data)

        if self.configs.resume:
            assert self.configs.resume.endswith(".pth") and os.path.isfile(self.configs.resume)

            if self.rank == 0:
                self.logger.info(f"=> Loaded checkpoint from {self.configs.resume}")
            ckpt = torch.load(self.configs.resume, map_location=f"cuda:{self.rank}")
            util_net.reload_model(self.model, ckpt['state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            torch.cuda.empty_cache()

            # learning rate scheduler
            self.iters_start = ckpt['iters_start']
            self.tic = 0.0
            for ii in range(1, self.iters_start+1):
                self.adjust_lr(ii)

            # logging
            if self.rank == 0:
                self.log_step = ckpt['log_step']
                self.log_step_img = ckpt['log_step_img']

            # EMA model
            if self.rank == 0 and hasattr(self, 'ema_rate'):
                ema_ckpt_path = self.ema_ckpt_dir / ("ema_"+Path(self.configs.resume).name)
                self.logger.info(f"=> Loaded EMA checkpoint from {str(ema_ckpt_path)}")
                ema_ckpt = torch.load(ema_ckpt_path, map_location=f"cuda:{self.rank}")
                _load_ema_state(self.ema_state, ema_ckpt)
            torch.cuda.empty_cache()

            # AMP scaler
            if self.amp_scaler is not None:
                if "amp_scaler" in ckpt:
                    self.amp_scaler.load_state_dict(ckpt["amp_scaler"])
                    if self.rank == 0:
                        self.logger.info("Loading scaler from resumed state...")

            # reset the seed
            self.setup_seed(seed=self.iters_start)
        else:
            self.iters_start = 0

    def setup_optimizaton(self):
        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                           lr=self.configs.train.lr,
                                           weight_decay=self.configs.train.weight_decay)

        # amp settings
        self.amp_scaler = amp.GradScaler() if self.configs.train.use_amp else None

    def build_model(self):
        params = self.configs.model.get('params', dict)
        model = util_common.get_obj_from_str(self.configs.model.target)(**params)
        model.cuda()
        if self.configs.model.ckpt_path is not None:
            ckpt_path = self.configs.model.ckpt_path
            if self.rank == 0:
                self.logger.info(f"Initializing model from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=f"cuda:{self.rank}")
            if 'state_dict' in ckpt:
                ckpt = ckpt['state_dict']
            util_net.reload_model(model, ckpt)
        if self.configs.train.compile.flag:
            if self.rank == 0:
                self.logger.info("Begin compiling model...")
            model = torch.compile(model)
            if self.rank == 0:
                self.logger.info("Compiling Done")
        if self.num_gpus > 1:
            self.model = DDP(model, device_ids=[self.rank,], static_graph=False)  # wrap the network
        else:
            self.model = model

        # EMA
        if self.rank == 0 and hasattr(self.configs.train, 'ema_rate'):
            self.ema_model = deepcopy(model).cuda()
            self.ema_state = OrderedDict(
                {key:deepcopy(value.data) for key, value in self.model.state_dict().items()}
                )
            self.ema_ignore_keys = [x for x in self.ema_state.keys() if ('running_' in x or 'num_batches_tracked' in x)]

        # model information
        self.print_model_info()

    def build_dataloader(self):
        def _wrap_loader(loader):
            while True: yield from loader

        # make datasets
        datasets = {'train': create_dataset(self.configs.data.get('train', dict)), }
        if hasattr(self.configs.data, 'val') and self.rank == 0:
            datasets['val'] = create_dataset(self.configs.data.get('val', dict))
        if self.rank == 0:
            for phase in datasets.keys():
                length = len(datasets[phase])
                self.logger.info('Number of images in {:s} data set: {:d}'.format(phase, length))

        # make dataloaders
        if self.num_gpus > 1:
            sampler = udata.distributed.DistributedSampler(
                    datasets['train'],
                    num_replicas=self.num_gpus,
                    rank=self.rank,
                    )
        else:
            sampler = None
        dataloaders = {'train': _wrap_loader(udata.DataLoader(
                        datasets['train'],
                        batch_size=self.configs.train.batch[0] // self.num_gpus,
                        shuffle=False if self.num_gpus > 1 else True,
                        drop_last=True,
                        num_workers=min(self.configs.train.num_workers, 4),
                        pin_memory=True,
                        prefetch_factor=self.configs.train.get('prefetch_factor', 2) if self.configs.train.num_workers > 1 else None,
                        worker_init_fn=my_worker_init_fn,
                        sampler=sampler,
                        ))}
        if hasattr(self.configs.data, 'val') and self.rank == 0:
            dataloaders['val'] = udata.DataLoader(datasets['val'],
                                                  batch_size=self.configs.train.batch[1],
                                                  shuffle=False,
                                                  drop_last=False,
                                                  num_workers=0,
                                                  pin_memory=True,
                                                 )

        self.datasets = datasets
        self.dataloaders = dataloaders
        self.sampler = sampler

    def print_model_info(self):
        if self.rank == 0:
            num_params = util_net.calculate_parameters(self.model) / 1000**2
            # self.logger.info("Detailed network architecture:")
            # self.logger.info(self.model.__repr__())
            self.logger.info(f"Number of parameters: {num_params:.2f}M")

    def prepare_data(self, data, dtype=torch.float32, phase='train'):
        data = {key:value.cuda().to(dtype=dtype) for key, value in data.items()}
        return data

    def validation(self):
        pass

    def train(self):
        self.init_logger()       # setup logger: self.logger

        self.build_model()       # build model: self.model, self.loss

        self.setup_optimizaton() # setup optimization: self.optimzer, self.sheduler

        self.resume_from_ckpt()  # resume if necessary

        self.build_dataloader()  # prepare data: self.dataloaders, self.datasets, self.sampler

        self.model.train()
        num_iters_epoch = math.ceil(len(self.datasets['train']) / self.configs.train.batch[0])
        for ii in range(self.iters_start, self.configs.train.iterations):
            self.current_iters = ii + 1

            # prepare data
            data = self.prepare_data(next(self.dataloaders['train']))

            # training phase
            self.training_step(data)

            # validation phase
            if 'val' in self.dataloaders and (ii+1) % self.configs.train.get('val_freq', 10000) == 0:
                self.validation()

            #update learning rate
            self.adjust_lr()

            # save checkpoint
            if (ii+1) % self.configs.train.save_freq == 0:
                self.save_ckpt()

            if (ii+1) % num_iters_epoch == 0 and self.sampler is not None:
                self.sampler.set_epoch(ii+1)

        # close the tensorboard/wandb processes
        self.close_logger()

    def training_step(self, data):
        pass

    def adjust_lr(self, current_iters=None):
        assert hasattr(self, 'lr_scheduler')
        self.lr_scheduler.step()

    def save_ckpt(self):
        if self.rank == 0:
            ckpt_path = self.ckpt_dir / 'model_{:d}.pth'.format(self.current_iters)
            ckpt = {
                    'iters_start': self.current_iters,
                    'log_step': {phase:self.log_step[phase] for phase in ['train', 'val']},
                    'log_step_img': {phase:self.log_step_img[phase] for phase in ['train', 'val']},
                    'state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    }
            if self.amp_scaler is not None:
                ckpt['amp_scaler'] = self.amp_scaler.state_dict()
            torch.save(ckpt, ckpt_path)
            if hasattr(self, 'ema_rate'):
                ema_ckpt_path = self.ema_ckpt_dir / 'ema_model_{:d}.pth'.format(self.current_iters)
                torch.save(self.ema_state, ema_ckpt_path)

    def reload_ema_model(self):
        if self.rank == 0:
            if self.num_gpus > 1:
                model_state = {key[7:]:value for key, value in self.ema_state.items()}
            else:
                model_state = self.ema_state
            self.ema_model.load_state_dict(model_state)

    @torch.no_grad()
    def update_ema_model(self):
        if self.num_gpus > 1:
            dist.barrier()
        if self.rank == 0:
            source_state = self.model.state_dict()
            rate = self.ema_rate
            for key, value in self.ema_state.items():
                if key in self.ema_ignore_keys:
                    self.ema_state[key] = source_state[key]
                else:
                    self.ema_state[key].mul_(rate).add_(source_state[key].detach().data, alpha=1-rate)

    def logging_image(self, im_tensor, tag, phase, add_global_step=False, nrow=8):
        """
        Args:
            im_tensor: b x c x h x w tensor
            im_tag: str
            phase: 'train' or 'val'
            nrow: number of displays in each row
        """
        assert self.tf_logging or self.local_logging
        im_tensor = vutils.make_grid(im_tensor, nrow=nrow, normalize=True, scale_each=True) # c x H x W
        if self.local_logging:
            im_path = str(self.image_dir / phase / f"{tag}-{self.log_step_img[phase]}.png")
            im_np = im_tensor.cpu().permute(1,2,0).numpy()
            util_image.imwrite(im_np, im_path, chn="rgb" if im_np.shape[-1]==3 else "gray")
        if self.tf_logging:
            self.writer.add_image(
                    f"{phase}-{tag}-{self.log_step_img[phase]}",
                    im_tensor,
                    self.log_step_img[phase],
                    )
        if add_global_step:
            self.log_step_img[phase] += 1

    def logging_metric(self, metrics, tag, phase, add_global_step=False):
        """
        Args:
            metrics: dict
            tag: str
            phase: 'train' or 'val'
        """
        if self.tf_logging:
            tag = f"{phase}-{tag}"
            if isinstance(metrics, dict):
                self.writer.add_scalars(tag, metrics, self.log_step[phase])
            else:
                self.writer.add_scalar(tag, metrics, self.log_step[phase])
        if add_global_step:
                self.log_step[phase] += 1

    def load_model(self, model, ckpt_path=None, tag='model', strict=True):
        if self.rank == 0:
            self.logger.info(f'Loading {tag} from {ckpt_path}...')
        ckpt = torch.load(ckpt_path, map_location=f"cuda:{self.rank}")
        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        if strict:
            util_net.reload_model(model, ckpt)
        else:
            model.load_state_dict(ckpt, strict=False)
        if self.rank == 0:
            self.logger.info('Loaded Done')

    def freeze_model(self, net):
        for params in net.parameters():
            params.requires_grad = False

class TrainerDifIR(TrainerBase):
    def setup_optimizaton(self):
        super().setup_optimizaton()
        if self.configs.train.lr_schedule == 'cosin':
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer=self.optimizer,
                    T_max=self.configs.train.iterations - self.configs.train.warmup_iterations,
                    eta_min=self.configs.train.lr_min,
                    )

    def build_model(self):
        super().build_model()
        if self.rank == 0 and hasattr(self.configs.train, 'ema_rate'):
            self.ema_ignore_keys.extend([x for x in self.ema_state.keys() if 'relative_position_index' in x])

        # autoencoder
        if self.configs.autoencoder is not None:
            if self.rank == 0:
                self.logger.info(f"Restoring autoencoder from {self.configs.autoencoder.ckpt_path}")
            params = self.configs.autoencoder.get('params', dict)
            autoencoder = util_common.get_obj_from_str(self.configs.autoencoder.target)(**params)
            autoencoder.cuda()
            self.load_model(autoencoder, self.configs.autoencoder.ckpt_path, tag='autoencoder', strict=True)
            self.freeze_model(autoencoder)
            autoencoder.eval()

            if self.configs.train.compile.flag:
                if self.rank == 0:
                    self.logger.info("Begin compiling autoencoder model...")
                autoencoder = torch.compile(autoencoder)
                if self.rank == 0:
                    self.logger.info("Compiling Done")
            self.autoencoder = autoencoder
        else:
            self.autoencoder = None

        # LPIPS metric
        # lpips_loss = lpips.LPIPS(net='vgg').to(f"cuda:{self.rank}")
        # for params in lpips_loss.parameters():
        #     params.requires_grad_(False)
        # lpips_loss.eval()
        # if self.configs.train.compile.flag:
        #     if self.rank == 0:
        #         self.logger.info("Begin compiling LPIPS Metric...")
        #     lpips_loss = torch.compile(lpips_loss)
        #     if self.rank == 0:
        #         self.logger.info("Compiling Done")
        # self.lpips_loss = lpips_loss
        self.lpips_loss = pyiqa.create_metric('lpips', device=f"cuda:{self.rank}", as_loss=False)
        if self.configs.train.compile.flag:
            if self.rank == 0:
                self.logger.info("Begin compiling LPIPS Metric...")
            self.lpips_loss = torch.compile(self.lpips_loss)
            if self.rank == 0:
                self.logger.info("Compiling Done")

        params = self.configs.diffusion.get('params', dict)
        self.base_diffusion = util_common.get_obj_from_str(self.configs.diffusion.target)(**params)

    @torch.no_grad()
    def _dequeue_and_enqueue(self):
        """It is the training pair pool for increasing the diversity in a batch.

        Batch processing limits the diversity of synthetic degradations in a batch. For example, samples in a
        batch could not have different resize scaling factors. Therefore, we employ this training pair pool
        to increase the degradation diversity in a batch.
        """
        # initialize
        b, c, h, w = self.lq.size()
        if not hasattr(self, 'queue_size'):
            self.queue_size = self.configs.degradation.get('queue_size', b*10)
        if not hasattr(self, 'queue_lr'):
            assert self.queue_size % b == 0, f'queue size {self.queue_size} should be divisible by batch size {b}'
            self.queue_lr = torch.zeros(self.queue_size, c, h, w).cuda()
            _, c, h, w = self.gt.size()
            self.queue_gt = torch.zeros(self.queue_size, c, h, w).cuda()
            self.queue_ptr = 0
        if self.queue_ptr == self.queue_size:  # the pool is full
            # do dequeue and enqueue
            # shuffle
            idx = torch.randperm(self.queue_size)
            self.queue_lr = self.queue_lr[idx]
            self.queue_gt = self.queue_gt[idx]
            # get first b samples
            lq_dequeue = self.queue_lr[0:b, :, :, :].clone()
            gt_dequeue = self.queue_gt[0:b, :, :, :].clone()
            # update the queue
            self.queue_lr[0:b, :, :, :] = self.lq.clone()
            self.queue_gt[0:b, :, :, :] = self.gt.clone()

            self.lq = lq_dequeue
            self.gt = gt_dequeue
        else:
            # only do enqueue
            self.queue_lr[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.lq.clone()
            self.queue_gt[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.gt.clone()
            self.queue_ptr = self.queue_ptr + b

    @torch.no_grad()
    def prepare_data(self, data, dtype=torch.float32, realesrgan=None, medsrgan=None, phase='train'):
        if realesrgan is None:
            realesrgan = self.configs.data.get(phase, dict).type == 'lsdirffhq' or\
                            self.configs.data.get(phase, dict).type == 'combined'
        if medsrgan is None:
            medsrgan = self.configs.data.get(phase, dict).type == 'medsrgan'

        if realesrgan and phase == 'train':
            opts_degradation = self.configs.get('degradation', dict)
            #load all data on devices
            im_gt = data['gt'].cuda()
            kernel1 = data['kernel1'].cuda()
            kernel2 = data['kernel2'].cuda()
            sinc_kernel = data['sinc_kernel'].cuda()

            data = self.datasets[phase].degrade_fun(opts_degradation, im_gt, kernel1, kernel2, sinc_kernel)
            im_lq, im_gt = data['lq'], data['gt']
            im_lq = (im_lq - 0.5) / 0.5  # [0, 1] to [-1, 1]
            im_gt = (im_gt - 0.5) / 0.5  # [0, 1] to [-1, 1]
            self.lq, self.gt, flag_nan = replace_nan_in_batch(im_lq, im_gt)
            if flag_nan:
                with open(f"records_nan_rank{self.rank}.log", 'a') as f:
                    f.write(f'Find Nan value in rank{self.rank}\n')

            # training pair pool
            self._dequeue_and_enqueue()
            self.lq = self.lq.contiguous()  # for the warning: grad and param do not obey the gradient layout contract

            return {'lq':self.lq, 'gt':self.gt}
        elif medsrgan and phase == 'train':
            opts_degradation = self.configs.get('degradation', dict)
            #load all data on devices
            im_gt = data['gt'].cuda()
            kernel1 = data['kernel1'].cuda()

            data = self.datasets[phase].degrade_fun(opts_degradation, im_gt, kernel1)
            im_lq, im_gt = data['lq'], data['gt']
            im_lq = (im_lq - 0.5) / 0.5  # [0, 1] to [-1, 1]
            im_gt = (im_gt - 0.5) / 0.5  # [0, 1] to [-1, 1]
            self.lq, self.gt, flag_nan = replace_nan_in_batch(im_lq, im_gt)
            if flag_nan:
                with open(f"records_nan_rank{self.rank}.log", 'a') as f:
                    f.write(f'Find Nan value in rank{self.rank}\n')

            # training pair pool
            self._dequeue_and_enqueue()
            self.lq = self.lq.contiguous()  # for the warning: grad and param do not obey the gradient layout contract

            return {'lq':self.lq, 'gt':self.gt}
        elif phase == 'val':
            return {key:value.cuda().to(dtype=dtype) for key, value in data.items()}
        else:
            return {key:value.cuda().to(dtype=dtype) for key, value in data.items()}

    def backward_step(self, dif_loss_wrapper, micro_data, num_grad_accumulate, tt):
        context = torch.cuda.amp.autocast if self.configs.train.use_amp else nullcontext
        with context():
            losses, z_t, z0_pred = dif_loss_wrapper()
            losses['loss'] = losses['mse']
            loss = losses['loss'].mean() / num_grad_accumulate
        if self.amp_scaler is None:
            loss.backward()
        else:
            self.amp_scaler.scale(loss).backward()

        return losses, z0_pred, z_t

    def training_step(self, data):
        current_batchsize = data['gt'].shape[0]
        micro_batchsize = self.configs.train.microbatch
        num_grad_accumulate = math.ceil(current_batchsize / micro_batchsize)

        for jj in range(0, current_batchsize, micro_batchsize):
            micro_data = {key:value[jj:jj+micro_batchsize,] for key, value in data.items()}
            last_batch = (jj+micro_batchsize >= current_batchsize)
            tt = torch.randint(
                    0, self.base_diffusion.num_timesteps,
                    size=(micro_data['gt'].shape[0],),
                    device=f"cuda:{self.rank}",
                    )
            if self.configs.autoencoder is not None:
                latent_downsamping_sf = 2**(len(self.configs.autoencoder.params.ddconfig.ch_mult) - 1)
            else:
                latent_downsamping_sf = 1
            latent_resolution = micro_data['gt'].shape[-1] // latent_downsamping_sf
            if self.configs.autoencoder is not None:
                noise_chn = self.configs.autoencoder.params.embed_dim
            else:
                noise_chn = micro_data['gt'].shape[1]
            noise = torch.randn(
                    size= (micro_data['gt'].shape[0], noise_chn,) + (latent_resolution, ) * 2,
                    device=micro_data['gt'].device,
                    )
            if self.configs.model.params.cond_lq:
                model_kwargs = {'lq':micro_data['lq'],}
            else:
                model_kwargs = None
            compute_losses = functools.partial(
                self.base_diffusion.training_losses,
                self.model,
                micro_data['gt'],
                micro_data['lq'],
                tt,
                first_stage_model=self.autoencoder,
                model_kwargs=model_kwargs,
                noise=noise,
            )
            if last_batch or self.num_gpus <= 1:
                losses, z0_pred, z_t = self.backward_step(compute_losses, micro_data, num_grad_accumulate, tt)
            else:
                with self.model.no_sync():
                    losses, z0_pred, z_t = self.backward_step(compute_losses, micro_data, num_grad_accumulate, tt)

            # make logging
            if last_batch:
                self.log_step_train(losses, tt, micro_data, z_t, z0_pred.detach())
                if self.rank == 0 and self.wandb_logging:
                    self.wandb_run.log({"loss": losses['loss'].mean(), "mse": losses['mse'].mean()})

        if self.configs.train.use_amp:
            self.amp_scaler.step(self.optimizer)
            self.amp_scaler.update()
        else:
            self.optimizer.step()

        # grad zero
        self.model.zero_grad()

        if hasattr(self.configs.train, 'ema_rate'):
            self.update_ema_model()

    def adjust_lr(self, current_iters=None):
        base_lr = self.configs.train.lr
        warmup_steps = self.configs.train.warmup_iterations
        current_iters = self.current_iters if current_iters is None else current_iters
        if current_iters <= warmup_steps:
            for params_group in self.optimizer.param_groups:
                params_group['lr'] = (current_iters / warmup_steps) * base_lr
        else:
            if hasattr(self, 'lr_scheduler'):
                self.lr_scheduler.step()

    def log_step_train(self, loss, tt, batch, z_t, z0_pred, phase='train'):
        '''
        param loss: a dict recording the loss informations
        param tt: 1-D tensor, time steps
        '''
        if self.rank == 0:
            chn = batch['gt'].shape[1]
            num_timesteps = self.base_diffusion.num_timesteps
            record_steps = [1, (num_timesteps // 2) + 1, num_timesteps]
            if self.current_iters % self.configs.train.log_freq[0] == 1:
                self.loss_mean = {key:torch.zeros(size=(len(record_steps),), dtype=torch.float64)
                                  for key in loss.keys()}
                self.loss_count = torch.zeros(size=(len(record_steps),), dtype=torch.float64)
            for jj in range(len(record_steps)):
                for key, value in loss.items():
                    index = record_steps[jj] - 1
                    mask = torch.where(tt == index, torch.ones_like(tt), torch.zeros_like(tt))
                    current_loss = torch.sum(value.detach() * mask)
                    self.loss_mean[key][jj] += current_loss.item()
                self.loss_count[jj] += mask.sum().item()

            if self.current_iters % self.configs.train.log_freq[0] == 0:
                if torch.any(self.loss_count == 0):
                    self.loss_count += 1e-4
                for key in loss.keys():
                    self.loss_mean[key] /= self.loss_count
                log_str = 'Train: {:06d}/{:06d}, Loss/MSE: '.format(
                        self.current_iters,
                        self.configs.train.iterations)
                for jj, current_record in enumerate(record_steps):
                    log_str += 't({:d}):{:.1e}/{:.1e}, '.format(
                            current_record,
                            self.loss_mean['loss'][jj].item(),
                            self.loss_mean['mse'][jj].item(),
                            )
                log_str += 'lr:{:.2e}'.format(self.optimizer.param_groups[0]['lr'])
                self.logger.info(log_str)
                self.logging_metric(self.loss_mean, tag='Loss', phase=phase, add_global_step=True)
            if self.current_iters % self.configs.train.log_freq[1] == 0:
                self.logging_image(batch['lq'], tag='lq', phase=phase, add_global_step=False)
                self.logging_image(batch['gt'], tag='gt', phase=phase, add_global_step=False)
                micro_batchsize = 8
                with torch.no_grad():
                    decoded_chunks = []
                    for i in range(0, z_t.size(0), micro_batchsize):
                        z_t_chunk = z_t[i:i+micro_batchsize]

                        # Scale and decode the chunk
                        decoded_chunk = self.base_diffusion.decode_first_stage(
                            z_t_chunk,
                            self.autoencoder,
                        )

                        decoded_chunks.append(decoded_chunk)
                    x_t = torch.cat(decoded_chunks, dim=0)
                self.logging_image(x_t, tag='diffused', phase=phase, add_global_step=False)
                with torch.no_grad():
                    decoded_chunks = []
                    for i in range(0, z0_pred.size(0), micro_batchsize):
                        z0_pred_chunk = z0_pred[i:i+micro_batchsize]

                        # Scale and decode the chunk
                        decoded_chunk = self.base_diffusion.decode_first_stage(
                            z0_pred_chunk,
                            self.autoencoder,
                        )

                        decoded_chunks.append(decoded_chunk)
                    x0_pred = torch.cat(decoded_chunks, dim=0)
                self.logging_image(x0_pred, tag='x0-pred', phase=phase, add_global_step=True)

            if self.current_iters % self.configs.train.save_freq == 1:
                self.tic = time.time()
            if self.current_iters % self.configs.train.save_freq == 0:
                self.toc = time.time()
                elaplsed = (self.toc - self.tic)
                self.logger.info(f"Elapsed time: {elaplsed:.2f}s")
                self.logger.info("="*100)

    def validation(self, phase='val'):
        if self.rank == 0:
            if self.configs.train.use_ema_val:
                self.reload_ema_model()
                self.ema_model.eval()
            else:
                self.model.eval()

            indices = np.linspace(
                    0,
                    self.base_diffusion.num_timesteps,
                    self.base_diffusion.num_timesteps if self.base_diffusion.num_timesteps < 5 else 4,
                    endpoint=False,
                    dtype=np.int64,
                    ).tolist()
            if not (self.base_diffusion.num_timesteps-1) in indices:
                indices.append(self.base_diffusion.num_timesteps-1)
            batch_size = self.configs.train.batch[1]
            num_iters_epoch = math.ceil(len(self.datasets[phase]) / batch_size)
            mean_psnr = mean_lpips = 0
            for ii, data in enumerate(self.dataloaders[phase]):
                data = self.prepare_data(data, phase='val')
                if 'gt' in data:
                    im_lq, im_gt = data['lq'], data['gt']
                else:
                    im_lq = data['lq']
                num_iters = 0
                if self.configs.model.params.cond_lq:
                    model_kwargs = {'lq':data['lq'],}
                    if 'mask' in data:
                        model_kwargs['mask'] = data['mask']
                else:
                    model_kwargs = None
                tt = torch.tensor(
                        [self.base_diffusion.num_timesteps, ]*im_lq.shape[0],
                        dtype=torch.int64,
                        ).cuda()
                for sample in self.base_diffusion.p_sample_loop_progressive(
                        y=im_lq,
                        model=self.ema_model if self.configs.train.use_ema_val else self.model,
                        first_stage_model=self.autoencoder,
                        noise=None,
                        clip_denoised=True if self.autoencoder is None else False,
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
                        im_sr_progress = sample_decode['sample']
                        if num_iters + 1 == 1:
                            im_sr_all = im_sr_progress
                        else:
                            im_sr_all = torch.cat((im_sr_all, im_sr_progress), dim=1)
                    num_iters += 1
                    tt -= 1

                if 'gt' in data:
                    mean_psnr += util_image.batch_PSNR(
                            sample_decode['sample'] * 0.5 + 0.5,
                            im_gt * 0.5 + 0.5,
                            ycbcr=self.configs.train.val_y_channel,
                            )
                    mean_lpips += self.lpips_loss(
                            sample_decode['sample'] * 0.5 + 0.5,
                            im_gt * 0.5 + 0.5,
                            ).sum().item()

                if (ii + 1) % self.configs.train.log_freq[2] == 0:
                    self.logger.info(f'Validation: {ii+1:02d}/{num_iters_epoch:02d}...')

                    im_sr_all = rearrange(im_sr_all, 'b (k c) h w -> (b k) c h w', c=im_lq.shape[1])
                    self.logging_image(
                            im_sr_all,
                            tag='progress',
                            phase=phase,
                            add_global_step=False,
                            nrow=len(indices),
                            )
                    if 'gt' in data:
                        self.logging_image(im_gt, tag='gt', phase=phase, add_global_step=False)
                    self.logging_image(im_lq, tag='lq', phase=phase, add_global_step=True)

            if 'gt' in data:
                mean_psnr /= len(self.datasets[phase])
                mean_lpips /= len(self.datasets[phase])
                self.logger.info(f'Validation Metric: PSNR={mean_psnr:5.2f}, LPIPS={mean_lpips:6.4f}...')
                if self.rank == 0 and self.wandb_logging:
                    self.wandb_run.log({"validation_step":self.log_step["val"], "PSNR": mean_psnr, "LPIPS": mean_lpips})
                self.logging_metric(mean_psnr, tag='PSNR', phase=phase, add_global_step=False)
                self.logging_metric(mean_lpips, tag='LPIPS', phase=phase, add_global_step=True)

            self.logger.info("="*100)

            if not (self.configs.train.use_ema_val and hasattr(self.configs.train, 'ema_rate')):
                self.model.train()

class TrainerDifIRLPIPS(TrainerDifIR):
    def build_model(self):
        super().build_model()
        if self.rank == 0 and hasattr(self.configs.train, 'ema_rate'):
            self.ema_ignore_keys.extend([x for x in self.ema_state.keys() if 'relative_position_index' in x])

        # autoencoder
        if self.configs.autoencoder is not None:
            if self.rank == 0:
                self.logger.info(f"Restoring autoencoder from {self.configs.autoencoder.ckpt_path}")
            params = self.configs.autoencoder.get('params', dict)
            autoencoder = util_common.get_obj_from_str(self.configs.autoencoder.target)(**params)
            autoencoder.cuda()
            self.load_model(autoencoder, self.configs.autoencoder.ckpt_path, tag='autoencoder', strict=True)
            self.freeze_model(autoencoder)
            autoencoder.eval()

            if self.configs.train.compile.flag:
                if self.rank == 0:
                    self.logger.info("Begin compiling autoencoder model...")
                autoencoder = torch.compile(autoencoder)
                if self.rank == 0:
                    self.logger.info("Compiling Done")
            self.autoencoder = autoencoder
        else:
            self.autoencoder = None

        # LPIPS metric
        if hasattr(self.configs, 'lpips'):
            lpips_net = self.configs.lpips.net
        else:
            lpips_net = 'vgg'
        if self.rank == 0:
            self.logger.info(f"Loading LIIPS Metric: {lpips_net}...")
        lpips_loss = lpips.LPIPS(net=lpips_net).to(f"cuda:{self.rank}")
        for params in lpips_loss.parameters():
            params.requires_grad_(False)
        lpips_loss.eval()
        if self.configs.train.compile.flag:
            if self.rank == 0:
                self.logger.info("Begin compiling LPIPS Metric...")
            lpips_loss = torch.compile(lpips_loss, mode=self.configs.train.compile.mode)
            if self.rank == 0:
                self.logger.info("Compiling Done")
        self.lpips_loss = lpips_loss
        self.lpips_test_loss = pyiqa.create_metric('lpips', device=f"cuda:{self.rank}", as_loss=False)
        if self.configs.train.compile.flag:
            if self.rank == 0:
                self.logger.info("Begin compiling LPIPS Test Metric...")
            self.lpips_test_loss = torch.compile(self.lpips_test_loss)
            if self.rank == 0:
                self.logger.info("Compiling Done")

        params = self.configs.diffusion.get('params', dict)
        self.base_diffusion = util_common.get_obj_from_str(self.configs.diffusion.target)(**params)

    def backward_step(self, dif_loss_wrapper, micro_data, num_grad_accumulate, tt):
        loss_coef = self.configs.train.get('loss_coef')
        context = torch.cuda.amp.autocast if self.configs.train.use_amp else nullcontext
        # diffusion loss
        with context():
            losses, z_t, z0_pred = dif_loss_wrapper()
            x0_pred = self.base_diffusion.decode_first_stage(
                    z0_pred,
                    self.autoencoder,
                    ) # f16
            self.current_x0_pred = x0_pred.detach()

            # classification loss
            # losses["lpips"] = self.lpips_loss(
            #         x0_pred.clamp(-1.0, 1.0)/2 + 0.5,
            #         micro_data['gt']/2 + 0.5,
            #         ).to(z0_pred.dtype).view(-1)
            losses["lpips"] = self.lpips_loss(
                    x0_pred,
                    micro_data['gt'],
                    ).to(z0_pred.dtype).view(-1)
            flag_nan = torch.any(torch.isnan(losses["lpips"]))
            if flag_nan:
                losses["lpips"] = torch.nan_to_num(losses["lpips"], nan=0.0)

            losses["mse"] *= loss_coef[0]
            losses["lpips"] *= loss_coef[1]

            assert losses["mse"].shape == losses["lpips"].shape
            if flag_nan:
                losses["loss"] = losses["mse"]
            else:
                losses["loss"] = losses["mse"] + losses["lpips"]
            loss = losses['loss'].mean() / num_grad_accumulate
        if self.amp_scaler is None:
            loss.backward()
        else:
            self.amp_scaler.scale(loss).backward()

        return losses, z0_pred, z_t

    def log_step_train(self, loss, tt, batch, z_t, z0_pred, phase='train'):
        '''
        param loss: a dict recording the loss informations
        param tt: 1-D tensor, time steps
        '''
        if self.rank == 0:
            chn = batch['gt'].shape[1]
            num_timesteps = self.base_diffusion.num_timesteps
            record_steps = [1, (num_timesteps // 2) + 1, num_timesteps]
            if self.current_iters % self.configs.train.log_freq[0] == 1:
                self.loss_mean = {key:torch.zeros(size=(len(record_steps),), dtype=torch.float64)
                                  for key in loss.keys()}
                self.loss_count = torch.zeros(size=(len(record_steps),), dtype=torch.float64)
            for jj in range(len(record_steps)):
                for key, value in loss.items():
                    index = record_steps[jj] - 1
                    mask = torch.where(tt == index, torch.ones_like(tt), torch.zeros_like(tt))
                    assert value.shape == mask.shape
                    current_loss = torch.sum(value.detach() * mask)
                    self.loss_mean[key][jj] += current_loss.item()
                self.loss_count[jj] += mask.sum().item()

            if self.current_iters % self.configs.train.log_freq[0] == 0:
                if torch.any(self.loss_count == 0):
                    self.loss_count += 1e-4
                for key in loss.keys():
                    self.loss_mean[key] /= self.loss_count
                log_str = 'Train: {:06d}/{:06d}, MSE/LPIPS: '.format(
                        self.current_iters,
                        self.configs.train.iterations)
                for jj, current_record in enumerate(record_steps):
                    log_str += 't({:d}):{:.1e}/{:.1e}, '.format(
                            current_record,
                            self.loss_mean['mse'][jj].item(),
                            self.loss_mean['lpips'][jj].item(),
                            )
                log_str += 'lr:{:.2e}'.format(self.optimizer.param_groups[0]['lr'])
                self.logger.info(log_str)
                self.logging_metric(self.loss_mean, tag='Loss', phase=phase, add_global_step=True)
            if self.current_iters % self.configs.train.log_freq[1] == 0:
                self.logging_image(batch['lq'], tag='lq', phase=phase, add_global_step=False)
                self.logging_image(batch['gt'], tag='gt', phase=phase, add_global_step=False)
                x_t = self.base_diffusion.decode_first_stage(
                        self.base_diffusion._scale_input(z_t, tt),
                        self.autoencoder,
                        )
                self.logging_image(x_t, tag='diffused', phase=phase, add_global_step=False)
                self.logging_image(self.current_x0_pred, tag='x0-pred', phase=phase, add_global_step=True)

            if self.current_iters % self.configs.train.save_freq == 1:
                self.tic = time.time()
            if self.current_iters % self.configs.train.save_freq == 0:
                self.toc = time.time()
                elaplsed = (self.toc - self.tic)
                self.logger.info(f"Elapsed time: {elaplsed:.2f}s")
                self.logger.info("="*100)

class TrainerQDM(TrainerDifIR):
    def training_step(self, data):
        current_batchsize = data['gt'].shape[0]
        micro_batchsize = self.configs.train.microbatch
        num_grad_accumulate = math.ceil(current_batchsize / micro_batchsize)

        for jj in range(0, current_batchsize, micro_batchsize):
            micro_data = {key:value[jj:jj+micro_batchsize,] for key, value in data.items()}
            last_batch = (jj+micro_batchsize >= current_batchsize)
            tt = torch.randint(
                    0, self.base_diffusion.num_timesteps,
                    size=(micro_data['gt'].shape[0],),
                    device=f"cuda:{self.rank}",
            )

            if self.configs.autoencoder is not None:
                latent_downsamping_sf = 2**(len(self.configs.autoencoder.params.ddconfig.ch_mult) - 1)
            else:
                latent_downsamping_sf = 1
            latent_h, latent_w = micro_data['gt'].shape[-2:]
            latent_h, latent_w = latent_h // latent_downsamping_sf, latent_w // latent_downsamping_sf
            if self.configs.autoencoder is not None:
                noise_chn = self.configs.autoencoder.params.embed_dim
            else:
                noise_chn = micro_data['gt'].shape[1]
            noise = torch.randn(
                    size= (micro_data['gt'].shape[0], noise_chn,) + (latent_h, latent_w),
                    device=micro_data['gt'].device,
                    )
            
            if self.configs.model.params.cond_lq:
                model_kwargs = {'lq':micro_data['lq']}
            else:
                model_kwargs = None

            diffusion_mask = util_image.generate_quadtree_masks(
                micro_data['lq'],
                self.configs.train.threshold,
                latent_h,
                latent_w
            ).unsqueeze(1)
            model_mask = util_image.generate_quadtree_masks(
                micro_data['lq'],
                self.configs.train.threshold,
                latent_h//(self.configs.model.params.down_patch_size * self.configs.model.params.down_large_patch_size),
                latent_w//(self.configs.model.params.down_patch_size * self.configs.model.params.down_large_patch_size)
            )
            model_kwargs["mask"] = model_mask
            model_kwargs["up_pred"] = True

            compute_losses = functools.partial(
                self.base_diffusion.training_losses,
                self.model,
                micro_data['gt'],
                micro_data['lq'],
                diffusion_mask,
                tt,
                first_stage_model=self.autoencoder,
                model_kwargs=model_kwargs,
                noise=noise,
            )
            if last_batch or self.num_gpus <= 1:
                losses, z0_pred, z0_up_pred, z_t = self.backward_step(compute_losses, micro_data, num_grad_accumulate, tt)
            else:
                with self.model.no_sync():
                    losses, z0_pred, z0_up_pred, z_t = self.backward_step(compute_losses, micro_data, num_grad_accumulate, tt)

            # make logging
            if last_batch:
                self.log_step_train(losses, tt, micro_data, z_t, z0_pred.detach(), z0_up_pred.detach())
                if self.rank == 0 and self.wandb_logging:
                    self.wandb_run.log({"loss": losses['loss'].mean(), "mse": losses['mse'].mean(), "mse_up": losses["mse_up"].mean()})
        
        if self.configs.train.use_amp:
            self.amp_scaler.step(self.optimizer)
            self.amp_scaler.update()
        else:
            self.optimizer.step()

        # grad zero
        self.model.zero_grad()

        if hasattr(self.configs.train, 'ema_rate'):
            self.update_ema_model()
    
    def backward_step(self, dif_loss_wrapper, micro_data, num_grad_accumulate, tt):
        context = torch.cuda.amp.autocast if self.configs.train.use_amp else nullcontext
        loss_coef = self.configs.train.get('loss_coef')
        with context():
            losses, z_t, z0_pred, z0_up_pred = dif_loss_wrapper()
            mse = losses['mse'] * loss_coef[0]
            mse_up = losses["mse_up"] * loss_coef[1]
            losses["loss"] = mse + mse_up
            loss = (mse + mse_up).mean() / num_grad_accumulate            
            loss = losses['loss'].mean() / num_grad_accumulate
        if self.amp_scaler is None:
            loss.backward()
        else:
            self.amp_scaler.scale(loss).backward()

        return losses, z0_pred, z0_up_pred, z_t

    def log_step_train(self, loss, tt, batch, z_t, z0_pred, z0_up_pred, phase='train'):
        '''
        param loss: a dict recording the loss informations
        param tt: 1-D tensor, time steps
        '''
        if self.rank == 0:
            num_timesteps = self.base_diffusion.num_timesteps
            record_steps = [1, (num_timesteps // 2) + 1, num_timesteps]
            if self.current_iters % self.configs.train.log_freq[0] == 1:
                self.loss_mean = {key:torch.zeros(size=(len(record_steps),), dtype=torch.float64)
                                  for key in loss.keys()}
                self.loss_count = torch.zeros(size=(len(record_steps),), dtype=torch.float64)
            for jj in range(len(record_steps)):
                for key, value in loss.items():
                    index = record_steps[jj] - 1
                    mask = torch.where(tt == index, torch.ones_like(tt), torch.zeros_like(tt))
                    current_loss = torch.sum(value.detach() * mask)
                    self.loss_mean[key][jj] += current_loss.item()
                self.loss_count[jj] += mask.sum().item()

            if self.current_iters % self.configs.train.log_freq[0] == 0:
                if torch.any(self.loss_count == 0):
                    self.loss_count += 1e-4
                for key in loss.keys():
                    self.loss_mean[key] /= self.loss_count
                log_str = 'Train: {:06d}/{:06d}, Loss/MSE/MSE_UP: '.format(
                    self.current_iters,
                    self.configs.train.iterations)
                for jj, current_record in enumerate(record_steps):
                    log_str += 't({:d}):{:.1e}/{:.1e}/{:.1e}, '.format(
                            current_record,
                            self.loss_mean['loss'][jj].item(),
                            self.loss_mean['mse'][jj].item(),
                            self.loss_mean['mse_up'][jj].item(),
                            )
                log_str += 'lr:{:.2e}'.format(self.optimizer.param_groups[0]['lr'])
                self.logger.info(log_str)
                self.logging_metric(self.loss_mean, tag='Loss', phase=phase, add_global_step=True)
            if self.current_iters % self.configs.train.log_freq[1] == 0:
                self.logging_image(batch['lq'], tag='lq', phase=phase, add_global_step=False)
                self.logging_image(batch['gt'], tag='gt', phase=phase, add_global_step=False)
                micro_batchsize = 8
                with torch.no_grad():
                    decoded_chunks = []
                    for i in range(0, z_t.size(0), micro_batchsize):
                        z_t_chunk = z_t[i:i+micro_batchsize]

                        # Scale and decode the chunk
                        decoded_chunk = self.base_diffusion.decode_first_stage(
                            z_t_chunk,
                            self.autoencoder,
                        )

                        decoded_chunks.append(decoded_chunk)
                    x_t = torch.cat(decoded_chunks, dim=0)
                self.logging_image(x_t, tag='diffused', phase=phase, add_global_step=False)
                with torch.no_grad():
                    decoded_chunks = []
                    for i in range(0, z0_pred.size(0), micro_batchsize):
                        z0_pred_chunk = z0_pred[i:i+micro_batchsize]

                        # Scale and decode the chunk
                        decoded_chunk = self.base_diffusion.decode_first_stage(
                            z0_pred_chunk,
                            self.autoencoder,
                        )

                        decoded_chunks.append(decoded_chunk)
                    x0_pred = torch.cat(decoded_chunks, dim=0)
                self.logging_image(x0_pred, tag='x0-pred', phase=phase, add_global_step=False)

                with torch.no_grad():
                    decoded_chunks = []
                    for i in range(0, z0_up_pred.size(0), micro_batchsize):
                        z0_up_pred_chunk = z0_up_pred[i:i+micro_batchsize]

                        # Scale and decode the chunk
                        decoded_chunk = self.base_diffusion.decode_first_stage(
                            z0_up_pred_chunk,
                            self.autoencoder,
                        )

                        decoded_chunks.append(decoded_chunk)
                    x0_up_pred = torch.cat(decoded_chunks, dim=0)
                self.logging_image(x0_up_pred, tag='x0-up-pred', phase=phase, add_global_step=True)

            if self.current_iters % self.configs.train.save_freq == 1:
                self.tic = time.time()
            if self.current_iters % self.configs.train.save_freq == 0:
                self.toc = time.time()
                elaplsed = (self.toc - self.tic)
                self.logger.info(f"Elapsed time: {elaplsed:.2f}s")
                self.logger.info("="*100)

    def validation(self, phase='val'):
        if self.rank == 0:
            if self.configs.train.use_ema_val:
                self.reload_ema_model()
                self.ema_model.eval()
            else:
                self.model.eval()
            indices = np.linspace(
                    0,
                    self.base_diffusion.num_timesteps,
                    self.base_diffusion.num_timesteps if self.base_diffusion.num_timesteps < 5 else 4,
                    endpoint=False,
                    dtype=np.int64,
                    ).tolist()
            if not (self.base_diffusion.num_timesteps-1) in indices:
                indices.append(self.base_diffusion.num_timesteps-1)
            batch_size = self.configs.train.batch[1]
            num_iters_epoch = math.ceil(len(self.datasets[phase]) / batch_size)
            mean_psnr = mean_lpips = 0
            for ii, data in enumerate(self.dataloaders[phase]):
                data = self.prepare_data(data, phase='val')
                if 'gt' in data:
                    im_lq, im_gt = data['lq'], data['gt']
                else:
                    im_lq = data['lq']

                if self.configs.autoencoder is not None:
                    latent_downsamping_sf = 2**(len(self.configs.autoencoder.params.ddconfig.ch_mult) - 1)
                else:
                    latent_downsamping_sf = 1
                latent_h, latent_w = im_lq.shape[2:]
                latent_h, latent_w = latent_h * self.configs.diffusion.params.sf, latent_w * self.configs.diffusion.params.sf
                latent_h, latent_w = latent_h // latent_downsamping_sf, latent_w // latent_downsamping_sf
                
                diffusion_mask = util_image.generate_quadtree_masks(
                    im_lq,
                    self.configs.train.threshold,
                    latent_h,
                    latent_w
                ).unsqueeze(1)

                model_mask = util_image.generate_quadtree_masks(
                    im_lq,
                    self.configs.train.threshold,
                    latent_h//(self.configs.model.params.down_patch_size * self.configs.model.params.down_large_patch_size),
                    latent_w//(self.configs.model.params.down_patch_size * self.configs.model.params.down_large_patch_size)
                )

                if self.configs.model.params.cond_lq:
                    model_kwargs = {'lq':im_lq,}
                else:
                    model_kwargs = None
                model_kwargs["mask"] = model_mask
                model_kwargs["up_pred"] = False # default up_pred is True
                
                tt = torch.tensor(
                        [self.base_diffusion.num_timesteps, ]*im_lq.shape[0],
                        dtype=torch.int64,
                        ).cuda()
                
                num_iters = 0
                for sample in self.base_diffusion.p_sample_loop_progressive(
                        y=im_lq,
                        mask=diffusion_mask,
                        model=self.ema_model if self.configs.train.use_ema_val else self.model,
                        first_stage_model=self.autoencoder,
                        noise=None,
                        clip_denoised=True if self.autoencoder is None else False,
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
                        im_sr_progress = sample_decode['sample']
                        if num_iters + 1 == 1:
                            im_sr_all = im_sr_progress
                        else:
                            im_sr_all = torch.cat((im_sr_all, im_sr_progress), dim=1)
                    num_iters += 1
                    tt -= 1

                if 'gt' in data:
                    mean_psnr += util_image.batch_PSNR(
                            sample_decode['sample'] * 0.5 + 0.5,
                            im_gt * 0.5 + 0.5,
                            ycbcr=self.configs.train.val_y_channel,
                            )
                    mean_lpips += self.lpips_loss(
                            sample_decode['sample'] * 0.5 + 0.5,
                            im_gt * 0.5 + 0.5,
                            ).sum().item()

                if (ii + 1) % self.configs.train.log_freq[2] == 0:
                    self.logger.info(f'Validation: {ii+1:02d}/{num_iters_epoch:02d}...')

                    im_sr_all = rearrange(im_sr_all, 'b (k c) h w -> (b k) c h w', c=im_lq.shape[1])
                    self.logging_image(
                            im_sr_all,
                            tag='progress',
                            phase=phase,
                            add_global_step=False,
                            nrow=len(indices),
                            )
                    if 'gt' in data:
                        self.logging_image(im_gt, tag='gt', phase=phase, add_global_step=False)
                    self.logging_mask(diffusion_mask, phase=phase)
                    self.logging_image(im_lq, tag='lq', phase=phase, add_global_step=True)

            if 'gt' in data:
                mean_psnr /= len(self.datasets[phase])
                mean_lpips /= len(self.datasets[phase])
                self.logger.info(f'Validation Metric: PSNR={mean_psnr:5.2f}, LPIPS={mean_lpips:6.4f}...')
                if self.rank == 0 and self.wandb_logging:
                    self.wandb_run.log({"validation_step":self.log_step["val"], "PSNR": mean_psnr})
                    self.wandb_run.log({"validation_step":self.log_step["val"], "lpips": mean_lpips})
                self.logging_metric(mean_psnr, tag='PSNR', phase=phase, add_global_step=False)
                self.logging_metric(mean_lpips, tag='LPIPS', phase=phase, add_global_step=True)

            self.logger.info("="*100)

            if not (self.configs.train.use_ema_val and hasattr(self.configs.train, 'ema_rate')):
                self.model.train()

    def logging_mask(self, mask_tensor, phase, nrow=8):
        """
        Args:
            mask_tensor: b x h x w tensor
            phase: 'train' or 'val'
            nrow: number of displays in each row
        """
        assert self.tf_logging or self.local_logging
        mask_list = [mask_tensor[i] for i in range(mask_tensor.shape[0])]
        im_tensor = vutils.make_grid(mask_list, nrow=nrow, normalize=False, scale_each=False)  # c x H x W
        if self.local_logging:
            im_path = str(self.image_dir / phase / f"mask-{self.log_step_img[phase]}.png")
            grid_image = torchvision.transforms.ToPILImage()(im_tensor)
            grid_image.save(im_path)
        if self.tf_logging:
            self.writer.add_image(
                f"{phase}-mask-{self.log_step_img[phase]}",
                im_tensor,
                self.log_step_img[phase],
            )

class TrainerQDMLPIPS(TrainerDifIRLPIPS):
    def training_step(self, data):
        current_batchsize = data['gt'].shape[0]
        micro_batchsize = self.configs.train.microbatch
        num_grad_accumulate = math.ceil(current_batchsize / micro_batchsize)

        for jj in range(0, current_batchsize, micro_batchsize):
            micro_data = {key:value[jj:jj+micro_batchsize,] for key, value in data.items()}
            last_batch = (jj+micro_batchsize >= current_batchsize)
            tt = torch.randint(
                    0, self.base_diffusion.num_timesteps,
                    size=(micro_data['gt'].shape[0],),
                    device=f"cuda:{self.rank}",
            )

            if self.configs.autoencoder is not None:
                latent_downsamping_sf = 2**(len(self.configs.autoencoder.params.ddconfig.ch_mult) - 1)
            else:
                latent_downsamping_sf = 1
            latent_h, latent_w = micro_data['gt'].shape[-2:]
            latent_h, latent_w = latent_h // latent_downsamping_sf, latent_w // latent_downsamping_sf
            if self.configs.autoencoder is not None:
                noise_chn = self.configs.autoencoder.params.embed_dim
            else:
                noise_chn = micro_data['gt'].shape[1]
            noise = torch.randn(
                    size= (micro_data['gt'].shape[0], noise_chn,) + (latent_h, latent_w),
                    device=micro_data['gt'].device,
                    )
            
            if self.configs.model.params.cond_lq:
                model_kwargs = {'lq':micro_data['lq']}
            else:
                model_kwargs = None

            diffusion_mask = util_image.generate_quadtree_masks(
                micro_data['lq'],
                self.configs.train.threshold,
                latent_h,
                latent_w
            ).unsqueeze(1)
            model_mask = util_image.generate_quadtree_masks(
                micro_data['lq'],
                self.configs.train.threshold,
                latent_h//(self.configs.model.params.down_patch_size * self.configs.model.params.down_large_patch_size),
                latent_w//(self.configs.model.params.down_patch_size * self.configs.model.params.down_large_patch_size)
            )
            model_kwargs["mask"] = model_mask
            model_kwargs["up_pred"] = True # Default is True

            compute_losses = functools.partial(
                self.base_diffusion.training_losses,
                self.model,
                micro_data['gt'],
                micro_data['lq'],
                diffusion_mask,
                tt,
                first_stage_model=self.autoencoder,
                model_kwargs=model_kwargs,
                noise=noise,
            )
            if last_batch or self.num_gpus <= 1:
                losses, z0_pred, z0_up_pred, z_t = self.backward_step(compute_losses, micro_data, num_grad_accumulate, tt)
            else:
                with self.model.no_sync():
                    losses, z0_pred, z0_up_pred, z_t = self.backward_step(compute_losses, micro_data, num_grad_accumulate, tt)

            # make logging
            if last_batch:
                self.log_step_train(losses, tt, micro_data, z_t, z0_pred.detach(), z0_up_pred.detach())
                if self.rank == 0 and self.wandb_logging:
                    self.wandb_run.log(
                        {"loss": losses['loss'].mean(), 
                         "mse": losses['mse'].mean(), 
                         "mse_up": losses["mse_up"].mean(),
                         "lpips_down": losses["lpips"].mean(),
                        #  "lpips_up": losses["lpips_up"].mean()
                         })
                    
        if self.configs.train.use_amp:
            self.amp_scaler.step(self.optimizer)
            self.amp_scaler.update()
        else:
            self.optimizer.step()

        # grad zero
        self.model.zero_grad()

        if hasattr(self.configs.train, 'ema_rate'):
            self.update_ema_model()
    
    def backward_step(self, dif_loss_wrapper, micro_data, num_grad_accumulate, tt):
        context = torch.cuda.amp.autocast if self.configs.train.use_amp else nullcontext
        loss_coef = self.configs.train.get('loss_coef')
        stream_loss_coef = self.configs.train.get('stream_loss_coef')
        with context():
            losses, z_t, z0_pred, z0_up_pred = dif_loss_wrapper()            
            x0_pred = self.base_diffusion.decode_first_stage(
                z0_pred,
                self.autoencoder,
                ) # f16
            
            # x0_up_pred = self.base_diffusion.decode_first_stage(
            #     z0_up_pred,
            #     self.autoencoder,
            #     ) # f16
            
            self.current_x0_pred = x0_pred.detach()
            # self.current_x0_up_pred = x0_up_pred.detach()

            # classification loss
            losses["lpips"] = self.lpips_loss(
                    x0_pred,
                    micro_data['gt'],
                    ).to(z0_pred.dtype).view(-1)
            # losses["lpips_up"] = self.lpips_loss(
            #         x0_up_pred.clamp(-1.0, 1.0)/2 + 0.5,
            #         micro_data['gt']/2 + 0.5,
            #         ).to(z0_pred.dtype).view(-1).repeat(losses["mse"].shape[0])
            
            flag_nan = torch.any(torch.isnan(losses["lpips"]))
            if flag_nan:
                losses["lpips"] = torch.nan_to_num(losses["lpips"], nan=0.0)
            # flag_nan_up = torch.any(torch.isnan(losses["lpips_up"]))
            # if flag_nan_up:
            #     losses["lpips_up"] = torch.nan_to_num(losses["lpips_up"], nan=0.0)

            losses["mse"] *= loss_coef[0]
            # losses["mse"] = losses["mse"].mean(dim=0, keepdim=True)
            losses["lpips"] *= loss_coef[1]
            # assert losses["mse"].shape == losses["lpips"].shape
            if flag_nan:
                loss = losses["mse"]
            else:
                loss = losses["mse"] + losses["lpips"]

            # losses["mse_up"] *= loss_coef[0]
            # losses["mse_up"] = losses["mse_up"].mean(dim=0, keepdim=True)
            # losses["lpips_up"] *= loss_coef[1]
            # assert losses["mse_up"].shape == losses["lpips_up"].shape
            # if flag_nan_up:
            #     loss_up = losses["mse_up"]
            # else:
            #     loss_up = losses["mse_up"] + losses["lpips_up"]
            loss_up = losses["mse_up"]
            
            loss = loss * stream_loss_coef[0]
            loss_up = loss_up * stream_loss_coef[1]
            losses["loss"] = loss + loss_up
            loss = (loss + loss_up).mean() / num_grad_accumulate

        if self.amp_scaler is None:
            loss.backward()
        else:
            self.amp_scaler.scale(loss).backward()

        return losses, z0_pred, z0_up_pred, z_t

    def log_step_train(self, loss, tt, batch, z_t, z0_pred, z0_up_pred, phase='train'):
        '''
        param loss: a dict recording the loss informations
        param tt: 1-D tensor, time steps
        '''
        if self.rank == 0:
            num_timesteps = self.base_diffusion.num_timesteps
            record_steps = [1, (num_timesteps // 2) + 1, num_timesteps]
            if self.current_iters % self.configs.train.log_freq[0] == 1:
                self.loss_mean = {key:torch.zeros(size=(len(record_steps),), dtype=torch.float64)
                                  for key in loss.keys()}
                self.loss_count = torch.zeros(size=(len(record_steps),), dtype=torch.float64)
            for jj in range(len(record_steps)):
                for key, value in loss.items():
                    index = record_steps[jj] - 1
                    mask = torch.where(tt == index, torch.ones_like(tt), torch.zeros_like(tt))
                    assert value.shape == mask.shape
                    current_loss = torch.sum(value.detach() * mask)
                    self.loss_mean[key][jj] += current_loss.item()
                self.loss_count[jj] += mask.sum().item()

            if self.current_iters % self.configs.train.log_freq[0] == 0:
                if torch.any(self.loss_count == 0):
                    self.loss_count += 1e-4
                for key in loss.keys():
                    self.loss_mean[key] /= self.loss_count
                log_str = 'Train: {:06d}/{:06d}, MSE/LPIPS/MSE_UP: '.format(
                        self.current_iters,
                        self.configs.train.iterations)
                for jj, current_record in enumerate(record_steps):
                    log_str += 't({:d}):{:.1e}/{:.1e}/{:.1e}, '.format(
                            current_record,
                            self.loss_mean['mse'][jj].item(),
                            self.loss_mean['lpips'][jj].item(),
                            self.loss_mean['mse_up'][jj].item(),
                            # self.loss_mean['lpips_up'][jj].item(),
                            )
                log_str += 'lr:{:.2e}'.format(self.optimizer.param_groups[0]['lr'])
                self.logger.info(log_str)
                self.logging_metric(self.loss_mean, tag='Loss', phase=phase, add_global_step=True)
            if self.current_iters % self.configs.train.log_freq[1] == 0:
                self.logging_image(batch['lq'], tag='lq', phase=phase, add_global_step=False)
                self.logging_image(batch['gt'], tag='gt', phase=phase, add_global_step=False)
                x_t = self.base_diffusion.decode_first_stage(
                        self.base_diffusion._scale_input(z_t, tt),
                        self.autoencoder,
                        )
                self.logging_image(x_t, tag='diffused', phase=phase, add_global_step=False)
                self.logging_image(self.current_x0_pred, tag='x0-pred', phase=phase, add_global_step=False)
                micro_batchsize = 8
                with torch.no_grad():
                    decoded_chunks = []
                    for i in range(0, z0_up_pred.size(0), micro_batchsize):
                        z0_up_pred_chunk = z0_up_pred[i:i+micro_batchsize]

                        # Scale and decode the chunk
                        decoded_chunk = self.base_diffusion.decode_first_stage(
                            z0_up_pred_chunk,
                            self.autoencoder,
                        )

                        decoded_chunks.append(decoded_chunk)
                    x0_up_pred = torch.cat(decoded_chunks, dim=0)
                
                self.logging_image(x0_up_pred, tag='x0-up-pred', phase=phase, add_global_step=True)

            if self.current_iters % self.configs.train.save_freq == 1:
                self.tic = time.time()
            if self.current_iters % self.configs.train.save_freq == 0:
                self.toc = time.time()
                elaplsed = (self.toc - self.tic)
                self.logger.info(f"Elapsed time: {elaplsed:.2f}s")
                self.logger.info("="*100)
        
    def validation(self, phase='val'):
        if self.rank == 0:
            if self.configs.train.use_ema_val:
                self.reload_ema_model()
                self.ema_model.eval()
            else:
                self.model.eval()
            indices = np.linspace(
                    0,
                    self.base_diffusion.num_timesteps,
                    self.base_diffusion.num_timesteps if self.base_diffusion.num_timesteps < 5 else 4,
                    endpoint=False,
                    dtype=np.int64,
                    ).tolist()
            if not (self.base_diffusion.num_timesteps-1) in indices:
                indices.append(self.base_diffusion.num_timesteps-1)
            batch_size = self.configs.train.batch[1]
            num_iters_epoch = math.ceil(len(self.datasets[phase]) / batch_size)
            mean_psnr = mean_lpips = 0
            for ii, data in enumerate(self.dataloaders[phase]):
                data = self.prepare_data(data, phase='val')
                if 'gt' in data:
                    im_lq, im_gt = data['lq'], data['gt']
                else:
                    im_lq = data['lq']

                if self.configs.autoencoder is not None:
                    latent_downsamping_sf = 2**(len(self.configs.autoencoder.params.ddconfig.ch_mult) - 1)
                else:
                    latent_downsamping_sf = 1
                latent_h, latent_w = im_gt.shape[-2:]
                latent_h, latent_w = latent_h // latent_downsamping_sf, latent_w // latent_downsamping_sf
                
                diffusion_masks = util_image.generate_quadtree_masks(
                    im_lq,
                    self.configs.train.threshold,
                    latent_h,
                    latent_w
                ).unsqueeze(1)
                model_masks = util_image.generate_quadtree_masks(
                    im_lq,
                    self.configs.train.threshold,
                    latent_h//(self.configs.model.params.down_patch_size * self.configs.model.params.down_large_patch_size),
                    latent_w//(self.configs.model.params.down_patch_size * self.configs.model.params.down_large_patch_size)
                )

                if self.configs.model.params.cond_lq:
                    model_kwargs = {'lq':im_lq,}
                else:
                    model_kwargs = None
                model_kwargs["mask"] = model_masks
                model_kwargs["up_pred"] = False
                
                tt = torch.tensor(
                        [self.base_diffusion.num_timesteps, ]*im_lq.shape[0],
                        dtype=torch.int64,
                        ).cuda()
                
                num_iters = 0
                for sample in self.base_diffusion.p_sample_loop_progressive(
                        y=im_lq,
                        mask=diffusion_masks,
                        model=self.ema_model if self.configs.train.use_ema_val else self.model,
                        first_stage_model=self.autoencoder,
                        noise=None,
                        clip_denoised=True if self.autoencoder is None else False,
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
                        im_sr_progress = sample_decode['sample']
                        if num_iters + 1 == 1:
                            im_sr_all = im_sr_progress
                        else:
                            im_sr_all = torch.cat((im_sr_all, im_sr_progress), dim=1)
                    num_iters += 1
                    tt -= 1

                if 'gt' in data:
                    mean_psnr += util_image.batch_PSNR(
                            sample_decode['sample'] * 0.5 + 0.5,
                            im_gt * 0.5 + 0.5,
                            ycbcr=self.configs.train.val_y_channel,
                            )
                    mean_lpips += self.lpips_test_loss(
                            sample_decode['sample'] * 0.5 + 0.5,
                            im_gt * 0.5 + 0.5,
                            ).sum().item()
                    

                if (ii + 1) % self.configs.train.log_freq[2] == 0:
                    self.logger.info(f'Validation: {ii+1:02d}/{num_iters_epoch:02d}...')

                    im_sr_all = rearrange(im_sr_all, 'b (k c) h w -> (b k) c h w', c=im_lq.shape[1])
                    self.logging_image(
                            im_sr_all,
                            tag='progress',
                            phase=phase,
                            add_global_step=False,
                            nrow=len(indices),
                            )
                    if 'gt' in data:
                        self.logging_image(im_gt, tag='gt', phase=phase, add_global_step=False)
                    self.logging_mask(diffusion_masks, phase=phase)
                    self.logging_image(im_lq, tag='lq', phase=phase, add_global_step=True)

            if 'gt' in data:
                mean_psnr /= len(self.datasets[phase])
                mean_lpips /= len(self.datasets[phase])
                self.logger.info(f'Validation Metric: PSNR={mean_psnr:5.2f}, LPIPS={mean_lpips:6.4f}...')
                if self.rank == 0 and self.wandb_logging:
                    self.wandb_run.log({"validation_step":self.log_step["val"], "PSNR": mean_psnr})
                    self.wandb_run.log({"validation_step":self.log_step["val"], "lpips": mean_lpips})
                self.logging_metric(mean_psnr, tag='PSNR', phase=phase, add_global_step=False)
                self.logging_metric(mean_lpips, tag='LPIPS', phase=phase, add_global_step=True)

            self.logger.info("="*100)

            if not (self.configs.train.use_ema_val and hasattr(self.configs.train, 'ema_rate')):
                self.model.train()

    def logging_mask(self, mask_tensor, phase, nrow=8):
        """
        Args:
            mask_tensor: b x h x w tensor
            phase: 'train' or 'val'
            nrow: number of displays in each row
        """
        assert self.tf_logging or self.local_logging
        mask_list = [mask_tensor[i] for i in range(mask_tensor.shape[0])]
        im_tensor = vutils.make_grid(mask_list, nrow=nrow, normalize=False, scale_each=False)  # c x H x W
        if self.local_logging:
            im_path = str(self.image_dir / phase / f"mask-{self.log_step_img[phase]}.png")
            grid_image = torchvision.transforms.ToPILImage()(im_tensor)
            grid_image.save(im_path)
        if self.tf_logging:
            self.writer.add_image(
                f"{phase}-mask-{self.log_step_img[phase]}",
                im_tensor,
                self.log_step_img[phase],
            )

def replace_nan_in_batch(im_lq, im_gt):
    '''
    Input:
        im_lq, im_gt: b x c x h x w
    '''
    if torch.isnan(im_lq).sum() > 0:
        valid_index = []
        im_lq = im_lq.contiguous()
        for ii in range(im_lq.shape[0]):
            if torch.isnan(im_lq[ii,]).sum() == 0:
                valid_index.append(ii)
        assert len(valid_index) > 0
        im_lq, im_gt = im_lq[valid_index,], im_gt[valid_index,]
        flag = True
    else:
        flag = False
    return im_lq, im_gt, flag

def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
