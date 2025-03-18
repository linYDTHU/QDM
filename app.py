#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Modified by Donglin Yang

import warnings
warnings.filterwarnings("ignore")

import argparse
import numpy as np
import gradio as gr
from pathlib import Path
from omegaconf import OmegaConf
from sampler import QDMSampler
import os
from tqdm import tqdm

from utils import util_common
from utils import util_image
from basicsr.utils.download_util import load_file_from_url

def get_configs(task="Real-world SR X4", threshold=0.00, mask_forward=True, chunk_size=64):
    if task == "Real-world SR X4":
    
        configs = OmegaConf.load("./configs/inference/realsr_qdm_l.yaml")

        autoencoder_ckpt_name = "autoencoder_vq_f4.pth"
        autoencoder_ckpt_dir = "./weights"
        util_common.mkdir(autoencoder_ckpt_dir, delete=False, parents=True)
        autoencoder_ckpt_path = Path(autoencoder_ckpt_dir) / autoencoder_ckpt_name
        if not autoencoder_ckpt_path.exists():
            load_file_from_url(
                url="https://github.com/zsyOAOA/ResShift/releases/download/v2.0/autoencoder_vq_f4.pth", 
                model_dir=model_ckpt_dir,
                progress=True,
                file_name=model_ckpt_name,
            )

        model_ckpt_name = "realsr_qdm_l.pth"
        model_ckpt_dir = "./weights"
        util_common.mkdir(model_ckpt_dir, delete=False, parents=True)
        model_ckpt_path = Path(model_ckpt_dir) / model_ckpt_name
        if not model_ckpt_path.exists():
            load_file_from_url(
                url="",  #todo: add download link
                model_dir=model_ckpt_dir,
                progress=True,
                file_name=model_ckpt_name,
            )
        configs.ckpt_path = str(model_ckpt_path)

        configs.threshold = threshold
        configs.mask_forward = mask_forward
        configs.chunk_size = chunk_size
        configs.chop_size = 128
        configs.chop_stride = 112
    elif task == "Medical SR X4":
        configs = OmegaConf.load("./configs/inference/medx4_qdm_l.yaml")

        autoencoder_ckpt_name = "medical_autoencoder.ckpt"
        autoencoder_ckpt_dir = "./weights"
        util_common.mkdir(autoencoder_ckpt_dir, delete=False, parents=True)
        autoencoder_ckpt_path = Path(autoencoder_ckpt_dir) / autoencoder_ckpt_name
        if not autoencoder_ckpt_path.exists():
            load_file_from_url(
                url="",  #todo: add download link
                model_dir=model_ckpt_dir,
                progress=True,
                file_name=model_ckpt_name,
            )

        model_ckpt_name = "medx4_qdm_l.pth"
        model_ckpt_dir = "./weights"
        util_common.mkdir(model_ckpt_dir, delete=False, parents=True)
        model_ckpt_path = Path(model_ckpt_dir) / model_ckpt_name
        if not model_ckpt_path.exists():
            load_file_from_url(
                url="",  #todo: add download link
                model_dir=model_ckpt_dir,
                progress=True,
                file_name=model_ckpt_name,
            )
        configs.ckpt_path = str(model_ckpt_path)

        configs.threshold = threshold
        configs.mask_forward = mask_forward
        configs.chunk_size = chunk_size
        configs.chop_size = 128
        configs.chop_stride = 112
    elif task == "Medical SR X8":
        configs = OmegaConf.load("./configs/inference/medx8_qdm_l.yaml")

        autoencoder_ckpt_name = "medical_autoencoder.ckpt"
        autoencoder_ckpt_dir = "./weights"
        util_common.mkdir(autoencoder_ckpt_dir, delete=False, parents=True)
        autoencoder_ckpt_path = Path(autoencoder_ckpt_dir) / autoencoder_ckpt_name
        if not autoencoder_ckpt_path.exists():
            load_file_from_url(
                url="",  #todo: add download link
                model_dir=model_ckpt_dir,
                progress=True,
                file_name=model_ckpt_name,
            )

        model_ckpt_name = "medx8_qdm_l.pth"
        model_ckpt_dir = "./weights"
        util_common.mkdir(model_ckpt_dir, delete=False, parents=True)
        model_ckpt_path = Path(model_ckpt_dir) / model_ckpt_name
        if not model_ckpt_path.exists():
            load_file_from_url(
                url="",  #todo: add download link
                model_dir=model_ckpt_dir,
                progress=True,
                file_name=model_ckpt_name,
            )
        configs.ckpt_path = str(model_ckpt_path)

        configs.threshold = threshold
        configs.mask_forward = mask_forward
        configs.chunk_size = chunk_size
        configs.chop_size = 64
        configs.chop_stride = 48
    else:
        raise ValueError(f"Task {task} is not supported.")
    return configs

def predict_single(in_path, task="Real-world SR X4", threshold=0.00, mask_forward=True, chunk_size=64, chop_bs=1, seed=12345):
    configs = get_configs(task, threshold, mask_forward, chunk_size)
    output_path = 'quadtree_output'
    sampler = QDMSampler(configs, in_path, output_path, chop_size=configs.chop_size, chop_stride=configs.chop_stride, chop_bs=chop_bs, seed=seed)

    out_dir = Path(output_path)
    if not out_dir.exists():
        out_dir.mkdir()
    sampler.inference()

    out_path = out_dir / f"{Path(in_path).stem}.png"
    assert out_path.exists(), 'Super-resolution failed!'
    im_sr = util_image.imread(out_path, chn="rgb", dtype="uint8")

    return im_sr, str(out_path)

title = "QDM: Quadtree-Based Region-Adaptive Sparse Diffusion Models for Efficient Image Super-Resolution"

article = r"""
If you've found QDM useful for your research or projects, please show your support by ⭐ the <a href='https://github.com/linYDTHU/QDM' target='_blank'>Github Repo</a>. Thanks!
[![GitHub Stars](https://img.shields.io/github/stars/linYDTHU/QDM?affiliations=OWNER&color=green&style=social)](https://github.com/linYDTHU/QDM)
---
If our work is useful for your research, please consider citing:
```bibtex

```
📋 **License**
This project is licensed under <a rel="license" href="https://github.com/linYDTHU/QDM/master/LICENSE">MIT License</a>.
Redistribution and use for non-commercial purposes should follow this license.
📧 **Contact**
If you have any questions, please feel free to contact me via <b>ydlin718@gmail.com</b>.
![visitors](https://visitor-badge.laobi.icu/badge?page_id=linYDTHU/QDM)
"""
description = r"""
<b>Official Gradio demo</b> for <a href='https://github.com/linYDTHU/QDM' target='_blank'><b>QDM: Quadtree-Based Region-Adaptive Sparse Diffusion Models for Efficient Image Super-Resolution</b></a>.<br>
🔥 QDM is a region-adaptive diffusion SR framework.<br>
"""

with gr.Blocks() as demo:
    gr.Markdown(f"# {title}")
    gr.Markdown(description)

    with gr.Tabs():
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="filepath", label="Input: Low Quality Image")
                task = gr.Dropdown(
                    choices=["Real-world SR X4","Medical SR X4","Medical SR X8"],
                    value="Real-world SR X4",
                    label="Task",
                )
                threshold = gr.Number(value=0.00, minimum=0.00, label="Quadtree threshold")
                mask_forward=gr.Checkbox(label="Mask forward")
                chunk_size = gr.Number(value=64, minimum=1, precision=0, label="Chunk size")
                chop_bs = gr.Number(value=1, minimum=1, precision=0, label="Chopping batch size(Adjust small when OOM)")
                seed = gr.Number(value=12345, precision=0, label="Random seed")
                process_btn = gr.Button("Process")
            with gr.Column():
                output_image = gr.Image(type="numpy", label="Output: High Quality Image")
                output_file = gr.File(label="Download the output")
        process_btn.click(
            fn=predict_single,
            inputs=[input_image, task, threshold, mask_forward, chunk_size, chop_bs, seed],
            outputs=[output_image, output_file]
        )

    gr.Markdown(article)

demo.queue(max_size=5)
demo.launch(share=True)