# QDM: Quadtree-Based Region-Adaptive Sparse Diffusion Models for Efficient Image Super-Resolution
<div align="center">
  <a href="https://arxiv.org/abs/2503.12015">
    <img src="https://img.shields.io/badge/arXiv-2503.12015-b31b1b.svg" alt="arXiv">
  </a>
  <img src="https://visitor-badge.laobi.icu/badge?page_id=linYDTHU/QDM" alt="visitors">
  <a href="https://github.com/linYDTHU/QDM">
    <img src="https://img.shields.io/github/stars/linYDTHU/QDM?affiliations=OWNER&color=green&style=social" alt="GitHub Stars">
  </a>
</div>


If you've found QDM useful for your research or projects, please show your support by ⭐ in this repo. Thanks!

---
>Deep learning-based super-resolution (SR) methods often perform pixel-wise computations uniformly across entire images, even in homogeneous regions where high-resolution refinement is redundant. 
We propose the Quadtree Diffusion Model (QDM), a region-adaptive diffusion framework that leverages a quadtree structure to selectively enhance detail-rich regions while reducing computations in homogeneous areas.
By guiding the diffusion with a quadtree derived from the low-quality input, QDM identifies key regions—represented by leaf nodes—where fine detail is essential and applies minimal refinement elsewhere.
This mask-guided, two-stream architecture adaptively balances quality and efficiency, producing high-fidelity outputs with low computational redundancy.
Experiments demonstrate QDM’s effectiveness in high-resolution SR tasks across diverse image types, particularly in medical imaging (e.g., CT scans), where large homogeneous regions are prevalent.
Furthermore, QDM outperforms or is comparable to state-of-the-art SR methods on standard benchmarks while significantly reducing computational costs, highlighting its efficiency and suitability for resource-limited environments.
><img src="./assets/Quadtree_Diagram.png" align="middle" width="1000">

---
## Update
- **2025.11.18**: Released a new arXiv version with tumor region reconstruction and real-world SR results. Refer to the paper for details. Use `print_roi_metrics.py` to replicate the tumor reconstruction results. Access results for all methods [here](https://drive.google.com/file/d/1IjmAPvqfPqdxwjRv0P9ulSdlMu4ZM_Ua/view?usp=sharing). Updated real-world SR with Gaussian-weighted patch-level aggregation as per [this reference](https://github.com/zsyOAOA/InvSR/blob/master/utils/util_image.py#L904) in `utils/util_image.py`.
- **2025.03.18**: Release codes & pretrained checkpoints, and update README.
- **2025.03.14**: Create this repo.

## Requirements
* More detail (See [requirements.txt](requirements.txt))
A suitable [conda](https://conda.io/) environment named `quadtree_diffusion` can be created and activated with:

```
conda create -n quadtree_diffusion python=3.10
conda activate quadtree_diffusion
pip install -r requirements.txt
```

## Examples
### Real-World Image Super-Resolution
[<img src="assets/realsr_1.png" height="330px"/>](https://imgsli.com/MzQ5ODk4) [<img src="assets/realsr_2.png" height="330px"/>](https://imgsli.com/MzUxMTc0) [<img src="assets/realsr_5.png" height="330px"/>](https://imgsli.com/MzYwMDQ3)

[<img src="assets/realsr_3.png" height="320px"/>](https://imgsli.com/MzUxMTcz) [<img src="assets/realsr_4.png" height="320px"/>](https://imgsli.com/MzQ5OTA2) [<img src="assets/realsr_6.png" height="320px"/>](https://imgsli.com/MzYwMDUx) 

### Medical Image Super-Resolution
[<img src="assets/medx8_sr_1.png" height="330"/>](https://imgsli.com/MzYwMDAy) [<img src="assets/medx8_sr_2.png" height="330px"/>](https://imgsli.com/MzYwMDA1) [<img src="assets/medx8_sr_3.png" height="330px"/>](https://imgsli.com/MzYwMDA2) 

## Fast Testing Guide

### Download Pretrained Checkpoints

#### First-Stage Models (Autoencoders)
1. ​**Real-world SR Task**: [Download Link](https://github.com/zsyOAOA/ResShift/releases/download/v2.0/autoencoder_vq_f4.pth)
2. ​**Medical SR Task**: [Download Link](https://drive.google.com/file/d/177wL116e495OTxxaBeXZh9c0Wz7_slHU/view?usp=sharing)
   
​**Note**: Place the downloaded models in the `weights` directory.

### QDM-L Checkpoints
We provide pretrained checkpoints for the QDM-L model for the following tasks:
- [Real-world SR X4](https://drive.google.com/file/d/1N30YnuhBYkjOC0K9igV0ixGj9-Wtu1b3/view?usp=sharing)
- [Medical SR X4](https://drive.google.com/file/d/165iWcLFRPmIZPBDygiaeK6iKmMDCTlN6/view?usp=sharing)
- [Medical SR X8](https://drive.google.com/file/d/1c9eL54BdwUcr-YWYFFLUwkPDJ_qQPYZz/view?usp=sharing)
  
​**Note**: Ensure all downloaded weights are placed in the `weights` directory.

### Inference

#### 🚀 Multi-GPU Acceleration
If you have multiple GPUs available, you can accelerate the inference process using the following command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 --nnodes=1 inference.py \
  -i [Input Directory or Image] \
  -o [Output Dir] \
  --seed [Seed] \
  --chop_bs [Chopping Batch Size] \
  --chop_size [Chopping Size] \
  --cfg_path [Config Path] \
  --ckpt_path [Checkpoint Path] \
  --distributed
```
#### 💻 Single-GPU Execution
```bash
python inference.py \
  -i [Input Directory or Image] \
  -o [Output Dir] \
  --seed [Seed] \
  --chop_bs [Chopping Batch Size] \
  --chop_size [Chopping Size] \
  --cfg_path [Config Path] \
  --ckpt_path [Checkpoint Path]
```
#### 🔧Configuration Tips
- When processing very large images, you can adjust `--chop_bs` to balance efficiency and memory usage.
- We provide multiple configuration files for different tasks in the `configs/inference` directory. **​Make sure to select the appropriate configuration file for your specific task.**
- You can add `--process` argument to output the mask guided diffusion process demonstrated in the paper.

<img src="./assets/Diffusion_Process.png" align="middle" width="1000">

## Training
### Preparing Stage

This repository supports two super-resolution (SR) tasks: **Real-World SR** and **Medical CT SR**. Follow the steps below to prepare the necessary training and testing datasets.

#### Real-World SR Task

We integrate training data from six established benchmarks:

- **LSDIR** – [Access Dataset](https://huggingface.co/ofsoundof/LSDIR)
- **DIV2K** – [Access Dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/)
- **DIV8K** – [Access Dataset](https://huggingface.co/datasets/yangtao9009/DIV8K)
- **OutdoorSceneTraining** – [Access Dataset](https://drive.google.com/drive/folders/16PIViLkv4WsXk4fV1gDHvEtQxdMq6nfY)
- **Flicker2K** – [Access Dataset](https://huggingface.co/datasets/goodfellowliu/Flickr2K)
- **FFHQ Subset** – A curated selection of 10,000 facial images from the [FFHQ](https://github.com/NVlabs/ffhq-dataset) dataset

#### Preprocessing Steps

- **Filtering OutdoorSceneTraining:**  
  Filter out images with spatial dimensions smaller than 512 pixels. Update the directory path inside the script as needed, then run:
  ```bash
  python scripts/filter_images.py
  ```
- **Synthetic LSDIR_TEST:**  
  Download the pre-synthesized LSDIR_TEST dataset from [this link](https://drive.google.com/file/d/1IhGtO6niw7IPK_m4QagB3msV5SMBI0B8/view?usp=sharing) or generate your own by running:
  ```bash
  python scripts/prepare_lsdir_test.py
  ```

#### Medical CT SR Task

For the medical CT super-resolution task, we utilize clinical CT scans from two well-established segmentation challenges: **HaN-Seg** and **SegRap2023**. Download the datasets using the following links:

- Training Set: [Download](https://drive.google.com/file/d/1L91BY58fRQ8JBzpDc7_P9ffhH6SOb2Zc/view?usp=sharing)
- Medx4 Testing Set: [Download](https://drive.google.com/file/d/1wcmxRjcdoeLZE8lOE3e-8oaq6R0SVYin/view?usp=sharing)
- Medx8 Testing Set: [Download](https://drive.google.com/file/d/1WxHZ-V3tZrNHBqaI6Whxn6HDPDmk56kl/view?usp=sharing)

### Training Scripts
You can start your training process via running:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 --nnodes=1 main.py --cfg_path [Config Path] --save_dir [Logging Folder]
```
We provide multiple configuration files for different tasks in the `configs/train` directory. ​

## Citations
Please consider citing our paper in your publications if it helps. Here is the bibtex:

```
@misc{yang2025qdmquadtreebasedregionadaptivesparse,
      title={QDM: Quadtree-Based Region-Adaptive Sparse Diffusion Models for Efficient Image Super-Resolution}, 
      author={Donglin Yang and Paul Vicol and Xiaojuan Qi and Renjie Liao and Xiaofan Zhang},
      year={2025},
      eprint={2503.12015},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.12015}, 
}
```

## License

This project is licensed under <a rel="license" href="./LICENSE">MIT License</a>. Redistribution and use should follow this license.

## Acknowledgement

This project is primarily based on [ResShift](https://github.com/zsyOAOA/ResShift) and [LDM](https://github.com/CompVis/latent-diffusion). We also adopt [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) to synthesize the LR/HR pairs. We design QDM mainly based on [DiT](https://github.com/facebookresearch/DiT). Thanks for their awesome works.

### Contact
If you have any questions, please feel free to contact me via `ydlin718@gmail.com`.
