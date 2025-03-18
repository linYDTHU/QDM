#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse

from omegaconf import OmegaConf

from utils import util_common

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("-i", "--in_path", type=str, default="", help="Input path.")
    parser.add_argument("-o", "--out_path", type=str, default="./results", help="Output path.")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed.")
    parser.add_argument("--chop_bs", type=int, default=1, help="Chopping Batch size.")
    parser.add_argument("--chop_size", type=int, default=128, help="Chopping Batch size.")
    parser.add_argument(
        "--cfg_path",
        type=str,
        default=None,
        help="Your own custom config path, default is None",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Your own custom checkpoint path, default is None",
    )
    parser.add_argument('--process', action='store_true', help='save process')
    parser.add_argument('--distributed', action='store_true', help='distributed inference')
    args = parser.parse_args()

    return args

def main():
    args = get_parser()
    configs = OmegaConf.load(args.cfg_path)
    OmegaConf.update(configs, 'ckpt_path', args.ckpt_path)

    if args.chop_size==128:
        sampler = util_common.get_obj_from_str(configs.sampler.target)(
            configs,
            in_path=args.in_path,
            out_path=args.out_path,
            chop_bs=args.chop_bs,
            chop_size=args.chop_size,
            chop_stride=112,
            seed=args.seed,
            distributed=args.distributed,
        )
    elif args.chop_size==64:
        sampler = util_common.get_obj_from_str(configs.sampler.target)(
                    configs,
                    in_path=args.in_path,
                    out_path=args.out_path,
                    chop_bs=args.chop_bs,
                    chop_size=args.chop_size,
                    chop_stride=48,
                    seed=args.seed,
                    distributed=args.distributed,
                )
    else:
        raise ValueError("Chop size must be 128 or 64")

    if not args.process:
        sampler.inference()
    else: 
        sampler.inference_process()

if __name__ == '__main__':
    main()
