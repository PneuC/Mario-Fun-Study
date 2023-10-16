"""
  @Time : 2021/9/16 20:40 
  @Author : Ziqi Wang
  @File : sac_train.py
  Script for training models (generator/designer/cnet):
  Usage:
    python train.py [model] --args, [model] is one of {generator, designer, cnet}
"""

import argparse
from src.gan import adversarial_train
from src.rl import sac_train
from src.repair import cnet

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    parser_gan = subparsers.add_parser('generator', help='Train GAN generator')
    parser_d = subparsers.add_parser('designer')
    parser_cnet = subparsers.add_parser('cnet')

    adversarial_train.set_parser(parser_gan)
    parser_gan.set_defaults(entry=adversarial_train.train_gan)

    sac_train.set_parser(parser_d)
    parser_d.set_defaults(entry=sac_train.train_designer)

    cnet.set_parser(parser_cnet)
    parser_cnet.set_defaults(entry=cnet.train_cnet)

    args = parser.parse_args()
    entry = args.entry
    entry(args)
