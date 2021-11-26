# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications: Remove auxiliary models

import argparse

import torch

# import subprocess


def parse_args():
    parser = argparse.ArgumentParser(
        description='Process a checkpoint to be published')
    parser.add_argument('in_file', help='input checkpoint filename')
    parser.add_argument('out_file', help='output checkpoint filename')
    args = parser.parse_args()
    return args


def process_checkpoint(in_file, out_file):
    checkpoint = torch.load(in_file, map_location='cpu')
    # remove optimizer for smaller file size
    if 'optimizer' in checkpoint:
        del checkpoint['optimizer']
    # remove auxiliary models
    for k in list(checkpoint['state_dict'].keys()):
        if 'imnet_model' in k or 'ema_model' in k:
            del checkpoint['state_dict'][k]
    # if it is necessary to remove some sensitive data in checkpoint['meta'],
    # add the code here.
    if 'meta' in checkpoint:
        del checkpoint['meta']
    # inspect checkpoint
    print('Checkpoint keys:', checkpoint.keys())
    print('Checkpoint state_dict keys:', checkpoint['state_dict'].keys())
    # save checkpoint
    torch.save(checkpoint, out_file)
    # sha = subprocess.check_output(['sha256sum', out_file]).decode()
    # final_file = out_file.rstrip('.pth') + '-{}.pth'.format(sha[:8])
    # subprocess.Popen(['mv', out_file, final_file])


def main():
    args = parse_args()
    process_checkpoint(args.in_file, args.out_file)


if __name__ == '__main__':
    main()
