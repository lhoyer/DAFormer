# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0

import argparse
import json
import logging
from copy import deepcopy

from experiments import generate_experiment_cfgs
from mmcv import Config, get_logger
from prettytable import PrettyTable

from mmseg.models import build_segmentor


def human_format(num):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    # add more suffixes if you need them
    return '%.2f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])


def count_parameters(model):
    table = PrettyTable(['Modules', 'Parameters'])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, human_format(param)])
        total_params += param
    # print(table)
    print(f'Total Trainable Params: {human_format(total_params)}')
    return total_params


# Run: python -m tools.param_count
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--exp',
        nargs='?',
        type=int,
        default=100,
        help='Experiment id as defined in experiment.py',
    )
    args = parser.parse_args()
    get_logger('mmseg', log_level=logging.ERROR)
    cfgs = generate_experiment_cfgs(args.exp)
    for cfg in cfgs:
        with open('configs/tmp_param.json', 'w') as f:
            json.dump(cfg, f)
        cfg = Config.fromfile('configs/tmp_param.json')

        model = build_segmentor(deepcopy(cfg['model']))
        # model.init_weights()
        # count_parameters(model)
        print(f'Encoder {cfg["name_encoder"]}:')
        count_parameters(model.backbone)
        print(f'Decoder {cfg["name_decoder"]}:')
        count_parameters(model.decode_head)
