# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications: Add code archive generation

import os
import tarfile

from mmcv.utils import collect_env as collect_base_env
from mmcv.utils import get_git_hash

import mmseg


def collect_env():
    """Collect the information of the running environments."""
    env_info = collect_base_env()
    env_info['MMSegmentation'] = f'{mmseg.__version__}+{get_git_hash()[:7]}'

    return env_info


def is_source_file(x):
    if x.isdir() or x.name.endswith(('.py', '.sh', '.yml', '.json', '.txt')) \
            and '.mim' not in x.name and 'jobs/' not in x.name:
        # print(x.name)
        return x
    else:
        return None


def gen_code_archive(out_dir, file='code.tar.gz'):
    archive = os.path.join(out_dir, file)
    os.makedirs(os.path.dirname(archive), exist_ok=True)
    with tarfile.open(archive, mode='w:gz') as tar:
        tar.add('.', filter=is_source_file)
    return archive


if __name__ == '__main__':
    for name, val in collect_env().items():
        print('{}: {}'.format(name, val))
