# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import re

import yaml
import torch
from collections import OrderedDict

import datetime


def load_checkpoint(model: torch.nn.Module, path: str) -> dict:
    rank = int(os.environ.get('RANK', 0))
    logging.info('[Rank {}] Checkpoint: loading from checkpoint {}'.format(
        rank, path))
    checkpoint = torch.load(path, map_location='cpu', mmap=True)
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint,
                                                          strict=False)
    if rank == 0:
        for key in missing_keys:
            logging.info("missing tensor: {}".format(key))
        for key in unexpected_keys:
            logging.info("unexpected tensor: {}".format(key))
    info_path = re.sub('.pt$', '.yaml', path)
    configs = {}
    if os.path.exists(info_path):
        with open(info_path, 'r') as fin:
            configs = yaml.load(fin, Loader=yaml.FullLoader)
    return configs


def save_state_dict_and_infos(state_dict, path: str, infos=None):
    rank = int(os.environ.get('RANK', 0))
    logging.info('[Rank {}] Checkpoint: save to checkpoint {}'.format(
        rank, path))
    torch.save(state_dict, path)
    info_path = re.sub('.pt$', '.yaml', path)
    if infos is None:
        infos = {}
    infos['save_time'] = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
    with open(info_path, 'w') as fout:
        data = yaml.dump(infos)
        fout.write(data)


def save_checkpoint(model: torch.nn.Module, path: str, infos=None):
    '''
    Args:
        infos (dict or None): any info you want to save.
    '''
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    elif isinstance(model, torch.nn.parallel.DistributedDataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    save_state_dict_and_infos(state_dict, path, infos)


def filter_modules(model_state_dict, modules):
    rank = int(os.environ.get('RANK', 0))
    new_mods = []
    incorrect_mods = []
    mods_model = model_state_dict.keys()
    for mod in modules:
        if any(key.startswith(mod) for key in mods_model):
            new_mods += [mod]
        else:
            incorrect_mods += [mod]
    if incorrect_mods and rank == 0:
        logging.warning(
            "module(s) %s don't match or (partially match) "
            "available modules in model.",
            incorrect_mods,
        )
        logging.warning("for information, the existing modules in model are:")
        logging.warning("%s", mods_model)

    return new_mods


def load_trained_modules(model: torch.nn.Module, args: None):
    # Load encoder modules with pre-trained model(s).
    enc_model_path = args.enc_init
    enc_modules = args.enc_init_mods
    main_state_dict = model.state_dict()
    logging.warning("model(s) found for pre-initialization")
    if os.path.isfile(enc_model_path):
        logging.info('Checkpoint: loading from checkpoint %s for CPU' %
                     enc_model_path)
        model_state_dict = torch.load(enc_model_path, map_location='cpu')
        modules = filter_modules(model_state_dict, enc_modules)
        partial_state_dict = OrderedDict()
        for key, value in model_state_dict.items():
            if any(key.startswith(m) for m in modules):
                partial_state_dict[key] = value
        main_state_dict.update(partial_state_dict)
    else:
        logging.warning("model was not found : %s", enc_model_path)

    model.load_state_dict(main_state_dict)
    configs = {}
    return configs


# 以下是新添加的代码
def load_trained_model(model: torch.nn.Module, path: str, encoder_pretrain: int=0):
    # Load encoder modules with pre-trained model(s).
    main_state_dict = model.state_dict()
    partial_state_dict = OrderedDict()
    key_value_match = {'ShapMatch':[], 'ShapeMismatch':[], 'KeyNotFound':[]}
    print("model(s) found for pre-initialization")
    if os.path.isfile(path):
        print('Checkpoint:  %s ' % path)
        model_state_dict = torch.load(path, map_location='cpu')
        for key, value in model_state_dict.items():
            if encoder_pretrain == 1:
              key = 'encoder.' + key
              key = 'decoder.' + key
            elif encoder_pretrain == -1:
              key = key.replace('encoder.', '')
            elif encoder_pretrain == -2:
              key = key.replace('decoder.', '')
            elif encoder_pretrain == -3:
              key = key.replace('encoder.', '')
              key = key.replace('decoder.', '')
            if key in main_state_dict:
                if value.shape == main_state_dict[key].shape:
                    key_value_match['ShapMatch'] += [key]
                    partial_state_dict[key] = value
                # else:
                #     key_value_match['ShapeMismatch'] += [key]
                #     partial_state_dict[key] = main_state_dict[key]
                else:
                    shapes = main_state_dict[key].shape
                    if len(value.shape) == 1:
                        partial_state_dict[key] = value[:shapes[0]]
                    elif len(value.shape) == 2:
                        partial_state_dict[key] = value[:shapes[0], :shapes[1]]
                    else:
                        partial_state_dict[key] = main_state_dict[key]
            else:
                key_value_match['KeyNotFound'] += [key]
    else:
        print("model was not found : %s", path)
    
    print("%d Key(s) not found in model" % len(key_value_match['KeyNotFound']))
    print("%d Key(s) with mismatched shape" % len(key_value_match['ShapeMismatch']))
    print("%d Key(s) with matched shape" % len(key_value_match['ShapMatch']))

    model.load_state_dict(partial_state_dict, strict=False)
    configs = {}
    return configs


def migration(model, configs):
    if 'encoder_flag' in configs and configs['encoder_flag'] is not None:
        encoder_flag = configs['encoder_flag']
    else:
        encoder_flag = 0
    if 'checkpoint' in configs and configs['checkpoint'] is not None:
      checkpoint = configs['checkpoint']
      if os.path.exists(checkpoint):
        infos = load_checkpoint(model, checkpoint)
        print(f'Load checkpoint: {checkpoint}')
      else:
        infos = {}
        print(f'No such checkpoint: {checkpoint}')
    elif 'enc_init' in configs and configs['enc_init'] is not None:
        pretrain_model = configs['enc_init']
        infos = load_trained_model(model, pretrain_model, encoder_flag)
        print(f'Load pretrain model: {pretrain_model}')
    else:
        print('No checkpoint or pretrain model')
        infos = {}
    configs["init_infos"] = infos
    print(configs)
    return model