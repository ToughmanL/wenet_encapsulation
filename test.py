# -*- encoding: utf-8 -*-
'''
File       :test.py
Description:
Date       :2024/12/20
Author     :lxk
Version    :1.0
Contact		:xk.liu@siat.ac.cn
License		:GPL
'''

import wenet_infer
import torch 
torch.backends.cudnn.enabled = False
model_dir = "conformer_179"
model_object = wenet_infer.load_model(model_dir, gpu=0, beam=10)
audio_file = "test.wav"
result = model_object.transcribe(audio_file)
print(result[0])