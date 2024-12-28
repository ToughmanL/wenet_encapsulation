# Copyright (c) 2023 Binbin Zhang (binbzha@qq.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import io
import yaml
import copy
import logging

import torch
import librosa
import torchaudio
import torchaudio.compliance.kaldi as kaldi

from wenet.dataset.dataset import Dataset
from wenet.utils.config import override_config
from wenet.utils.init_model import init_model
from wenet.utils.init_tokenizer import init_tokenizer
from wenet.utils.context_graph import ContextGraph
from wenet.utils.ctc_utils import get_blank_id

from argparse import Namespace

args = Namespace(config='config.ini',
    test_data='test_data.csv',
    data_type='raw',
    gpu=-1,
    dtype='fp32',
    num_workers=0,
    checkpoint='model.ckpt',
    beam_size=10,
    length_penalty=0.0,
    blank_penalty=0.0,
    result_dir='results/',
    batch_size=16,
    # modes=['ctc_prefix_beam_search', 'attention', 'attention_rescoring'],
    modes=['attention'],
    search_ctc_weight=1.0,
    search_transducer_weight=0.0,
    ctc_weight=0.5,
    transducer_weight=0.0,
    attn_weight=0.0,
    decoding_chunk_size=-1,
    num_decoding_left_chunks=-1,
    simulate_streaming=False,
    reverse_weight=0.0,
    override_config=[],
    word='',
    hlg='',
    lm_scale=0.0,
    decoder_scale=0.0,
    r_decoder_scale=0.0,
    context_bias_mode='',
    context_list_path='',
    context_graph_score=0.0,
    use_lora=False,
    debug=False)

class Model:
  def __init__(self,
        model_dir: str,
        gpu: int = 0,
        beam: int = 10,
        resample_rate: int = 16000):
    config = os.path.join(model_dir, "train.yaml")
    model_path = os.path.join(model_dir, "model.pt")
    
    self.resample_rate = resample_rate
    with open(config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    if configs.get('cmvn') is not None:
        cmvn_path = os.path.join(model_dir, "global_cmvn")
        configs['cmvn_conf']['cmvn_file'] = cmvn_path
    
    if configs['tokenizer'] != 'whisper':
        bpe_path = os.path.join(model_dir, "bpe.model")
        units_path = os.path.join(model_dir, "units.txt")
        configs['tokenizer_conf']['bpe_path'] = bpe_path
        configs['tokenizer_conf']['symbol_table_path'] = units_path

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    use_cuda = gpu >= 0 and torch.cuda.is_available()
    self.device = torch.device('cuda' if use_cuda else 'cpu')

    self.tokenizer = init_tokenizer(configs)
    setattr(args, 'config', config)
    setattr(args, 'checkpoint', model_path)

    self.model, configs = init_model(args, configs)
    self.model = self.model.to(self.device)
    self.model.eval()

    self.dtype = torch.float32
    if args.dtype == 'fp16':
        self.dtype = torch.float16
    elif args.dtype == 'bf16':
        self.dtype = torch.bfloat16

    context_graph = None
    _, self.blank_id = get_blank_id(configs, self.tokenizer.symbol_table)
    self.feat_type = configs['dataset_conf']['feats_type']

  def compute_feats(self, audio_file: str) -> torch.Tensor:
    with open(audio_file, 'rb') as fin:
        wav_file = fin.read()
    with io.BytesIO(wav_file) as file_obj:
        waveform, sample_rate = torchaudio.load(file_obj)    
    if sample_rate != self.resample_rate:
        waveform = torchaudio.transforms.Resample(
        orig_freq=sample_rate, new_freq=self.resample_rate)(waveform)
    if self.feat_type == 'fbank':
        waveform = waveform * (1 << 15)
        feats = kaldi.fbank(waveform,
            num_mel_bins=80,
            frame_length=25,
            frame_shift=10,
            energy_floor=0.0,
            sample_frequency=self.resample_rate)
        feats = feats.unsqueeze(0)
    elif self.feat_type == 'log_mel_spectrogram':
        waveform = waveform.squeeze(0)
        n_fft = 400
        hop_length = 160
        num_mel_bins = 128
        window = torch.hann_window(n_fft)
        stft = torch.stft(waveform,
                n_fft,
                hop_length,
                window=window,
                return_complex=True)
        magnitudes = stft[..., :-1].abs()**2
        filters = torch.from_numpy(librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=num_mel_bins))
        mel_spec = filters @ magnitudes
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        feats = log_spec.transpose(0, 1)
        feats = feats.unsqueeze(0)
    feats_length = torch.tensor([feats.size(1)], dtype=torch.int32)
    return feats, feats_length

  @torch.no_grad()
  def _decode(self,
    audio_file: str,
    tokens_info: bool = False) -> dict:
    context_graph = None

    with torch.cuda.amp.autocast(enabled=True, dtype=self.dtype, cache_enabled=False):
        feats, feats_lengths = self.compute_feats(audio_file)
        feats = feats.to(self.device)
        feats_lengths = feats_lengths.to(self.device)
        infos = {"tasks": ["transcribe"], "langs": ["en"]}
        results = self.model.decode(
            args.modes,
            feats,
            feats_lengths,
            args.beam_size,
            decoding_chunk_size=args.decoding_chunk_size,
            num_decoding_left_chunks=args.num_decoding_left_chunks,
            ctc_weight=args.ctc_weight,
            simulate_streaming=args.simulate_streaming,
            reverse_weight=args.reverse_weight,
            context_graph=context_graph,
            blank_id=self.blank_id,
            blank_penalty=args.blank_penalty,
            length_penalty=args.length_penalty,
            infos=infos)
        hyps = results['attention']
        tokens = hyps[0].tokens
        confidence = hyps[0].confidence
        tokens_confidence = hyps[0].tokens_confidence
        result = self.tokenizer.detokenize(tokens)
    return result

  def transcribe(self, audio_file: str, tokens_info: bool = False) -> dict:
    return self._decode(audio_file, tokens_info)


def load_model(model_dir: str = None,
     gpu: int = 0,
     beam: int = 10) -> Model:
  return Model(model_dir, gpu, beam)


if __name__ == '__main__':
  model_dir = "conformer_179"
  model = load_model(model_dir, gpu=0, beam=10)
  audio_file = "test.wav"
  result = model.transcribe(audio_file)
  print(result)