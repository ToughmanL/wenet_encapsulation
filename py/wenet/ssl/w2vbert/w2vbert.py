# -*- encoding: utf-8 -*-
'''
File       :w2vbert.py
Description:
Date       :2024/11/27
Author     :lxk
Version    :1.0
Contact		:xk.liu@siat.ac.cn
License		:GPL
'''


import torch

from typing import Tuple, Dict, List

from wenet.transformer.asr_model import ASRModel
from wenet.transformer.ctc import CTC
from wenet.ssl.w2vbert.encoder import W2VBERTEncoder
from wenet.transformer.decoder import TransformerDecoder
from wenet.utils.common import IGNORE_ID, add_whisper_tokens, th_accuracy


class W2VBERTModel(ASRModel):

    def __init__(
        self,
        vocab_size: int,
        encoder: W2VBERTEncoder,
        decoder: TransformerDecoder,
        ctc: CTC = None,
        ctc_weight: float = 0.5,
        ignore_id: int = IGNORE_ID,
        reverse_weight: float = 0.0,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        special_tokens: dict = None,
    ):
        super().__init__(vocab_size, encoder, decoder, ctc, ctc_weight,
                         ignore_id, reverse_weight, lsm_weight,
                         length_normalized_loss, special_tokens)
        assert reverse_weight == 0.0
        self.sos = special_tokens["sot"]
        self.eos = special_tokens["eot"]
        self.decode_maxlen = self.decoder.embed[1].max_len

    # TODO(xcsong): time align
    def set_alignment_heads(self, dump: bytes):
        raise NotImplementedError

    @property
    def is_multilingual(self):
        return self.vocab_size >= 51865

    @property
    def num_languages(self):
        return self.vocab_size - 51765 - int(self.is_multilingual)

    def _calc_att_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_mask: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
        infos: Dict[str, List[str]],
    ) -> Tuple[torch.Tensor, float]:
        prev_len = ys_pad.size(1)
        ys_in_pad, ys_out_pad = add_whisper_tokens(self.special_tokens,
                                                   ys_pad,
                                                   self.ignore_id,
                                                   tasks=infos['tasks'],
                                                   no_timestamp=True,
                                                   langs=infos['langs'],
                                                   use_prev=False)
        cur_len = ys_in_pad.size(1)
        ys_in_lens = ys_pad_lens + cur_len - prev_len

        # 1. Forward decoder
        decoder_out, r_decoder_out, _ = self.decoder(encoder_out, encoder_mask,
                                                     ys_in_pad, ys_in_lens)

        # 2. Compute attention loss
        loss_att = self.criterion_att(decoder_out, ys_out_pad)
        acc_att = th_accuracy(
            decoder_out.view(-1, self.vocab_size),
            ys_out_pad,
            ignore_label=self.ignore_id,
        )
        return loss_att, acc_att

