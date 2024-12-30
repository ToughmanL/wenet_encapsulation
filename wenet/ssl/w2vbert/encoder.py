# -*- encoding: utf-8 -*-
'''
File       :encoder.py
Description:
Date       :2024/11/27
Author     :lxk
Version    :1.0
Contact		:xk.liu@siat.ac.cn
License		:GPL
'''


import math
from typing import Dict, Optional, Tuple, Union
import torch

from wenet.ssl.bestrq.mask import compute_mask_indices_v2
from wenet.ssl.wav2vec2.quantizer import Wav2vecGumbelVectorQuantizer
from wenet.ssl.wav2vec2.wav2vec2_model import (_compute_contrastive_loss,
                                               _sample_negative_indices)
from wenet.transformer.attention import RelPositionMultiHeadedAttention

from wenet.transformer.encoder import ConformerEncoder, TransformerEncoder
from wenet.transformer.encoder_layer import ConformerEncoderLayer
from wenet.utils.mask import make_non_pad_mask


class W2VBERTEncoder(torch.nn.Module):
    def __init__(
        self,
        encoder: Union[ConformerEncoder, TransformerEncoder],
        embedding_dim: int = 256,
        num_embeddings: int = 320,
        num_codebooks: int = 1,
        mask_prob: float = 0.065,
        mask_length: int = 10,
        min_masks: int = 2,
        num_negatives: int = 100,
        features_regularization_weight: float = 0.01,
        max_gumbel_temperature: float = 2.0,
        min_gumbel_temperature: float = 0.1,
        gumbel_temperature_decay: float = 0.999995,
        contrastive_logits_temperature: float = 0.1,
        diversity_weight: float = 0.0,
        bias: bool = True,
        contrastive_blocks: int = 6,
        masked_blocks: int = 6,
        contrastive_weight: float = 1.0,
        mlm_weight: float = 1.0,
        warmup_steps: int = 25000,
    ) -> None:
        super().__init__()
        assert mask_prob > 0.0
        assert (contrastive_blocks > 0 and masked_blocks > 0 and
                contrastive_blocks + masked_blocks == len(encoder.encoders))
        self.contrastive_blocks = contrastive_blocks
        self.masked_blocks = masked_blocks

        self.mask_prob = mask_prob
        self.mask_length = mask_length
        self.min_masks = min_masks
        self.num_negatives = num_negatives

        self.features_regularization_weight = features_regularization_weight
        self.diversity_weight = diversity_weight

        self.contrastive_weight = contrastive_weight
        self.mlm_weight = mlm_weight
        self.warmup_steps = warmup_steps
        # encoder
        self.encoder = encoder

        # quantizer
        self.num_codebooks = num_codebooks
        self.quantizer = Wav2vecGumbelVectorQuantizer(
            self.encoder.output_size(),
            num_codebooks=num_codebooks,
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            hard=False,
        )
        self.max_gumbel_temp = max_gumbel_temperature
        self.min_gumbel_temp = min_gumbel_temperature
        self.gumbel_temp_decay = gumbel_temperature_decay

        self.num_codevectors_per_group = num_embeddings
        self.num_codevector_groups = num_codebooks

        self.contrastive_logits_temp = contrastive_logits_temperature

        # NOET(Mddct): mask_em is replaced by random value in Wav-BERT
        # self.mask_emb = torch.nn.parameter.Parameter(
        #     torch.empty(self.encoder.output_size()).uniform_(),
        #     requires_grad=True,
        # )
        # TODO(Mddct): support causal or lookahead mask or keep consistent with
        # wenet dynamic chunk training

        # # n softmax
        self.encoder_top_n_out = torch.nn.parameter.Parameter(
            torch.empty(num_codebooks, self.encoder.output_size(),
                        num_embeddings))
        torch.nn.init.trunc_normal_(self.encoder_top_n_out, std=0.02)
        self.bias = bias
        if bias:
            self.encoder_top_n_out_bias = torch.nn.parameter.Parameter(
                torch.empty(num_codebooks, num_embeddings))
            torch.nn.init.zeros_(self.encoder_top_n_out_bias)

        # reset parameter
        self.reset_encoder_parameter()

    def reset_encoder_parameter(self):
        def _reset_parameter(module: torch.nn.Module):
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.trunc_normal_(module.weight.data,
                                            mean=0.0,
                                            std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, torch.nn.Conv1d):
                torch.nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    k = math.sqrt(module.groups /
                                  (module.in_channels * module.kernel_size[0]))
                    torch.nn.init.uniform_(module.bias, a=-k, b=k)
            elif isinstance(module, torch.Tensor):
                torch.nn.init.trunc_normal_(module)
            else:
                raise NotImplementedError("other module not support now")

        encoders = self.encoder.encoders
        for _, layer in enumerate(encoders):
            self_attn = layer.self_attn
            _reset_parameter(self_attn.linear_q)
            _reset_parameter(self_attn.linear_k)
            _reset_parameter(self_attn.linear_v)
            _reset_parameter(self_attn.linear_out)
            if isinstance(self_attn, RelPositionMultiHeadedAttention):
                _reset_parameter(self_attn.pos_bias_u)
                _reset_parameter(self_attn.pos_bias_v)
            if isinstance(layer, ConformerEncoderLayer):
                conv1, conv2 = (layer.conv_module.pointwise_conv1,
                                layer.conv_module.depthwise_conv)
                _reset_parameter(conv1)
                _reset_parameter(conv2)

    @torch.jit.unused
    def forward(
        self,
        batch: Dict,
        device: torch.device,
    ):
        steps = batch.get('steps', None)
        xs = batch['feats'].to(device)
        xs_lens = batch['feats_lengths'].to(device)
        assert xs.size(0) == xs_lens.size(0)
        assert steps is not None

        # 1 forward subsampling
        # NOTE(Mddct): use subsampling as feature extraction
        xs, pos_emb, masks = self._forward_subsampling(xs, xs_lens)
        # 2 mask features
        masked_xs, masked_masks = self._apply_mask(xs, masks.squeeze(1))
        # 3 forward encoder blocks
        xs, masks = self._forward_encoder_blocks(masked_xs, masks, pos_emb, masks)
        return xs, masks

    def _apply_mask(
            self, xs: torch.Tensor,
            xs_masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        masks = compute_mask_indices_v2(xs.size()[:-1],
                                        ~xs_masks,
                                        self.mask_prob,
                                        self.mask_length,
                                        min_masks=self.min_masks,
                                        device=xs.device)
        masks_expand = masks.unsqueeze(-1)  # [B, T, 1]

        mask_emb = torch.normal(mean=0,
                                std=0.1,
                                size=xs.size(),
                                device=xs.device)
        xs = torch.where(masks_expand, mask_emb, xs)

        return xs, masks


    def _forward_subsampling(
        self, xs: torch.Tensor, xs_lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        masks = make_non_pad_mask(xs_lens).unsqueeze(1)  # (B, 1, T)
        if self.encoder.global_cmvn is not None:
            xs = self.encoder.global_cmvn(xs)
        xs, pos_emb, masks = self.encoder.embed(xs, masks)
        return xs, pos_emb, masks

    def _forward_encoder_blocks(
        self, xs: torch.Tensor, xs_masks: torch.Tensor, pos_emb: torch.Tensor,
        mask_pad: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        masks = xs_masks
        xs: torch.Tensor
        # forward contrastive layers get context vector for Contrastive Loss
        for layer in self.encoder.encoders[:self.contrastive_blocks]:
            xs, masks, _, _ = layer(xs, xs_masks, pos_emb, mask_pad)
        return xs, masks
