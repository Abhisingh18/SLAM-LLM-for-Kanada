"""Wav2Vec2-BERT 2.0 (Meta, https://huggingface.co/facebook/w2v-bert-2.0)
frozen speech encoder wrapper for SLAM-ASR.

Unlike data2vec-AQC/HuBERT (which consume raw waveform directly through an
internal CNN frontend), Wav2Vec2-BERT expects precomputed 80-dim log-mel
filterbank *input_features* (via its `SeamlessM4TFeatureExtractor`), not raw
`input_values`. This wrapper does that feature extraction internally so it
can still be called with the same (raw audio, 1=real/0=pad mask) convention
the rest of SLAM-LLM's encoders use.
"""

import torch
import torch.nn as nn


class Wav2Vec2BertEncoder(nn.Module):
    def __init__(self, model, feature_extractor):
        super().__init__()
        self.model = model
        self.feature_extractor = feature_extractor
        self._output_size = model.config.output_hidden_size
        # Parameter-free LayerNorm to match data2vec-aqc's output scale (see
        # the same rationale in xeus_encoder.py) -- Wav2Vec2-BERT's raw scale
        # is less extreme than XEUS's but still not matched to data2vec-aqc's.
        self.output_norm = torch.nn.LayerNorm(self._output_size, elementwise_affine=False)

    def forward(self, input_values, attention_mask=None):
        # input_values: (B, T) raw waveform, 16kHz. attention_mask: 1=real/0=pad.
        device = input_values.device
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1).long().tolist()
        else:
            lengths = [input_values.shape[1]] * input_values.shape[0]

        raw_list = [
            input_values[i, :lengths[i]].detach().float().cpu().numpy()
            for i in range(input_values.shape[0])
        ]

        feats = self.feature_extractor(
            raw_list, sampling_rate=16000, return_tensors="pt", padding=True,
        )
        input_features = feats["input_features"].to(device=device, dtype=self.model.dtype)
        feat_attention_mask = feats.get("attention_mask")
        if feat_attention_mask is not None:
            feat_attention_mask = feat_attention_mask.to(device)

        out = self.model(input_features=input_features, attention_mask=feat_attention_mask)
        out.last_hidden_state = self.output_norm(out.last_hidden_state)
        return out

    def output_size(self):
        return self._output_size

    @classmethod
    def load(cls, model_config):
        from transformers import Wav2Vec2BertModel, SeamlessM4TFeatureExtractor

        ckpt = model_config.encoder_path
        model = Wav2Vec2BertModel.from_pretrained(ckpt)
        model.config.apply_spec_augment = False  # deterministic when frozen
        feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(ckpt)
        return cls(model, feature_extractor)
