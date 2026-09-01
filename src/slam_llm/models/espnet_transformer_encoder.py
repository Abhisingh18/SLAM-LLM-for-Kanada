"""ESPnet2 Transformer ASR encoder (Nithya/Akshaya's checkpoint) as a frozen
SLAM-ASR speech encoder.

Ported from Akshaya's SLAM-LLM fork
(akshaya/REPO_SLAM/SLAM-LLM/src/slam_llm/models/encoder.py, class
EspnetTransformerEncoder) for encoder #3 of Prof. Umesh's 5-encoder Kannada
comparison study.

Loader for the espnet2 ASR checkpoint's frontend + global_mvn + encoder
stack (per this run's config.yaml: frontend=default, normalize=global_mvn,
encoder=transformer -- NOT e_branchformer; the checkpoint's encoder.* keys
are self_attn/feed_forward/norm1/norm2, matching
espnet2.asr.encoder.transformer_encoder.TransformerEncoder, not
EBranchformerEncoder's cgmlp/merge_proj/norm_mha/norm_mlp/norm_final
layout). Only frontend/normalize/encoder are loaded -- decoder.*/ctc.* are
dropped since only the encoder feeds the LLM.

Pipeline (raw 16kHz waveform -> LLM-ready frames):
  raw audio -> DefaultFrontend (STFT+logmel, 80-dim) -> GlobalMVN
  (checkpoint's baked-in mean/std) -> TransformerEncoder (conv2d 4x
  subsample, 12 blocks, 512-dim) -> encoder_outs
"""

import logging
import torch
import torch.nn as nn


class EspnetTransformerEncoder(nn.Module):
    def __init__(self, frontend, mean, std, encoder):
        super().__init__()
        self.frontend = frontend
        self.register_buffer("mvn_mean", mean)
        self.register_buffer("mvn_std", std)
        self.encoder = encoder

    @classmethod
    def load(cls, model_config):
        from espnet2.asr.frontend.default import DefaultFrontend
        from espnet2.asr.encoder.transformer_encoder import TransformerEncoder
        log = logging.getLogger("EspnetTransformerEncoder")

        ck = torch.load(model_config.encoder_path, map_location="cpu")
        assert isinstance(ck, dict) and "encoder.after_norm.weight" in ck, (
            "expected a flat espnet2 state_dict (frontend./normalize./"
            "encoder./decoder./ctc. prefixed keys), got something else -- "
            "this loader does not handle fairseq-style {'model':..., 'cfg':...} "
            "checkpoints (see Data2VecAQCEncoder for that format).")

        # frontend: fixed (non-trainable) log-mel; fs/n_mels must match how
        # the checkpoint was trained (config.yaml frontend_conf: fs=16k,
        # n_mels defaults to 80, matching encoder_conf.input_size below).
        frontend = DefaultFrontend(fs=16000, n_mels=80)

        encoder = TransformerEncoder(
            input_size=80,
            output_size=getattr(model_config, "encoder_dim", 512),
            attention_heads=8,
            linear_units=2048,
            num_blocks=12,
            dropout_rate=0.1,
            positional_dropout_rate=0.1,
            attention_dropout_rate=0.1,
            input_layer="conv2d",
            normalize_before=True,
        )
        enc_sd = {k[len("encoder."):]: v for k, v in ck.items()
                   if k.startswith("encoder.")}
        missing, unexpected = encoder.load_state_dict(enc_sd, strict=False)
        log.info("[espnet_transformer] ckpt: %s", model_config.encoder_path)
        log.info("[espnet_transformer] missing keys: %s", missing)
        log.info("[espnet_transformer] unexpected keys: %s", unexpected)

        mean = ck["normalize.mean"].float()
        std = ck["normalize.std"].float()

        return cls(frontend, mean, std, encoder)

    def extract_features(self, source, padding_mask=None):
        """source: (B, num_samples) raw 16kHz waveform.
        padding_mask: 1=pad (fairseq convention, same as data2vec_aqc's
        branch in slam_model.py -- pass `1 - audio_mask`).
        """
        if padding_mask is not None:
            ilens = (~padding_mask.bool()).sum(dim=1)
        else:
            ilens = torch.full((source.size(0),), source.size(1),
                                dtype=torch.long, device=source.device)
        feats, feats_lens = self.frontend(source, ilens)
        feats = (feats - self.mvn_mean) / self.mvn_std
        enc_out, _enc_olens, _ = self.encoder(feats, feats_lens)
        return enc_out
