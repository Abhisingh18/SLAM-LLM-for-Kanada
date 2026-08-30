"""XEUS (CMU/ESPnet) frozen speech encoder wrapper for SLAM-ASR.

XEUS (https://huggingface.co/espnet/xeus) is an ESPnet2 SSL model (577M
params, E-Branchformer encoder, 1M hours pretraining).

HISTORY: our first version of this file manually patched the checkpoint's
config.yaml (injecting flat `masker`/`loss` keys + a `normalize_output`
CNNFrontend kwarg shim) because `SSLTask.build_model_from_file()` -- the
official espnet2 loader, no patching needed -- failed with `AttributeError:
'Namespace' object has no attribute 'masker'`. That turned out to be an
espnet2 PACKAGE version problem, not a checkpoint problem: our installed
`espnet==202402` predates the checkpoint's (`xeus_checkpoint_new.pth`)
version-202412+ mainline `util`/`loss` config schema. Upgrading to
`espnet==202506` (matching Tomson's Hindi XEUS setup's env) fixed
`build_model_from_file()` outright -- confirmed working, loading 100% of
checkpoint params (vs. the old patch route's ~98.3%, which left one
positional-embedding conv submodule randomly-initialized due to an old-vs-new
internal naming mismatch). This file now uses the same official loader
Tomson's setup uses, no manual patching.
"""

import torch
import torch.nn as nn


class XeusEncoder(nn.Module):
    """Wraps the full ESPnetSSLModel (frontend + encoder) and exposes
    `.extract_features(source, wav_lengths)` -> [B, T, 1024], matching
    Tomson's Hindi XEUS setup's calling convention (slam_model.py calls this
    directly, not via a `.last_hidden_state`-wrapped object).
    """

    def __init__(self, model, output_size, *, tgt_layer=None):
        super().__init__()
        self.model = model
        self._output_size = output_size
        self.tgt_layer = tgt_layer
        # Parameter-free LayerNorm to match data2vec-aqc's output scale.
        # NOTE: Tomson's Hindi setup does NOT have this -- his checkpoint's
        # raw output is apparently already well-scaled (different/newer
        # checkpoint schema, see module docstring). OUR checkpoint's raw
        # output measured std~2.05 / absmax~34 vs data2vec-aqc's std~0.29 --
        # keeping this fix since it's checkpoint-specific, not a structural
        # bug in the forward pass itself (zero learnable params, so it's
        # unaffected by freeze_encoder=true handling; safe to keep even if
        # eventually proven unnecessary once/if we switch to Tomson's exact
        # checkpoint file).
        self.output_norm = torch.nn.LayerNorm(output_size, elementwise_affine=False)

    def extract_features(self, source, wav_lengths):
        # Replicates espnet's ESPnetSSLModel.encode() (frontend -> normalize ->
        # preencoder -> encoder), matching Tomson's Hindi XEUS setup line-for-
        # line, instead of using the `.encode(...)` convenience wrapper.
        from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
        m = self.model
        speech, speech_lengths = m._extract_feats(source, wav_lengths)
        if m.normalize is not None:
            speech, speech_lengths = m.normalize(speech, speech_lengths)
        if m.preencoder is not None:
            speech, speech_lengths = m.preencoder(speech, speech_lengths)
        # The frontend length formula can disagree with the actual feature
        # tensor in EITHER direction, so clamp lengths to the tensor and pin
        # the mask width to it: keeps make_pad_mask's `maxlen >= lengths.max()`
        # assert happy and makes the mask match the e-branchformer attention
        # exactly.
        T = speech.size(1)
        speech_lengths = speech_lengths.clamp(max=T)
        pad_masks = make_pad_mask(speech_lengths, maxlen=T).to(speech.device)
        speech, _, _ = m.encoder(
            speech, speech_lengths, masks=pad_masks, return_all_hs=True)
        final_output, per_layer = speech[0], speech[1]
        feats = per_layer[self.tgt_layer] if self.tgt_layer is not None else final_output

        feats = self.output_norm(feats)
        return feats

    def output_size(self):
        return self._output_size

    @classmethod
    def load(cls, model_config):
        from espnet2.tasks.ssl import SSLTask
        ckpt = model_config.encoder_path
        # None -> espnet reads the config.yaml sitting next to encoder_path
        # (matches Tomson's Hindi setup's `model_config.encoder_config`).
        cfg_path = getattr(model_config, "encoder_config", None) \
            or getattr(model_config, "encoder_config_path", None)
        espnet_model, _ = SSLTask.build_model_from_file(cfg_path, ckpt, "cpu")

        output_size = espnet_model.encoder.output_size()
        tgt_layer = getattr(model_config, "encoder_tgt_layer", None)
        return cls(espnet_model, output_size, tgt_layer=tgt_layer)
