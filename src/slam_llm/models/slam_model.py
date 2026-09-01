import os
import types
import torch
import soundfile as sf
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import List, Optional, Tuple, Union
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoModel, AutoModelForSeq2SeqLM, T5ForConditionalGeneration
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

from slam_llm.utils.config_utils import generate_peft_config
from slam_llm.utils.train_utils import print_module_size, print_model_size
from peft import PeftModel, PeftConfig
from torch.nn import CrossEntropyLoss
from slam_llm.utils.metric import compute_accuracy

import logging
logger = logging.getLogger(__name__)

def model_factory(train_config, model_config, **kwargs):
    # return necessary components for training
    tokenizer = setup_tokenizer(train_config, model_config, **kwargs)

    encoder = setup_encoder(train_config, model_config, **kwargs)

    # llm
    llm = setup_llm(train_config, model_config, **kwargs)

    # projector
    encoder_projector = setup_encoder_projector(
        train_config, model_config, **kwargs
    )
    model = slam_model(
        encoder,
        llm,
        encoder_projector,
        tokenizer,
        train_config,
        model_config,
        **kwargs,
    )

    ckpt_path = kwargs.get("ckpt_path", None) #FIX(MZY): load model ckpt(mainly projector, related to model_checkpointing/checkpoint_handler.py: save_model_checkpoint_peft)
    if ckpt_path is not None:
            logger.info("loading other parts from: {}".format(ckpt_path))
            ckpt_dict = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(ckpt_dict, strict=False)

    print_model_size(model, train_config, int(os.environ["RANK"]) if train_config.enable_fsdp or train_config.enable_ddp else 0)
    return model, tokenizer


def setup_tokenizer(train_config, model_config, **kwargs):
    # Load the tokenizer and add special tokens
    if "vallex" in model_config.llm_name.lower():
        return None  
    elif "mupt" in model_config.llm_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_config.llm_path,
                                            trust_remote_code=True,
                                            use_fast=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_config.llm_path)
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def setup_encoder(train_config, model_config, **kwargs):
    encoder_list = model_config.encoder_name.split(",") if model_config.encoder_name else []
    if len(encoder_list) == 0:
        return None
    if len(encoder_list) == 1:
        encoder_name = encoder_list[0]
        if encoder_name == "whisper" or encoder_name == "qwen-audio":
            from slam_llm.models.encoder import WhisperWrappedEncoder
            encoder = WhisperWrappedEncoder.load(model_config)
        if encoder_name == "beats": 
            from slam_llm.models.encoder import BEATsEncoder
            encoder = BEATsEncoder.load(model_config)
        if encoder_name == "eat":
            from slam_llm.models.encoder import EATEncoder
            encoder = EATEncoder.load(model_config)
        if encoder_name == "clap": 
            from slam_llm.models.encoder import CLAPEncoder
            encoder = CLAPEncoder.load(model_config)
        if encoder_name == "SpatialAST":
            from slam_llm.models.encoder import SpatialASTEncoder
            encoder = SpatialASTEncoder.load(model_config)
        if encoder_name == "wavlm":
            from slam_llm.models.encoder import WavLMEncoder
            encoder = WavLMEncoder.load(model_config)
        if encoder_name == "av_hubert":
            from slam_llm.models.encoder import AVHubertEncoder
            encoder = AVHubertEncoder.load(model_config)
        if encoder_name == "hubert":
            from slam_llm.models.encoder import HubertEncoder
            encoder = HubertEncoder.load(model_config)
        if encoder_name == "hubert_hf":
            from slam_llm.models.encoder import HubertHFEncoder
            encoder = HubertHFEncoder.load(model_config)
        if encoder_name == "musicfm":
            from slam_llm.models.encoder import MusicFMEncoder
            encoder = MusicFMEncoder.load(model_config)
        if encoder_name == "emotion2vec":
            from slam_llm.models.encoder import Emotion2vecEncoder
            encoder = Emotion2vecEncoder.load(model_config)
        if encoder_name == "data2vec_aqc":
            from slam_llm.models.encoder import Data2VecAQCEncoder
            encoder = Data2VecAQCEncoder.load(model_config)
        if encoder_name == "espnet_transformer":
            from slam_llm.models.espnet_transformer_encoder import EspnetTransformerEncoder
            encoder = EspnetTransformerEncoder.load(model_config)
        if encoder_name == "xeus":
            from slam_llm.models.xeus_encoder import XeusEncoder
            encoder = XeusEncoder.load(model_config)
        if encoder_name == "wav2vec2_bert":
            from slam_llm.models.wav2vec2_bert_encoder import Wav2Vec2BertEncoder
            encoder = Wav2Vec2BertEncoder.load(model_config)
        if encoder_name == "hubert_embedding_generator":
            # frozen HuBERT-xlarge + sinusoidal PE + N self-attention blocks
            # (lm_head removed); taps the post-SA 1280-d embeddings.
            from slam_llm.models.hubert_embedding_generator import HubertEmbeddingGenerator
            encoder = HubertEmbeddingGenerator.load(model_config)

        if "llama" in encoder_name.lower():
            from slam_llm.models.encoder import HfTextEncoder
            encoder = HfTextEncoder.load(model_config)
    print_module_size(encoder, encoder_name, int(os.environ["RANK"]) if train_config.enable_fsdp or train_config.enable_ddp else 0)

    if train_config.freeze_encoder:
        for name, param in encoder.named_parameters(): 
            param.requires_grad = False
        encoder.eval()
    print_module_size(encoder, encoder_name, int(os.environ["RANK"]) if train_config.enable_fsdp or train_config.enable_ddp else 0)

    return encoder

def _load_gemma4_text_llm(llm_path, use_cache):
    """Load only the text decoder of a multimodal Gemma-4 checkpoint (e.g.
    google/gemma-4-12B-it) as a Gemma4ForCausalLM.

    Gemma4UnifiedForConditionalGeneration stores text weights under the
    `language_model.` prefix. We extract only those tensors into a fresh
    Gemma4ForCausalLM so LoRA attaches only to the text decoder.
    """
    import glob
    from safetensors.torch import safe_open
    from transformers import Gemma4ForCausalLM
    text_cfg = AutoConfig.from_pretrained(llm_path).text_config
    if use_cache is not None:
        text_cfg.use_cache = use_cache
    prefix = "language_model."
    state = {}
    for shard in sorted(glob.glob(os.path.join(llm_path, "*.safetensors"))):
        with safe_open(shard, framework="pt") as f:
            for k in f.keys():
                if k.startswith(prefix):
                    state[k[len(prefix):]] = f.get_tensor(k)
    model = Gemma4ForCausalLM(text_cfg)
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.tie_weights()
    missing = [m for m in missing if m != "lm_head.weight"]
    if missing or unexpected:
        logger.warning(f"gemma-4 text LM load: missing={missing} unexpected={unexpected}")
    return model


def _load_gemma3_text_llm(llm_path, use_cache):
    """Load only the text decoder of a multimodal Gemma-3 checkpoint (e.g.
    google/gemma-3-4b-it) as a Gemma3ForCausalLM.

    The `-it`/multimodal checkpoint is `Gemma3ForConditionalGeneration`: a SigLIP
    vision tower + multi_modal_projector + text decoder, with the text weights
    stored under the `language_model.` prefix and a tied lm_head (not stored).
    For speech->text we only want the text decoder, so we read just the
    `language_model.*` tensors into a fresh Gemma3ForCausalLM. This skips loading
    the vision stack (saves memory per rank) and, crucially, stops LoRA from
    attaching to the vision tower's q/k/v_proj. The resulting module exposes
    `model.embed_tokens` and the usual causal-LM forward/generate, so the rest of
    SLAM-LLM treats it exactly like the text-only gemma-3-4b-pt model.
    """
    import glob
    from safetensors.torch import safe_open
    from transformers import Gemma3ForCausalLM
    text_cfg = AutoConfig.from_pretrained(llm_path).text_config
    if use_cache is not None:  # strict gemma3 config rejects None; leave default otherwise
        text_cfg.use_cache = use_cache
    prefix = "language_model."
    state = {}
    for shard in sorted(glob.glob(os.path.join(llm_path, "*.safetensors"))):
        with safe_open(shard, framework="pt") as f:
            for k in f.keys():
                if k.startswith(prefix):
                    state[k[len(prefix):]] = f.get_tensor(k)
    model = Gemma3ForCausalLM(text_cfg)
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.tie_weights()  # lm_head.weight is tied to embed_tokens (not stored in the checkpoint)
    missing = [m for m in missing if m != "lm_head.weight"]
    if missing or unexpected:
        logger.warning(f"gemma-3 text LM load: missing={missing} unexpected={unexpected}")
    return model


def setup_llm(train_config, model_config, **kwargs):
    from pkg_resources import packaging
    use_cache = False if train_config.enable_fsdp or train_config.enable_ddp else None
    if (train_config.enable_fsdp or train_config.enable_ddp) and train_config.low_cpu_fsdp:
        """
        for FSDP, we can save cpu memory by loading pretrained model on rank0 only.
        this avoids cpu oom when loading large models like llama 70B, in which case
        model alone would consume 2+TB cpu mem (70 * 4 * 8). This will add some comms
        overhead and currently requires latest nightly.
        """
        # v = packaging.version.parse(torch.__version__)
        # verify_latest_nightly = v.is_devrelease and v.dev >= 20230701
        # if not verify_latest_nightly:
        #     raise Exception("latest pytorch nightly build is required to run with low_cpu_fsdp config, "
        #                     "please install latest nightly.")
        rank = int(os.environ["RANK"])
        if rank == 0:
            if "vallex" in model_config.llm_name.lower():
                from src.slam_llm.models.vallex.vallex_config import VallexConfig
                from src.slam_llm.models.vallex.vallex_model import VALLE
                vallex_config = VallexConfig(
                    **model_config
                )
                model = VALLE(vallex_config)
            elif "aya" in model_config.llm_name.lower():
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_config.llm_path,
                    load_in_8bit=True if train_config.quantization else None,
                    device_map="auto" if train_config.quantization else None,
                    use_cache=use_cache,
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_config.llm_path,
                    load_in_8bit=True if train_config.quantization else None,
                    device_map="auto" if train_config.quantization else None,
                    use_cache=use_cache,
                )
        else:
            llama_config = AutoConfig.from_pretrained(model_config.llm_path)
            llama_config.use_cache = use_cache
            # with torch.device("meta"):
            if "aya" in model_config.llm_name.lower():
                model = AutoModelForSeq2SeqLM(llama_config)
            else:
                model = AutoModelForCausalLM(llama_config) #(FIX:MZY): torch 2.0.1 does not support `meta`

    else:
        if "vallex" in model_config.llm_name.lower():
            from src.slam_llm.models.vallex.vallex_config import VallexConfig
            from src.slam_llm.models.vallex.vallex_model import VALLE
            vallex_config = VallexConfig(
                **model_config
            )
            model = VALLE(vallex_config)
        elif "aya" in model_config.llm_name.lower():
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_config.llm_path,
                load_in_8bit=True if train_config.quantization else None,
                device_map="auto" if train_config.quantization else None,
                use_cache=use_cache,
            )
        elif getattr(AutoConfig.from_pretrained(model_config.llm_path), "model_type", "") == "gemma4_unified":
            # Multimodal Gemma-4 checkpoint (e.g. gemma-4-12B-it): keep only the text decoder.
            model = _load_gemma4_text_llm(model_config.llm_path, use_cache)
        elif getattr(AutoConfig.from_pretrained(model_config.llm_path), "model_type", "") == "gemma3":
            # Multimodal Gemma-3 checkpoint (e.g. gemma-3-4b-it): keep only the text decoder.
            model = _load_gemma3_text_llm(model_config.llm_path, use_cache)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_config.llm_path,
                load_in_8bit=True if train_config.quantization else None,
                device_map="auto" if train_config.quantization else None,
                use_cache=use_cache,
            )
    if (train_config.enable_fsdp or train_config.enable_ddp) and train_config.use_fast_kernels:
        """
        For FSDP and FSDP+PEFT, setting 'use_fast_kernels' will enable
        using of Flash Attention or Xformer memory-efficient kernels
        based on the hardware being used. This would speed up fine-tuning.
        """
        try:
            from optimum.bettertransformer import BetterTransformer
            model = BetterTransformer.transform(model)
        except ImportError:
            logger.warning("Module 'optimum' not found. Please install 'optimum' it before proceeding.")

    print_module_size(model, model_config.llm_name, int(os.environ["RANK"]) if train_config.enable_fsdp or train_config.enable_ddp else 0)

    # Prepare the model for int8 training if quantization is enabled
    if train_config.quantization:
        model = prepare_model_for_kbit_training(model)

    if train_config.freeze_llm: # TODO:to test offical `freeze_layers` and `num_freeze_layers`
        for name, param in model.named_parameters(): 
            param.requires_grad = False
        model.eval()
        
    if kwargs.get("peft_ckpt", None): # (FIX:MZY):reload will get wrong results when decoding
        logger.info("loading peft_ckpt from: {}".format(kwargs.get("peft_ckpt")))
        model = PeftModel.from_pretrained(model=model, model_id=kwargs.get("peft_ckpt"), is_trainable=True)
        model.print_trainable_parameters()
    elif train_config.use_peft:
        logger.info("setup peft...")
        peft_config = generate_peft_config(train_config)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    print_module_size(model, model_config.llm_name, int(os.environ["RANK"]) if train_config.enable_fsdp or train_config.enable_ddp else 0)
    return model

def setup_encoder_projector(train_config, model_config, **kwargs):
    if model_config.encoder_projector == "linear":
        from slam_llm.models.projector import EncoderProjectorConcat
        encoder_projector = EncoderProjectorConcat(model_config)
    elif model_config.encoder_projector == "cov1d-linear":
        from slam_llm.models.projector import EncoderProjectorCov1d
        encoder_projector = EncoderProjectorCov1d(model_config)
    elif model_config.encoder_projector == "q-former":
        from slam_llm.models.projector import EncoderProjectorQFormer
        encoder_projector = EncoderProjectorQFormer(model_config)
    else:
        return None
    print_module_size(encoder_projector, model_config.encoder_projector, int(os.environ["RANK"]) if train_config.enable_fsdp or train_config.enable_ddp else 0)
    return encoder_projector


class slam_model(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        llm: nn.Module,
        encoder_projector: nn.Module,
        tokenizer, 
        train_config, 
        model_config, 
        **kwargs
    ):
        super().__init__()
        # modality encoder 
        self.encoder = encoder

        # llm
        self.llm = llm

        # projector
        self.encoder_projector = encoder_projector

        # tokenizer
        self.tokenizer = tokenizer
        self.metric = kwargs.get("metric", "acc")

        self.train_config = train_config
        self.model_config = model_config

        if train_config.get("enable_deepspeed", False):
            def new_forward(self, input):
                output = F.layer_norm(
                    input.float(),
                    self.normalized_shape,
                    self.weight.float() if self.weight is not None else None,
                    self.bias.float() if self.bias is not None else None,
                    self.eps,
                )
                return output.type_as(input)
            for item in self.modules():
                # Only patch STOCK nn.LayerNorm.forward (matches Tomson's Hindi
                # XEUS setup exactly). Some encoders subclass nn.LayerNorm and
                # override forward to transpose (e.g. XEUS's frontend
                # TransposedLayerNorm, [B,C,T]<->[B,T,C]); clobbering that drops
                # the transpose and F.layer_norm hits the wrong axis. Stock
                # LayerNorms (LLM, whisper/hubert, XEUS final_norm) patch as
                # before, INCLUDING any subclass that doesn't override forward
                # (broader than the old `type(item) is nn.LayerNorm` exact-type
                # check, which excluded those too).
                if isinstance(item, nn.LayerNorm) and type(item).forward is nn.LayerNorm.forward:
                    item.forward = types.MethodType(new_forward, item)



    def forward(self,
                input_ids: torch.LongTensor = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                **kwargs,
                ):
        audio_mel = kwargs.get("audio_mel", None)
        audio_mel_mask = kwargs.get("audio_mel_mask", None)
        audio_mel_post_mask = kwargs.get("audio_mel_post_mask", None) # 2x downsample for whisper

        audio = kwargs.get("audio", None)
        audio_mask = kwargs.get("audio_mask", None)
        visual = kwargs.get("visual", None)
        visual_mask = kwargs.get("visual_mask", None)
        text = kwargs.get("text", None)

        # for text encoder
        instruct_ids = kwargs.get("instruct_ids", None)
        instruct_mask = kwargs.get("instruct_mask", None)

        modality_mask = kwargs.get("modality_mask", None)
        
        zh_data = kwargs.get("zh", None)
        en_data = kwargs.get("en", None)

        encoder_outs = None
        if audio_mel is not None or audio is not None or visual is not None or text is not None:
            if self.train_config.freeze_encoder: # freeze encoder
                self.encoder.eval()

            if self.model_config.encoder_name == "whisper":
                encoder_outs = self.encoder.extract_variable_length_features(audio_mel.permute(0, 2, 1)) # bs*seq*dim
            if self.model_config.encoder_name == "beats":
                encoder_outs, audio_mel_post_mask = self.encoder.extract_features(audio_mel, audio_mel_mask) # bs*seq*dim
            if self.model_config.encoder_name == "eat":
                encoder_outs = self.encoder.model.extract_features(audio_mel.unsqueeze(dim=1), padding_mask = None, mask=False, remove_extra_tokens = False)['x']
            if self.model_config.encoder_name == "clap": 
                if text is not None: 
                    encoder_outs = self.encoder.encode_text(text).unsqueeze(1)  # [btz, 1, dim]        
                elif audio is not None: 
                    encoder_outs = self.encoder.encode_audio(audio)  # with projection-based decoding 
            if self.model_config.encoder_name == "SpatialAST":
                encoder_outs = self.encoder(audio) # output: [bs, seq_len=3+512, dim=768]
            if self.model_config.encoder_name == "wavlm":
                encoder_outs = self.encoder.extract_features(audio, 1 - audio_mask) #(FIX:MZY): 1-audio_mask is needed for wavlm as the padding mask
            if self.model_config.encoder_name == "hubert":
                results = self.encoder(source = audio, padding_mask = 1-audio_mask)
                if self.model_config.encoder_type == "pretrain":
                    encoder_outs, audio_mel_post_mask = results["x"], results["padding_mask"]
                if self.model_config.encoder_type == "finetune":
                    encoder_outs, audio_mel_post_mask = results["encoder_out"], results["padding_mask"]
                    encoder_outs = encoder_outs.transpose(0, 1)
            if self.model_config.encoder_name == "xeus":
                # Frozen ESPnet2 XEUS SSL encoder — matches Tomson's Hindi XEUS
                # setup's calling convention: raw 16kHz wav + real per-utterance
                # SAMPLE counts (audio_mask is 1=real/0=pad at sample
                # resolution, so the row sum is exactly each utterance's
                # length -- no 1-mask inversion here). extract_features()
                # returns the [B,T,1024] tensor directly (not wrapped in a
                # .last_hidden_state object). See xeus_encoder.py for details.
                if audio_mask is not None:
                    wav_lengths = audio_mask.sum(dim=1).long()
                else:
                    wav_lengths = torch.full(
                        (audio.size(0),), audio.size(1),
                        dtype=torch.long, device=audio.device)
                encoder_outs = self.encoder.extract_features(audio, wav_lengths)
            if self.model_config.encoder_name == "wav2vec2_bert":
                # Frozen Wav2Vec2-BERT 2.0 — raw audio + 1=real/0=pad mask in,
                # log-mel feature extraction happens inside the wrapper.
                # See wav2vec2_bert_encoder.py.
                encoder_outs = self.encoder(input_values=audio, attention_mask=audio_mask).last_hidden_state
            if self.model_config.encoder_name == "hubert_hf":
                # HF `transformers` HuBERT (FillerHubertModel body, lm_head dropped):
                # emits 768-d last_hidden_state -- the standard SLAM-ASR tap. audio_mask
                # is 1=real / 0=pad, which is exactly HF's attention_mask convention
                # (note: the fairseq `hubert` branch above uses 1-audio_mask instead).
                encoder_outs = self.encoder(input_values=audio, attention_mask=audio_mask).last_hidden_state
                if getattr(self.model_config, "strip_frames_before_projector", False) and modality_mask is not None:
                    # Frame-budget mode: the dataset reserved only round(fps*sec)//k
                    # audio token slots (dataset_config.audio_frames_per_sec), so at
                    # most max(slots)*k LEFTMOST encoder frames can ever be scattered
                    # into the LLM input. Slice them here so the projector skips the
                    # (mostly <fill>/silence) tail. Per-sample precision still comes
                    # from the scatter below, which copies only slots_i tokens.
                    k = getattr(self.encoder_projector, "k", 1)
                    max_budget_frames = int(modality_mask.sum(dim=1).max().item()) * k
                    encoder_outs = encoder_outs[:, :max_budget_frames, :]
            if self.model_config.encoder_name == "hubert_embedding_generator":
                # HuBERT-xlarge + PE + SA blocks (lm_head removed): emits the
                # post-SA 1280-d tap as .last_hidden_state. Same audio_mask
                # (1=real/0=pad) + duration-ceiling strip as the hubert_hf branch.
                encoder_outs = self.encoder(input_values=audio, attention_mask=audio_mask).last_hidden_state
                if getattr(self.model_config, "strip_frames_before_projector", False) and modality_mask is not None:
                    k = getattr(self.encoder_projector, "k", 1)
                    max_budget_frames = int(modality_mask.sum(dim=1).max().item()) * k
                    encoder_outs = encoder_outs[:, :max_budget_frames, :]
            if self.model_config.encoder_name == "av_hubert":
                results = self.encoder(source={'video':visual, 'audio':audio}, padding_mask=visual_mask) # bs*seq*dim  
                encoder_outs, audio_mel_post_mask = results["encoder_out"], results["padding_mask"]
                encoder_outs = encoder_outs.transpose(0, 1)
                audio_mel_post_mask = (~audio_mel_post_mask).float()
            if self.model_config.encoder_name == 'musicfm':
                encoder_outs = self.encoder.extract_features(audio, padding_mask = None) # MusicFM doesn't support padding mask 
            if self.model_config.encoder_name == "emotion2vec":
                encoder_outs = self.encoder.extract_features(audio, None)['x'] # bs*seq*dim
            if self.model_config.encoder_name == "data2vec_aqc":
                # fairseq convention: padding_mask is 1=pad (inverse of audio_mask).
                padding_mask = (1 - audio_mask) if audio_mask is not None else None
                if not torch.isfinite(audio).all():
                    n_bad = (~torch.isfinite(audio)).sum().item()
                    logger.warning(f"[data2vec_aqc] non-finite audio input: {n_bad} elt(s); replacing with 0")
                    audio = torch.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
                encoder_outs = self.encoder.extract_features(audio, padding_mask)
            if self.model_config.encoder_name == "espnet_transformer":
                # Same fairseq convention as data2vec_aqc: padding_mask is
                # 1=pad (inverse of audio_mask).
                padding_mask = (1 - audio_mask) if audio_mask is not None else None
                encoder_outs = self.encoder.extract_features(audio, padding_mask)
            if self.encoder is None:
                encoder_outs = audio_mel if audio_mel is not None else audio

            if self.model_config.encoder_projector == "q-former":
                encoder_outs = self.encoder_projector(encoder_outs, audio_mel_post_mask)
            if self.model_config.encoder_projector == "linear":
                encoder_outs = self.encoder_projector(encoder_outs)
            if self.model_config.encoder_projector == "cov1d-linear": 
                encoder_outs = self.encoder_projector(encoder_outs) 

        if instruct_ids is not None:
            if self.encoder is not None:
                encoder_outs = self.encoder(input_ids=instruct_ids, attention_mask=instruct_mask).last_hidden_state

            if self.model_config.encoder_projector == "q-former":
                encoder_outs = self.encoder_projector(encoder_outs, instruct_mask)
            if self.model_config.encoder_projector == "linear":
                encoder_outs = self.encoder_projector(encoder_outs)

        if input_ids is not None:
            input_ids[input_ids == -1] = 0
            if isinstance(self.llm, T5ForConditionalGeneration):
                inputs_embeds = self.llm.shared(input_ids)
            else:
                if hasattr(self.llm.model, "embed_tokens"):
                    inputs_embeds = self.llm.model.embed_tokens(input_ids)
                elif hasattr(self.llm.model.model, "embed_tokens"):
                    inputs_embeds = self.llm.model.model.embed_tokens(input_ids)
                else:
                    inputs_embeds = self.llm.model.model.model.embed_tokens(input_ids)

        if modality_mask is not None:
            modality_mask_start_indices = (modality_mask == True).float().argmax(dim=1)
            modality_lengths = torch.clamp(modality_mask.sum(dim=1), max=encoder_outs.shape[1]).tolist()

            encoder_outs_pad = torch.zeros_like(inputs_embeds)
            for i in range(encoder_outs.shape[0]):
                encoder_outs_pad[
                    i, modality_mask_start_indices[i]:modality_mask_start_indices[i]+modality_lengths[i]
                ] = encoder_outs[i][:modality_lengths[i]]
            
            inputs_embeds = encoder_outs_pad + inputs_embeds * (~modality_mask[:, :, None])

        if kwargs.get("inference_mode", False):
            return inputs_embeds, attention_mask

        if zh_data is not None and en_data is not None:
            model_outputs, acc = self.llm(zh=zh_data, en=en_data)
        else:
            model_outputs = self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
            acc = -1
            if self.metric:
                with torch.no_grad():
                    preds = torch.argmax(model_outputs.logits, -1)
                    acc = compute_accuracy(preds.detach()[:, :-1], labels.detach()[:, 1:], ignore_label=-100)

        return model_outputs, acc
    
    @torch.no_grad()
    def generate(self,
                input_ids: torch.LongTensor = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                **kwargs,
                ):
        kwargs["inference_mode"] = True

        inputs_embeds, attention_mask = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

        model_outputs = self.llm.generate(
            inputs_embeds=inputs_embeds,
            # max_length=kwargs.get("max_length", 200),
            max_new_tokens=kwargs.get("max_new_tokens", 200),
            num_beams=kwargs.get("num_beams", 4),
            do_sample=kwargs.get("do_sample", False),
            min_length=kwargs.get("min_length", 1),
            top_p=kwargs.get("top_p", 1.0),
            repetition_penalty=kwargs.get("repetition_penalty", 1.0),
            length_penalty=kwargs.get("length_penalty", 1.0),
            temperature=kwargs.get("temperature", 1.0),
            attention_mask=attention_mask,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id
        )

        return model_outputs
