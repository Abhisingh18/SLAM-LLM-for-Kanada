from dataclasses import dataclass, field
from typing import Optional, List

from torch.distributed.fsdp import ShardingStrategy


@dataclass
class ModelConfig:
    file: str = "examples/asr_librispeech/model/slam_model_asr.py:model_factory"
    llm_name: str = "vicuna-13b-v1.5"
    llm_path: str = "PATH/to/LLAMA/7B"
    llm_type: str = "decoder_only"
    llm_dim: int = 4096
    encoder_name: Optional[str] = None
    encoder_ds_rate: int = 2
    encoder_path: Optional[str] = None
    encoder_config: Optional[str] = field(default=None, metadata={
        "help": "path to an ESPnet SSL config.yaml (used by encoder_name=xeus; "
                "None -> ESPnet reads config.yaml sitting next to encoder_path)."
    })
    encoder_dim: int = 1280
    encoder_projector: str = "linear"
    encoder_projector_ds_rate: int = 5
    modal: str = "audio"
    normalize: Optional[bool] = field(default=False, metadata={
        "help": "whether input is normalized, used for models such as wavlm"
    })
    encoder_type: str = field(default="finetune", metadata={
        "help": "whether model is only pretrained or finetuned, used for models such as hubert"
    })
    encoder_tgt_layer: Optional[int] = field(default=None, metadata={
        "help": "0-indexed fairseq-native target layer for early-exit inside "
                "the transformer encoder. None = run all blocks / final layer "
                "(default). e.g. 20 = early-exit AFTER the 21st block. "
                "Currently consumed by encoder_name=data2vec_aqc."
    })
    encoder_path_hf: Optional[str] = field(default=None, metadata={
        "help": "HuggingFace whisper model id/dir (alternative to encoder_path)"
    })
    whisper_decode: bool = field(default=False, metadata={
        "help": "if true, load full Whisper model (encoder+decoder) instead of just encoder"
    })
    strip_frames_before_projector: bool = field(default=False, metadata={
        "help": "if true (hubert_hf only), slice encoder output frames to the audio "
                "token budget (modality_mask slots * ds_rate) BEFORE the projector, "
                "keeping only the leftmost frames. Pair with "
                "dataset_config.audio_frames_per_sec so the budget is duration-based "
                "(e.g. 15 frames/s) instead of the full 50 frames/s."
    })

@dataclass
class PeftConfig:
    peft_method: str = "lora" # None , llama_adapter, prefix
    r: int = 8
    lora_alpha: int = 32
    target_modules: List = field(default_factory=lambda: [ "q_proj", "v_proj" ])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    lora_dropout: float = 0.05
    inference_mode: bool = False

@dataclass
class TrainConfig:
    model_name:str = "PATH/to/LLAMA/7B"
    enable_ddp:bool = False
    enable_deepspeed:bool = False
    enable_fsdp:bool = False
    low_cpu_fsdp:bool = False
    run_validation:bool = True
    batch_size_training:int = 4
    batching_strategy:str = field(default="packing", metadata={
        "help":"alternative: padding"
    }) #
    context_length:int = 4096
    gradient_accumulation_steps:int = 1
    num_epochs:int = 3
    num_workers_dataloader:int = 1
    warmup_steps:int = 1000
    total_steps:int = 100000
    validation_interval:int = 1000
    checkpoint_interval:int = 1000  # decouples checkpoint-save cadence from
    # eval cadence (validation_interval) when set lower than it; defaults to
    # matching validation_interval's old value so scripts that don't pass it
    # keep the old coupled behavior.
    max_eval_steps:int = 0  # 0/unset = evaluate the full eval_dataloader each
    # validation call (old behavior). Set >0 to cap eval to that many batches
    # per call, so eval can run frequently without paying the full-dev-set cost.
    lr:float = 1e-4
    weight_decay:float = 0.0
    gamma:float = 0.85
    seed:int = 42
    use_fp16:bool = False
    mixed_precision:bool = True
    val_batch_size:int = 1

    use_peft:bool = False
    peft_config:PeftConfig = field(default_factory=PeftConfig)
    output_dir:str = "PATH/to/save/PEFT/model"
    freeze_layers:bool = False
    num_freeze_layers:int = 1
    quantization:bool = False
    one_gpu:bool = False
    save_model:bool = True
    dist_checkpoint_root_folder:str = "PATH/to/save/FSDP/model" # will be used if using FSDP
    dist_checkpoint_folder:str = "fine-tuned" # will be used if using FSDP
    save_optimizer:bool = False # will be used if using FSDP
    use_fast_kernels:bool = False # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    run_test_during_validation:bool = False
    run_test_during_validation_file:str = "test.wav"
    run_test_during_validation_prompt:str = "<|ASR|>"
    freeze_llm:bool = field(default=False, metadata={
        "help": "whether to freeze llm when finetuning, should be true when use peft finetuning"
    })
    freeze_encoder:bool = False
    resume_ckpt:Optional[str] = field(default=None, metadata={
        "help": "Path to a DeepSpeed checkpoint directory (the folder holding the "
                "`latest` file and global_stepN/, e.g. .../asr_epoch_1_step_16000). "
                "When set, the engine restores trainable weights + optimizer (AdamW "
                "moments) + LR scheduler + global step via model_engine.load_checkpoint(), "
                "and the epoch/step parsed from the dir name re-aligns the dataloader "
                "(skip-replay) so training continues in place instead of starting over."
    })

@dataclass
class DataConfig:
    dataset: str = "speech_dataset"
    file: str = "src/slam_llm/datasets/speech_dataset.py:get_speech_dataset"
    train_data_path: Optional[str] = None
    val_data_path: Optional[str] = None
    train_split: str = "train"
    test_split:str = "validation"
    prompt: Optional[str] = None
    data_path: Optional[str] = None
    max_words: Optional[int] = None
    max_mel: Optional[float] = None
    fix_length_audio: int = -1
    inference_mode:bool = False
    input_type: str = field(default="raw", metadata={
                                "help":"Use raw when input is wav, mel when for whisper"
                            })
    mel_size: int = field(default=80, metadata={
        "help": "80 for whisper large v1 and v2, 128 for v3"
    })
    normalize: Optional[bool] = field(default=False, metadata={
        "help": "whether input is normalized, used for models such as wavlm"
    })
    prompt_style: str = field(default="vicuna", metadata={
        "help": "prompt template: vicuna (USER:/ASSISTANT:), gemma2 (<start_of_turn>...), or qwen2 (ChatML)"
    })
    use_history_context: bool = field(default=False, metadata={
        "help": "MaLa-ASR style: if true, fill the {prev_context} placeholder in the "
                "prompt with each sample's prev_context field (empty string when the "
                "sample has no prior context). Every utterance uses the same prompt."
    })
    audio_frames_per_sec: float = field(default=-1.0, metadata={
        "help": "if > 0 (input_type=raw only), reserve audio token slots from "
                "duration as round(audio_frames_per_sec * seconds) // ds_rate "
                "instead of the full encoder rate (samples//320 = 50 frames/s). "
                "e.g. 15 keeps only the leftmost 15 frames per second of audio. "
                "Pair with model_config.strip_frames_before_projector."
    })
    encoder_projector_ds_rate: int = field(default=5, metadata={
        "help": "projector downsample rate used to convert encoder frames to LLM "
                "audio token slots; must match model_config.encoder_projector_ds_rate "
                "(historically hardcoded to 5 in the raw/mel branches)."
    })
    encoder_frame_offset: int = field(default=0, metadata={
        "help": "subtract this many frames from samples//320 before computing "
                "reserved audio-token slots, for encoders whose real frame count "
                "undershoots that formula by a fixed amount (verified: "
                "wav2vec2_bert's log-mel+stride-2 pipeline is always exactly 1 "
                "frame short -- set to 1 for encoder_name=wav2vec2_bert). Without "
                "this the dataset over-reserves slots and the extra slot gets "
                "scattered as an all-zero embedding into every training example."
    })

@dataclass
class FSDPConfig:
    mixed_precision: bool = True
    use_fp16: bool = False
    # sharding_strategy = "FULL_SHARD" #ShardingStrategy = ShardingStrategy.FULL_SHARD
    sharding_strategy: ShardingStrategy = "NO_SHARD" #ShardingStrategy.NO_SHARD #MZY: set NO_SHARD when use DDP
    checkpoint_type: str = "SHARDED_STATE_DICT"  # alternatively can use SHARDED_STATE_DICT save one file per rank, and can resize the world-size.
    fsdp_activation_checkpointing: bool = True
    fsdp_cpu_offload: bool = False
    pure_bf16: bool = False
    optimizer: str = "AdamW"

@dataclass
class LogConfig:
    use_wandb: bool = False
    wandb_dir: str = "/root/test_wandb"
    wandb_entity_name: str = "project_name"
    wandb_project_name: str = "project_name"
    wandb_exp_name: str = "exp_name"
    log_file: str = "/root/test.log"
    log_interval: int = 5
    resume_wandb_id: Optional[str] = field(default=None, metadata={
        "help": "W&B run ID to resume (e.g. giob0w7s). When set, training appends to "
                "the existing run's charts instead of starting a new run."
    })
