import yaml
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    vocab_size: int = 50260
    block_size: int = 512
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.0
    bias: bool = False
    tie_weights: bool = True


@dataclass
class DataConfig:
    data_dir: str = "data/processed"
    num_workers: int = 4


@dataclass
class TrainingConfig:
    batch_size: int = 32
    gradient_accumulation_steps: int = 4
    learning_rate: float = 6.0e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    max_iters: int = 100000
    warmup_iters: int = 2000
    lr_decay_iters: int = 100000
    min_lr: float = 6.0e-5
    eval_interval: int = 500
    eval_iters: int = 200
    log_interval: int = 10
    save_interval: int = 1000
    out_dir: str = "checkpoints"
    dtype: str = "float16"
    compile: bool = False


@dataclass
class GenerationConfig:
    max_new_tokens: int = 256
    temperature: float = 0.8
    top_k: int = 50


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)


def load_config(path):
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return Config(
        model=ModelConfig(**raw.get("model", {})),
        data=DataConfig(**raw.get("data", {})),
        training=TrainingConfig(**raw.get("training", {})),
        generation=GenerationConfig(**raw.get("generation", {})),
    )
