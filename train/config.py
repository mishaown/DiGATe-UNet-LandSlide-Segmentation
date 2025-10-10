from dataclasses import dataclass

@dataclass
class Config:
    num_epochs: int
    learning_rate: float
    weight_decay: float
    batch_size: int
    model_save_path: str
    device: str 