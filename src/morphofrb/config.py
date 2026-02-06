from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class Config:
    batch_size:int = 32
    num_epochs:int = 500
    learning_rate: float = 1e-3
    set_lr_scheduler: bool = True
    learning_rate_milestones: List[int] = field(default_factory = lambda: [15, 30, 50, 65, 80, 100, 120, 140, 160, 180, 200])
    early_stopping: bool = True
    patience: int = 20
    weighted_sampling: bool = True
    fine_tune_chime_cattwo: bool = False # Fine tune on already fine tuned on cat-2 data
    replacement_sampling: bool = True  # Replacement sampling or not with weighted sampler
    shuffle_train: bool = False   # Set it to False if WeightedRandomSampler enabled on dataloader
    shuffle_val: bool = False
    train_dir: str = "/scratch/bk0055/frb_rnr_2025_jan_22/dataset/train"#'../dataset/trai'
    val_dir: str = "/scratch/bk0055/frb_rnr_2025_jan_22/dataset/validation"#'../dataset/validation'
    model_weight: str = "DEFAULT"
    learning_rate_scheduler: bool = True
    scheduler_gamma: float = 0.1
    early_stopping: bool = True
    print_training_details: bool = True
    feature_layers_to_unfreeze: Dict[int, List[int]] = field(default_factory=lambda: {0: [0]})
    unfreeze_classification_layer: bool = True
    
    set_workers: bool = True   # Set the number of cores for data loading(setting up workers on mac gives error)
    num_workers: int = 16     # Intended for HPC use