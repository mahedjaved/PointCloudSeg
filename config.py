"""
Description:Configuration Management for Point Cloud Segmentation Pipeline
all hyperparameters and settings are defined here for easy modification.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import yaml


@dataclass
class DatasetConfig:
    """Dataset-specific configuration"""
    
    name: str = "Tree-Detection-LIDAR-RGB"
    base_url: str = "https://www.kaggle.com/datasets/sentinel3734/tree-detection-lidar-rgb/data"
    data_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    num_classes: int = 9
    ignore_label: int = 0


@dataclass
class PreprocessingConfig:
    """Data preprocessing configuration"""
    
    # Normalization
    normalize = True
    center_origin = True
    unit_sphere = False
    
    # Voxelization
    voxelize = True
    voxel_size = 0.05  # 5cm voxels
    
    # Sampling
    num_points = 4096  # Fixed number of points per sample
    sampling_method = "fps"  # 'fps' or 'random'
    
    # Augmentation
    augment = True
    rotate_z = (-180.0, 180.0)  # degrees
    scale_range = (0.95, 1.05)
    jitter_std = 0.01
    dropout_ratio = 0.2
    flip_prob = 0.5
    
    # Feature engineering
    use_intensity = True
    use_coordinates = True
    compute_normals = False
    
    # Class balancing
    class_balanced_sampling = True
    oversample_minority = True
    
    # Multiprocessing
    num_workers = 4
    
    # Random seed for reproducibility
    seed = 42

@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    
    optimizer: str = "adam"
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    momentum: float = 0.9
    
    lr_scheduler: str = "cosine"
    lr_decay_rate: float = 0.1
    lr_decay_epochs: List[int] = field(default_factory=lambda: [100, 150])
    
    num_epochs: int = 200
    batch_size: int = 8
    accumulation_steps: int = 1
    
    loss_function: str = "cross_entropy"
    class_weights: Optional[List[float]] = None
    ignore_index: int = 0
    
    label_smoothing: float = 0.0
    mixup_alpha: float = 0.0
    
    eval_frequency: int = 5
    save_frequency: int = 10
    
    patience: int = 20
    min_delta: float = 0.001
    
    checkpoint_dir: Path = Path("checkpoints")
    save_best_only: bool = True
    
    log_dir: Path = Path("logs")
    log_frequency: int = 10
    
    device: str = "cuda"
    mixed_precision: bool = True
    num_workers: int = 4
    pin_memory: bool = True
    
    seed: int = 42
    deterministic: bool = True


@dataclass
class ModelConfig:
    name: str = "pointnet"
    in_channels: int = 4
    num_classes: int = 9


@dataclass
class Config:
    """Master configuration combining all sub-configs"""
    
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Project metadata
    project_name = "pointcloud_segmentation"
    version = "0.1.0"
    description = "Point Cloud Semantic Segmentation Pipeline"
    
    def validate(self) -> bool:
        """Validate configuration parameters"""
        errors = []
        
        if self.dataset.num_classes < 2:
            errors.append("num_classes must be >= 2")
        
        if self.preprocessing.voxel_size <= 0:
            errors.append("voxel_size must be positive")
        if self.preprocessing.num_points <= 0:
            errors.append("num_points must be positive")
        
        if errors:
            raise ValueError(f"Configuration validation failed:\n" + "\n".join(errors))
        
        return True

    # optional
    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        dataset_data = data.get("dataset", {})
        preprocessing_data = data.get("preprocessing", {})
        dataset = DatasetConfig(**dataset_data)
        preprocessing = PreprocessingConfig(**preprocessing_data)
        cfg = cls(dataset=dataset, preprocessing=preprocessing)
        cfg.project_name = data.get("project_name", cfg.project_name)
        cfg.version = data.get("version", cfg.version)
        cfg.description = data.get("description", cfg.description)
        return cfg

    def to_yaml(self, path: str) -> None:
        def _serialize(obj: Any) -> Any:
            if isinstance(obj, Path):
                return str(obj)
            if isinstance(obj, dict):
                return {k: _serialize(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_serialize(v) for v in obj]
            if hasattr(obj, "__dict__"):
                return _serialize(obj.__dict__)
            return obj

        data = _serialize(
            {
                "dataset": self.dataset,
                "preprocessing": self.preprocessing,
                "project_name": self.project_name,
                "version": self.version,
                "description": self.description,
            }
        )
        with open(path, "w") as f:
            yaml.safe_dump(data, f)


# Default configuration instance - serving as a singleton copy shared across modules
DEFAULT_CONFIG = Config()


def get_config(config_path: Optional[str] = None) -> Config:
    """
    Get configuration object.
    
    Args:
        config_path: Path to YAML config file. If None, returns default config.
    
    Returns:
        Config object
    """
    if config_path:
        config = Config.from_yaml(config_path)
    else:
        config = DEFAULT_CONFIG
    
    config.validate()
    return config


if __name__ == "__main__":
    config = get_config()
    print(f"Project: {config.project_name} v{config.version}")
    print(f"Dataset: {config.dataset.name}")
    
    config.to_yaml("configs/default.yaml")
    print("\nDefault configuration saved to configs/default.yaml")