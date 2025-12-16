"""
Point Cloud Preprocessing Pipeline

Implements data preprocessing with SOLID principles:
- Single Responsibility: Each class handles one preprocessing aspect
- Open/Closed: Each augmentation strategy is a separate class, and can be extended without modifying existing code
- Liskov Substitution: All concrete implementations of augmentation strategies are interchangeable
- Interface Segregation: Single responsibilty but applied to interfaces 
- Dependency Inversion: All concrete implementations of augmentation strategies must depend upon the abstract base class (AugmentationStrategy)

References:
    - Advancements in Point Cloud Data Augmentation for Deep Learning: A Survey
Qinfeng Zhua,b, Lei Fana
    - Code heavily inspired from: https://torch-points3d.readthedocs.io/en/latest/src/api/transforms.html
    - and https://www.open3d.org/docs/0.6.0/python_api/open3d.geometry.html
    - Downsampling in Point Clouds: https://medium.com/@lathwath5/point-cloud-downsampling-methods-and-python-implementations-6f91a129ac48
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional, List
import logging
from pathlib import Path

from config import PreprocessingConfig, get_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Abstract Base Classes (Dependency Inversion Principle)
# ============================================================================

# TODO: expand this to suit geometric transformations, normalization based transformatiosn etc.
class AugmentationStrategy(ABC):
    """Abstract base class for augmentation strategies"""
    
    @abstractmethod
    def apply(self, points: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply augmentation to point cloud and labels"""
        pass


# ============================================================================
# Basic Geometric Transformations
# ============================================================================

class RandomRotationZ(AugmentationStrategy):
    """rotate point cloud around Z-axis"""
    
    def __init__(self, angle_range: Tuple[float, float] = (-180, 180)):
        """
        Args:
            angle_range: (min, max) rotation angles in degrees
        """
        self.angle_range = angle_range
    
    def apply(self, points: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """rotate around Z-axis"""
        angle = np.random.uniform(*self.angle_range)
        angle_rad = np.deg2rad(angle)
        
        cos_theta = np.cos(angle_rad)
        sin_theta = np.sin(angle_rad)
        
        rotation_matrix = np.array([
            [cos_theta, -sin_theta, 0],
            [sin_theta, cos_theta, 0],
            [0, 0, 1]
        ])
        
        # X = X^prime @ R.T
        points[:, :3] = points[:, :3] @ rotation_matrix.T
        return points, labels


class RandomScaling(AugmentationStrategy):
    """randomly scale point cloud"""
    
    def __init__(self, scale_range: Tuple[float, float] = (0.95, 1.05)):
        """
        Args:
            scale_range: (min, max) scaling factors
        """
        self.scale_range = scale_range
    
    def apply(self, points: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """apply random scaling"""
        scale = np.random.uniform(*self.scale_range)
        points[:, :3] *= scale
        return points, labels


class RandomJittering(AugmentationStrategy):
    """add random noise to point coordinates"""
    
    def __init__(self, std: float = 0.01):
        """
        Args:
            std: Standard deviation of Gaussian noise
        """
        self.std = std
    
    def apply(self, points: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """add Gaussian noise"""
        noise = np.random.normal(0, self.std, size=points[:, :3].shape)
        points[:, :3] += noise
        return points, labels


class RandomDropout(AugmentationStrategy):
    """randomly drop points to simulate occlusion"""
    
    def __init__(self, dropout_ratio: float = 0.2):
        """
        Args:
            dropout_ratio: Fraction of points to drop
        """
        self.dropout_ratio = dropout_ratio
    
    def apply(self, points: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Randomly drop points"""
        n_points = len(points)
        n_keep = int(n_points * (1 - self.dropout_ratio))
        
        indices = np.random.choice(n_points, n_keep, replace=False)
        # Maintain order of the indices
        indices = np.sort(indices)  
        
        return points[indices], labels[indices]


class RandomFlip(AugmentationStrategy):
    """randomly flip point cloud along an axis"""
    
    def __init__(self, axis: int = 0, prob: float = 0.5):
        """
        Args:
            axis: axis to flip (0=X, 1=Y, 2=Z)
            prob: probability of flipping
        """
        self.axis = axis
        self.prob = prob
    
    def apply(self, points: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """randomly flip along axis"""
        if np.random.random() < self.prob:
            points[:, self.axis] *= -1
        return points, labels


# ============================================================================
# Normalization Based Preprocessing Components
# ============================================================================

class PointCloudNormalizer:
    """
    normalizes point cloud coordinates, intensity values and features.
    """
    
    def __init__(
        self,
        center_origin = True,
        unit_sphere = False,
        normalize_intensity = True
    ):
        """
        Args:
            center_origin: Center point cloud at origin
            unit_sphere: Scale to unit sphere
            normalize_intensity: Normalize intensity values to [0, 1]
        """
        self.center_origin = center_origin
        self.unit_sphere = unit_sphere
        self.normalize_intensity = normalize_intensity
    
    def normalize(self, points) -> Tuple[np.ndarray, dict]:
        """
        normalize point cloud.
        
        Args:
            points: (N, 3+) array with [x, y, z, intensity]
            
        Returns:
            normalized_points: Normalized point cloud
            stats: Dictionary with normalization statistics
        """
        # Done to avoid external modification
        points = points.copy()
        stats = {}
        
        # Center at origin
        if self.center_origin:
            centroid = np.mean(points[:, :3], axis=0)
            # Only the first three columns are spatial!
            points[:, :3] -= centroid
            stats['centroid'] = centroid
        
        # Scale to a unit sphere
        if self.unit_sphere:
            # Using norm vector for scaling
            max_dist = np.max(np.linalg.norm(points[:, :3], axis=1))
            if max_dist > 0:
                points[:, :3] /= max_dist
                stats['max_dist'] = max_dist
        
        # Normalize intensity using min-max scaling
        if self.normalize_intensity and points.shape[1] > 3:
            # Gather all intensity values
            intensity = points[:, 3]
            # Going to scour all intensity values to find min and max
            intensity_min = np.min(intensity)
            intensity_max = np.max(intensity)
            # Avoid division by zero
            if intensity_max > intensity_min:
                points[:, 3] = (intensity - intensity_min) / (intensity_max - intensity_min)
                stats['intensity_range'] = (intensity_min, intensity_max)
        
        return points, stats


class VoxelDownsampler:
    """
    voxel grid downsampling.
    """
    
    def __init__(self, voxel_size: float = 0.05):
        """
        Args:
            voxel_size: size of voxel grid in meters
        """
        self.voxel_size = voxel_size
    
    def downsample(
        self,
        points: np.ndarray,
        labels: np.ndarray,
        aggregation: str = 'mean'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        downsample point cloud using voxel grid. The idea is to aggregate the points into voxels of a given size, and then return one point per occupied voxel.
        
        Args:
            points: (N, D) point cloud
            labels: (N,) labels
            aggregation: How to aggregate features ('mean', 'max', 'random')
            
        Returns:
            downsampled_points: (M, D) where M < N
            downsampled_labels: (M,)
        """
        # Compute voxel indices
        voxel_indices = np.floor(points[:, :3] / self.voxel_size).astype(np.int32)
        
        # Create unique voxel keys
        voxel_keys = {}
        for i, voxel_idx in enumerate(voxel_indices):
            key = tuple(voxel_idx)
            if key not in voxel_keys:
                voxel_keys[key] = []
            voxel_keys[key].append(i)
        
        # Aggregate points in each voxel
        downsampled_points = []
        downsampled_labels = []
        
        for indices in voxel_keys.values():
            if aggregation == 'mean':
                downsampled_points.append(np.mean(points[indices], axis=0))
                # For labels, use majority vote
                unique, counts = np.unique(labels[indices], return_counts=True)
                downsampled_labels.append(unique[np.argmax(counts)])
            
            elif aggregation == 'max':
                # Max pooling for features
                downsampled_points.append(np.max(points[indices], axis=0))
                unique, counts = np.unique(labels[indices], return_counts=True)
                downsampled_labels.append(unique[np.argmax(counts)])
            
            elif aggregation == 'random':
                # Random sampling
                idx = np.random.choice(indices)
                downsampled_points.append(points[idx])
                downsampled_labels.append(labels[idx])
        
        return np.array(downsampled_points), np.array(downsampled_labels)


# ============================================================================
# Main Preprocessing Pipeline
# ============================================================================

class PointCloudPreprocessor:
    """
    Complete preprocessing pipeline.
    Composes multiple preprocessing components.
    """
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """
        Args:
            config: Preprocessing configuration
        """
        self.config = config or PreprocessingConfig()
        
        # Initialize components
        self.normalizer = PointCloudNormalizer(
            center_origin=self.config.center_origin,
            unit_sphere=self.config.unit_sphere,
            normalize_intensity=True
        )
        
        self.voxelizer = VoxelDownsampler(
            voxel_size=self.config.voxel_size
        ) if self.config.voxelize else None
        
        # Initialize augmentation pipeline
        self.augmentations = self._build_augmentation_pipeline()
        
        # Set random seed
        np.random.seed(self.config.seed)
    
    def _build_augmentation_pipeline(self) -> List[AugmentationStrategy]:
        """Build augmentation pipeline from config"""
        augmentations = []
        
        if self.config.augment:
            augmentations.extend([
                RandomRotationZ(self.config.rotate_z),
                RandomScaling(self.config.scale_range),
                RandomJittering(self.config.jitter_std),
                RandomDropout(self.config.dropout_ratio),
                RandomFlip(axis=0, prob=self.config.flip_prob)
            ])
        
        return augmentations
    
    def process(
        self,
        points: np.ndarray,
        labels: np.ndarray,
        augment: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Apply full preprocessing pipeline.
        
        Args:
            points: (N, 3+) point cloud [x, y, z, intensity?, ...]
            labels: (N,) semantic labels
            augment: Whether to apply augmentations
            
        Returns:
            processed_points: (num_points, D) preprocessed point cloud
            processed_labels: (num_points,) preprocessed labels
            metadata: Dictionary with preprocessing statistics
        """
        metadata = {}
        
        # 1. Normalize
        points, norm_stats = self.normalizer.normalize(points)
        metadata['normalization'] = norm_stats
        
        # 2. Voxelize (optional)
        if self.voxelizer:
            original_size = len(points)
            points, labels = self.voxelizer.downsample(points, labels)
            metadata['voxelization'] = {
                'original_size': original_size,
                'voxelized_size': len(points),
                'reduction_ratio': len(points) / original_size
            }
        
        # 3. Augment (if training)
        if augment and self.augmentations:
            for aug in self.augmentations:
                points, labels = aug.apply(points, labels)
            metadata['augmentation'] = 'applied'
        
        #TODO: add sampling to fixed size
        
        return points, labels, metadata


def main():
    sample_files = sorted(processed_dir.glob("*.npz"))
    sample = np.load(sample_files[0])
    points = sample["points"]
    label_scalar = int(sample["label"])
    labels = np.full(points.shape[0], label_scalar, dtype=np.int64)
    logger.info(f"Loaded sample from {sample_files[0].name} with {points.shape[0]} points")
    
    # Create preprocessor
    config = PreprocessingConfig(
        voxel_size=0.05,
        num_points=4096,
        augment=True
    )
    preprocessor = PointCloudPreprocessor(config)
    
    # Process
    processed_points, processed_labels, metadata = preprocessor.process(
        points, labels, augment=True
    )
    
    logger.info(f"\nProcessed point cloud: {processed_points.shape}")
    logger.info(f"Processed labels: {processed_labels.shape}")
    logger.info(f"\nMetadata:")
    for key, value in metadata.items():
        logger.info(f"  {key}: {value}")


if __name__ == "__main__":
    main()
