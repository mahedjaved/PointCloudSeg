# Model Choice: PointNet++ for Tree-Level Species Segmentation

This project tackles **single-tree species segmentation** from airborne laser scanning (ALS) point clouds using a PointNet++-inspired architecture. Below is a rationale for this choice, followed by a critical, research-level evaluation in the context of forest remote sensing.

## Why PointNet++ Is a Reasonable Choice

- **Native operation on raw point clouds**  
  ALS data are irregular, sparse, and anisotropic (distributed unevenly in different directions). PointNet++-style models operate directly on unordered point sets, avoiding rasterization or voxelization that can blur canopy structure or waste resolution in empty space. This is true for majority of deep learning models that prioritize preserving fine-grained spatial details while handling irregular input distributions.

- **Permutation invariance with learned geometry**  
  In brief, using PointNet++ design guarantees that shuffling the input points won't change the output (permutation invariance). The architecture combines point-wise shared MLPs (multi-layer perceptrons) with max pooling, giving strict invariance to point ordering while still learning meaningful spatial patterns from local neighborhoods and global context.

- **Scales to varying point densities**  
  Forest ALS campaigns often mix high- and low-density strips. Real-world forest ALS data tend to be messy. Different flight lines may scan the same area at different altitudes or angles, creating varying point densities. Some trees might be partially occluded by others, and data quality can vary across the survey area, different flight lines, and complex occlusions. PointNet++ was specifically designed to handle non-uniform sampling through its hierarchical feature abstraction. Ot can learn features from both dense and sparse regions. This makes it robust to the inconsistent data quality typical in operational forestry applications.

- **Computationally light for small trees and small datasets**  
  At the tree level, each sample has relatively few points (e.g. 2–5k). This would have been computationally expensive had we gone for 3D CNN approaches that operate on volumetric grids. The solution would have been computationally expensive and memory-intensive. Compared to 3D sparse CNNs on volumetric grids, a PointNet++-inspired network offers a favorable trade-off between capacity and computational cost, which is important given the modest dataset size.

- **Well-aligned with the preprocessing pipeline** 
  In our augmentation and preprocessing pipeline we make use of normalization, optional voxel downsampling. This means that every tree is represented in a consistent coordinate system with similar scale and position. PointNet++-style architectures excel with this kind of preprocessed data because the network can focus on learning discriminative shape features rather than having to deal with arbitrary translations, rotations, or scale variations.


  Reference

  [1] Qi, C.R., Su, H., Mo, K., & Guibas, L. J. (2017). PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR). arXiv:1612.00593
  [2] Qi, C. R., Yi, L., Su, H., & Guibas, L. J. (2017). PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space. Advances in Neural Information Processing Systems (NeurIPS). arXiv:1706.02413
  [3] Zaheer, M., Kottur, S., Ravanbakhsh, S., Poczos, B., Salakhutdinov, R., & Smola, A. (2017). Deep Sets. Advances in Neural Information Processing Systems (NeurIPS). arXiv:1703.06114