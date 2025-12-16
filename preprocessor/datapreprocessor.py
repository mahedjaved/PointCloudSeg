import logging
from pathlib import Path
from typing import Dict, List, Tuple

import geopandas as gpd
import laspy
import numpy as np

from config import get_config
from preprocessor.augmentation import PointCloudPreprocessor


logger = logging.getLogger(__name__)


def build_species_mapping() -> Dict[str, int]:
    return {
        "Birch": 0,
        "Aspen": 1,
        "Fir": 2,
        "Spruce": 3,
        "Alder": 4,
        "Tilia": 5,
        "Willow": 6,
        "Elm": 7,
        "Pine": 8,
    }


def extract_tree_points(
    pts_xyz: np.ndarray,
    pts_features: np.ndarray,
    tree_xy: Tuple[float, float],
    radius: float,
) -> np.ndarray:
    tx, ty = tree_xy
    dx = pts_xyz[:, 0] - tx
    dy = pts_xyz[:, 1] - ty
    dist2 = dx * dx + dy * dy
    mask = dist2 <= radius * radius
    return pts_features[mask]


def process_plot(
    las_path: Path,
    gdf_plot: gpd.GeoDataFrame,
    preprocessor: PointCloudPreprocessor,
    species_to_id: Dict[str, int],
    out_samples_dir: Path,
    radius: float,
    augment_repeats: int,
) -> List[str]:
    las = laspy.read(las_path)
    pts_xyz = np.vstack([las.x, las.y, las.z]).T.astype(np.float32)
    if hasattr(las, "intensity"):
        intensity = las.intensity.astype(np.float32)[:, None]
        pts_all = np.concatenate([pts_xyz, intensity], axis=1)
    else:
        pts_all = pts_xyz

    sample_ids: List[str] = []
    plot_id_str = las_path.stem

    for _, row in gdf_plot.iterrows():
        species = str(row["species"])
        if species not in species_to_id:
            continue
        label_id = species_to_id[species]

        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        tree_xy = (geom.x, geom.y)

        tree_pts = extract_tree_points(pts_xyz, pts_all, tree_xy, radius)
        if tree_pts.shape[0] == 0:
            continue

        labels_point = np.full(tree_pts.shape[0], label_id, dtype=np.int64)

        tree_no = int(row["tree_no"])
        base_id = f"{plot_id_str}_tree{tree_no:04d}"

        for k in range(augment_repeats):
            pts_proc, labels_proc, _ = preprocessor.process(
                tree_pts, labels_point, augment=True
            )
            label_scalar = int(labels_proc[0])
            sid = f"{base_id}_aug{k:02d}"
            out_path = out_samples_dir / f"{sid}.npz"
            np.savez(out_path, points=pts_proc.astype(np.float32), label=label_scalar)
            sample_ids.append(sid)

    return sample_ids


def run_preprocessing(
    radius: float = 2.0,
    augment_repeats: int = 2,
) -> None:
    logging.basicConfig(level=logging.INFO)
    cfg = get_config()

    data_dir = cfg.dataset.data_dir
    processed_dir = cfg.dataset.processed_dir

    als_dir = data_dir / "als"
    geojson_path = data_dir / "field_survey.geojson"

    out_samples_dir = processed_dir / "samples"
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_samples_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading field survey GeoJSON from %s", geojson_path)
    gdf = gpd.read_file(geojson_path)

    preprocessor = PointCloudPreprocessor(cfg.preprocessing)
    species_to_id = build_species_mapping()

    all_sample_ids: List[str] = []

    for las_path in sorted(als_dir.glob("*.las")):
        logger.info("Processing ALS file %s", las_path.name)
        stem = las_path.stem
        parts = stem.split("_")
        plot_token = parts[-1]
        try:
            plot_id = float(plot_token)
        except ValueError:
            logger.warning("Could not parse plot id from %s", stem)
            continue

        gdf_plot = gdf[gdf["plot"] == plot_id]
        if gdf_plot.empty:
            logger.warning("No trees found in GeoJSON for plot %s", plot_id)
            continue

        ids_plot = process_plot(
            las_path=las_path,
            gdf_plot=gdf_plot,
            preprocessor=preprocessor,
            species_to_id=species_to_id,
            out_samples_dir=out_samples_dir,
            radius=radius,
            augment_repeats=augment_repeats,
        )
        all_sample_ids.extend(ids_plot)

    if not all_sample_ids:
        logger.warning("No samples were generated")
        return

    all_sample_ids = sorted(all_sample_ids)
    n_total = len(all_sample_ids)
    # Split into train and val via 80/20 split
    n_train = int(0.8 * n_total)
    train_ids = all_sample_ids[:n_train]
    val_ids = all_sample_ids[n_train:]

    (processed_dir / "train.txt").write_text("\n".join(train_ids))
    (processed_dir / "val.txt").write_text("\n".join(val_ids))

    logger.info("Generated %d samples (%d train, %d val)", n_total, len(train_ids), len(val_ids))


if __name__ == "__main__":
    run_preprocessing()