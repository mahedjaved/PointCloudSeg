import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import get_config


def load_data(config_path) -> gpd.GeoDataFrame:
    cfg = get_config(config_path)
    geojson_path = cfg.dataset.data_dir / "field_survey.geojson"
    gdf = gpd.read_file(geojson_path)
    return gdf


def plot_species_counts(gdf: gpd.GeoDataFrame, out_dir: Path):
    counts = gdf["species"].value_counts().sort_values(ascending=False)
    plt.figure(figsize=(8, 4))
    counts.plot(kind="bar")
    plt.ylabel("count")
    plt.xlabel("species")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "species_counts.png", dpi=200)
    plt.close()


def plot_height_distribution(gdf: gpd.GeoDataFrame, out_dir: Path):
    heights = gdf["height"].dropna()
    plt.figure(figsize=(6, 4))
    plt.hist(heights, bins=30)
    plt.xlabel("height")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_dir / "height_histogram.png", dpi=200)
    plt.close()


def plot_age_vs_height(gdf: gpd.GeoDataFrame, out_dir: Path):
    subset = gdf.dropna(subset=["age", "height"])
    plt.figure(figsize=(6, 5))
    species = subset["species"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(species)))
    for c, sp in zip(colors, species):
        d = subset[subset["species"] == sp]
        plt.scatter(d["age"], d["height"], s=10, alpha=0.5, label=sp, color=c)
    plt.xlabel("age")
    plt.ylabel("height")
    plt.legend(markerscale=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "age_vs_height_by_species.png", dpi=200)
    plt.close()


def plot_height_box_by_species(gdf: gpd.GeoDataFrame, out_dir: Path):
    subset = gdf.dropna(subset=["height"])
    plt.figure(figsize=(8, 4))
    species_order = (
        subset.groupby("species")["height"].median().sort_values(ascending=False).index
    )
    data = [subset[subset["species"] == sp]["height"] for sp in species_order]
    plt.boxplot(data, labels=species_order, showfliers=False)
    plt.ylabel("height")
    plt.xlabel("species")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "height_boxplot_by_species.png", dpi=200)
    plt.close()


def plot_correlation_matrix(gdf: gpd.GeoDataFrame, out_dir: Path):
    num_cols = gdf.select_dtypes(include=["number"])
    corr = num_cols.corr()
    plt.figure(figsize=(6, 5))
    im = plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    labels = corr.columns.tolist()
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.yticks(range(len(labels)), labels)
    plt.tight_layout()
    plt.savefig(out_dir / "correlation_matrix.png", dpi=200)
    plt.close()


def plot_species_spatial(gdf: gpd.GeoDataFrame, out_dir: Path):
    plt.figure(figsize=(6, 6))
    gdf.plot(column="species", legend=True, markersize=5)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(out_dir / "species_spatial_distribution.png", dpi=200)
    plt.close()


def calculate_iqr_by_species(gdf: gpd.GeoDataFrame):
    subset = gdf.dropna(subset=["height"])
    species_list = subset["species"].unique()

    print("\n" + "="*60)
    print("IQR Analysis for Height by Species")
    print("="*60)

    for species in sorted(species_list):
        species_data = subset[subset["species"] == species]["height"]
        n_samples = len(species_data)

        if n_samples >= 4:
            q1 = species_data.quantile(0.25)
            q3 = species_data.quantile(0.75)
            iqr = q3 - q1
            median = species_data.median()

            print(f"\n{species}:")
            print(f"  Sample count: {n_samples}")
            print(f"  Q1 (25th percentile): {q1:.2f} m")
            print(f"  Median (50th percentile): {median:.2f} m")
            print(f"  Q3 (75th percentile): {q3:.2f} m")
            print(f"  IQR (Q3 - Q1): {iqr:.2f} m")
        else:
            print(f"\n{species}:")
            print(f"  Sample count: {n_samples}")
            print(f"  IQR: Cannot be calculated (insufficient data, need ≥4 samples)")
            if n_samples > 0:
                print(f"  Available heights: {sorted(species_data.values)}")

    print("\n" + "="*60)
    print("Note: IQR requires at least 4 data points to calculate Q1, median, and Q3")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    out_dir = Path("eda") / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    gdf = load_data(args.config)

    calculate_iqr_by_species(gdf)

    plot_species_counts(gdf, out_dir)
    plot_height_distribution(gdf, out_dir)
    plot_age_vs_height(gdf, out_dir)
    plot_height_box_by_species(gdf, out_dir)
    plot_correlation_matrix(gdf, out_dir)
    plot_species_spatial(gdf, out_dir)


if __name__ == "__main__":
    main()

