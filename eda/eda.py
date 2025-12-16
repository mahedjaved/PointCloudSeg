# ==============================================================
# Imports and configuration
# ==============================================================

# IMPORT command-line argument parser
# IMPORT filesystem path handling
# IMPORT geospatial dataframe library
# IMPORT plotting library
# IMPORT numerical computation library
# IMPORT data manipulation library

# IMPORT configuration loader


# ==============================================================
# Load field survey data
# ==============================================================

# FUNCTION load_data(config_path):
#     LOAD configuration using config_path
#     CONSTRUCT path to field survey GeoJSON
#     READ GeoJSON into geospatial dataframe
#     RETURN geodataframe


# ==============================================================
# Plotting functions for exploratory data analysis
# ==============================================================

# FUNCTION plot_species_counts(gdf, output_directory):
#     COUNT occurrences of each species
#     SORT counts descending
#     CREATE bar chart of species counts
#     LABEL axes
#     ROTATE x-axis labels for readability
#     SAVE figure to output_directory

# FUNCTION plot_height_distribution(gdf, output_directory):
#     EXTRACT height column, dropping missing values
#     CREATE histogram of heights
#     LABEL axes
#     SAVE figure to output_directory

# FUNCTION plot_age_vs_height(gdf, output_directory):
#     DROP rows with missing age or height
#     GET unique species and assign colors
#     FOR each species:
#         FILTER rows for species
#         PLOT scatter of age vs height
#     LABEL axes
#     ADD legend
#     SAVE figure to output_directory

# FUNCTION plot_height_box_by_species(gdf, output_directory):
#     Pre-processing stages below ----
#     DROP rows with missing height
#     ORDER species by median height
#     COLLECT height data for each species
#     CREATE boxplot for each species
#     LABEL axes
#     ROTATE x-axis labels
#     SAVE figure to output_directory


# ==============================================================
# Main script
# ==============================================================

# FUNCTION main():
#     PARSE command-line arguments for optional config path
#     CREATE output directory for figures
#     LOAD geodataframe using load_data
#     CALL all plotting functions to generate figures


# ==============================================================
# Script entry point
# ==============================================================

# IF script is run directly:
#     CALL main()
