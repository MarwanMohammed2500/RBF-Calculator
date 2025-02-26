# RBF (Radial Basis Function) Visualizer

## Overview

This project is a Streamlit-based interactive tool for visualizing the Radial Basis Function (RBF) transformation of 2D data points. Users can input data points, define centroids, and observe how RBF transformations affect the dataset.

## Features

- **Dynamic Data Input**: Users can input data points through an interactive table.
- **RBF Calculation**: Computes squared distances (`r1^2`, `r2^2`) from two centroids.
- **Radial Basis Function Transformation**: Applies the Gaussian RBF transformation to the distances.
- **Visualization**: Plots the original and transformed data points.
- **Interactive UI**: Built with Streamlit for a seamless user experience.

## Installation

To run this project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/MarwanMohammed2500/RBF-Calculator
   cd RBF-Calculator
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run main.py
   ```

## Dependencies

Ensure you have the following Python libraries installed:

- `numpy`
- `pandas`
- `matplotlib`
- `streamlit`

## Usage

1. **Input Data**: Use the editable table to enter data points (`category`, `x1`, `x2`).
2. **Define Centroids**: Enter centroid values `c1` and `c2` as comma-separated values.
3. **Set Variance (𝝈²)**: Provide a numerical value for the variance parameter.
4. **Calculate RBF Features**: The app computes the squared distances and transformed features (`∅1`, `∅2`).
5. **Visualize Data**: View scatter plots of original and transformed feature space.
6. **Review Data**: Display the full transformed dataset in a table format.

## Functionality

The core functionality is implemented in the `RBF` class:

- `calc_r()`: Computes squared distances from centroids.
- `calc_phi()`: Applies the Gaussian RBF transformation.
- `plot_x()`: Plots the original data in feature space.
- `plot_phi()`: Plots transformed RBF features.
- `show_table()`: Displays the processed data table.
