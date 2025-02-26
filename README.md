# RBF (Radial Basis Function) Calculator 

## Overview

This project is a Streamlit-based interactive tool for calculating and visualizing the Radial Basis Function (RBF) transformation of 2D data points. Users can input data points, define centroids, and observe how RBF transformations affect the dataset.

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

## Screenshots from the app
### When you open the app
![image](https://github.com/user-attachments/assets/f5dd9eb4-779f-437a-9985-2fadb06b0d4a)

Once you fill the required data fields, click on "Calculate R"

### Calculate R
![image](https://github.com/user-attachments/assets/8f9e1d26-7951-4be7-bd79-11a806091353)

Then Click on "Calculate Phi"

### Calculate Phi
![image](https://github.com/user-attachments/assets/cd9498b6-d7b8-425b-ab52-50d357636b07)
Then it shows the updated table with the original and transformed spaces, then you can click on "Plot", it shows the plots for both the original and transformed spaces

![image](https://github.com/user-attachments/assets/8fd8bfce-ecc6-41d1-9f13-58d73f7fac3c)
![image](https://github.com/user-attachments/assets/be5573a8-21e1-4a47-a668-80e9ee04ba38)


## Video Demo
https://github.com/user-attachments/assets/3a89b11c-5eea-498d-b374-fe95a7f469ac

# Test it out yourself if you want through (https://rbf-calculator.streamlit.app/)
