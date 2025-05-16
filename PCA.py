# PCA_GUI.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class PCA:
    def __init__(self):
        """
        Initialize PCA with a DataFrame and the selected columns.
        
        Parameters:
        data (pd.DataFrame): The input dataset.
        columns (list): The two columns on which PCA will be applied.
        """
        self.data = None
        self.columns = None
        self.mean_vector = None
        self.covariance_matrix = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.transformed_data = None

    def get_input(self):
        self.choice = st.selectbox("Do you have a dataset file (csv) or will you manually insert data?",
                              ("Choose", "I have a CSV file", "I'll manually insert my data"))
        if self.choice == "I have a CSV file":
            uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
            if uploaded_file is not None:
                self.data = pd.read_csv(uploaded_file)
                data_matrix = st.data_editor(self.data, num_rows="dynamic")
        elif self.choice == "I'll manually insert my data":
            number_of_features = st.text_input("How many features are in the data?:") # Get the number of features
            if isinstance(number_of_features, int):
                data_matrix = st.data_editor(pd.DataFrame(columns={f"Feature {col+1}": [] for col in range(number_of_features)}), num_rows="dynamic")
                if not data_matrix.empty:
                    self.data = pd.DataFrame(data_matrix)
            else:
                st.info("Insert an integer")

    def compute_eigen(self):
        """Computes the eigenvalues and eigenvectors of the covariance matrix."""
        self.covariance_matrix = np.cov(self.data.T)
        self.eigenvalues, self.eigenvectors = np.linalg.eig(self.covariance_matrix)
    
    def transform_data(self):
        """Transforms the original data using the eigenvectors."""
        self.transformed_data = np.matmul(self.data, self.eigenvectors)
    
    def fit_transform(self):
        """Runs PCA and returns the transformed dataset."""
        self.compute_eigen()
        self.transform_data()
          
    def get_results(self):
        """Returns the PCA results including eigenvalues and transformed data."""
        return {
            "Covariance Matrix": self.covariance_matrix,
            "Eigenvalues": self.eigenvalues,
            "Eigenvectors": self.eigenvectors,
            "Transformed Data": self.transformed_data
        }
    
    def display_results(self):
        results = self.get_results()
        st.subheader("Covariance Matrix")
        st.write(results["Covariance Matrix"])
        st.subheader("Eigenvalues")
        st.write(results["Eigenvalues"])
        st.subheader("Eigenvectors")
        st.write(results["Eigenvectors"])
        st.subheader("Transformed Data")
        st.write(pd.DataFrame(results["Transformed Data"], columns=["PC1", "PC2"]))
    
    def plot_results(self):
        self.columns = self.data.columns.tolist()
        
        # Create a 2x2 grid layout
        fig = plt.subplots(1, figsize=(25, 25))
        # gs = fig.add_gridspec(2, 2)
        
        # Original Data Plot
        fig1, ax1 = plt.subplots(1, figsize=(25, 15))
        ax1.scatter(self.data[self.columns[0]], self.data[self.columns[1]])
        ax1.set_xlabel(self.columns[0], fontsize=25)
        ax1.set_ylabel(self.columns[1], fontsize=25)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        ax1.set_title("Original Data", fontsize=25)
        
        # Transformed Data Plot
        fig2, ax2 = plt.subplots(1, figsize=(15, 15))
        transformed_df = pd.DataFrame(self.transformed_data, columns=["PC1", "PC2"])
        ax2.scatter(transformed_df["PC1"], transformed_df["PC2"])
        ax2.set_xlabel("Principal Component 1", fontsize=18)
        ax2.set_ylabel("Principal Component 2", fontsize=18)
        ax2.set_title("Transformed Data (PCA)", fontsize=18)
        
        # Covariance Matrix Heatmap
        fig3, ax3 = plt.subplots(1, figsize=(15, 15))  # Spans entire bottom row
        sns.heatmap(
            self.covariance_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            xticklabels=self.columns,
            yticklabels=self.columns,
            ax=ax3
        )
        ax3.set_title("Covariance Matrix Heatmap", fontsize=25)
        plt.tight_layout()
        
        st.pyplot(fig1)
        st.pyplot(fig2)
        st.pyplot(fig3)
        

class PCACalculator:
    def __init__(self):
        self.data = None
        self.columns = None
        self.pca = None
        # data = st.data_editor(pd.DataFrame(columns=self.columns), num_rows="dynamic")
        # self.data = pd.DataFrame(data)

    def compute_eigen(self):
        self.pca.compute_eigen()
        st.session_state.eigen_done = True
    
    def transform_data(self):
        self.pca.transform_data()
        st.session_state.transform_done = True