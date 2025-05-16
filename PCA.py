# PCA_GUI.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class PCA:
    def __init__(self):
        self.data = None
    
    def get_input(self):
        choice = st.selectbox("Do you have a dataset file (csv) or will you manually insert data?",
                              ("Select", "I have a CSV file", "I'll manually insert my data"))

        # If the user has a CSV File they want to upload
        if choice == "I have a CSV file":
            uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
            if uploaded_file is not None:
                self.data = pd.read_csv(uploaded_file)
                data_matrix = st.data_editor(self.data, num_rows="dynamic")
        # If the user will manually insert the data
        elif choice == "I'll manually insert my data":
            number_of_features = st.text_input("How many features are in the data?:") # Get the number of features
            if isinstance(number_of_features, int):
                data_matrix = st.data_editor(pd.DataFrame(columns={f"Feature {col+1}": [] for col in range(number_of_features)}), num_rows="dynamic")
                if not data_matrix.empty:
                    self.data = pd.DataFrame(data_matrix)
            else:
                st.warning("Insert an integer")
        
        
    
    def centralize(self):
        self.centered_data = self.data - self.data.mean()
    
    def covariance(self):
        self.covariance_matrix = np.cov(self.centered_data.T)
    
    def compute_eigens(self):
        self.eigenvalues, self.eigenvectors = np.linalg.eig(self.covariance_matrix)
        idx = np.argsort(self.eigenvalues)
        self.eigenvalues = self.eigenvalues[idx]
        self.eigenvectors = self.eigenvectors[:,idx]
        
        
    def project_data(self):
        self.transformed_data = np.dot(self.eigenvectors, self.data.T).T
        
    def display_results(self):
        st.subheader("Centred Data")
        st.write(self.centered_data)
        st.subheader("Covariance Matrix")
        st.write(self.covariance_matrix)
        st.subheader("Eigenvalues")
        st.write(self.eigenvalues)
        st.subheader("Eigenvectors")
        st.write(self.eigenvectors)
        st.subheader("Transformed Data")
        st.write(self.transformed_data)
    
    def plot_results(self):
        # Original Data Plot
        fig1, (ax1, ax2) = plt.subplots(2, figsize=(25, 25))
        ax1.scatter(self.data.iloc[:, 0], self.data.iloc[:, 1])
        ax1.set_xlabel(self.data.columns[0], fontsize=22)
        ax1.set_ylabel(self.data.columns[1], fontsize=22)
        ax1.tick_params(axis='both', which='major', labelsize=22)
        ax1.set_title("Original Data", fontsize=25)
        
        # Transformed Data Plot
        # fig2, ax2 = plt.subplots(1, figsize=(15, 15))
        transformed_df = pd.DataFrame(self.transformed_data, columns={f"PC{col+1}": [] for col in range(len(self.transformed_data.T))})
        ax2.scatter(transformed_df["PC1"], transformed_df["PC2"])
        ax2.set_xlabel("Principal Component 1", fontsize=22)
        ax2.set_ylabel("Principal Component 2", fontsize=22)
        ax2.set_title("Transformed Data (PCA)", fontsize=22)
        ax2.tick_params(axis='both', which='major', labelsize=22)

        
        # Covariance Matrix Heatmap
        fig2, ax3 = plt.subplots(1, figsize=(15, 15))  # Spans entire bottom row
        hm = sns.heatmap(
            self.covariance_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            xticklabels=self.data.columns,
            yticklabels=self.data.columns,
            ax=ax3,
            annot_kws={"fontsize":18}
        )
        ax3.set_title("Covariance Matrix Heatmap", fontsize=25)
        ax3.tick_params(axis='both', which='major', labelsize=19)
        plt.tight_layout()
        
        
        st.pyplot(fig1)
        st.pyplot(fig2)