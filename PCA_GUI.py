# PCA_GUI.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class PCA:
    def __init__(self, data, columns):
        """
        Initialize PCA with a DataFrame and the selected columns.
        
        Parameters:
        data (pd.DataFrame): The input dataset.
        columns (list): The two columns on which PCA will be applied.
        """
        self.data = data[columns]
        self.columns = columns
        self.mean_vector = self.data.mean()
        self.covariance_matrix = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.transformed_data = None
    
    def compute_covariance(self):
        """Computes the covariance matrix of the selected columns."""
        def covariance(feature1, feature2):
            n = len(feature1)
            mean_x = np.mean(feature1)
            mean_y = np.mean(feature2)
            return sum((feature1[i] - mean_x) * (feature2[i] - mean_y) for i in range(n)) / n
        
        var_x1 = self.data[self.columns[0]].var(ddof=0)
        var_x2 = self.data[self.columns[1]].var(ddof=0)
        cov_x1_x2 = covariance(self.data[self.columns[0]], self.data[self.columns[1]])
        
        self.covariance_matrix = np.array([
            [var_x1, cov_x1_x2],
            [cov_x1_x2, var_x2]
        ])
    
    def compute_eigen(self):
        """Computes the eigenvalues and eigenvectors of the covariance matrix."""
        self.eigenvalues, self.eigenvectors = np.linalg.eig(self.covariance_matrix)
    
    def transform_data(self):
        """Transforms the original data using the eigenvectors."""
        realExample_numpy = np.array(self.data)
        self.transformed_data = np.matmul(realExample_numpy, self.eigenvectors)
    
    def fit_transform(self):
        """Runs PCA and returns the transformed dataset."""
        self.compute_covariance()
        self.compute_eigen()
        self.transform_data()
        return self.transformed_data
    
    def get_results(self):
        """Returns the PCA results including eigenvalues and transformed data."""
        return {
            "Covariance Matrix": self.covariance_matrix,
            "Eigenvalues": self.eigenvalues,
            "Eigenvectors": self.eigenvectors,
            "Transformed Data": self.transformed_data
        }

# Example Usage
data = {
    "X1": [1.4, 1.6, -1.4, -2, -3, 2.4, 1.5, 2.3, -3.2, -4.1],
    "X2": [1.65, 1.975, -1.775, -2.525, -3.95, 3.075, 2.025, 2.75, -4.05, -4.85]
}

realExample = pd.DataFrame(data)

# Apply PCA
pca = PCA(realExample, ["X1", "X2"])
final_result = pca.fit_transform()

# Display results
print("Final Transformed Data:")
print(final_result)


class PCACalculator:
    def __init__(self):
        self.data = None
        self.columns = None
        self.pca = None
        self.example_data = pd.DataFrame({
            "X1": [1.4, 1.6, -1.4, -2, -3, 2.4, 1.5, 2.3, -3.2, -4.1],
            "X2": [1.65, 1.975, -1.775, -2.525, -3.95, 3.075, 2.025, 2.75, -4.05, -4.85]
        })
    
    def get_input(self):
        st.subheader("Step 1: Input Data")
        use_example = st.checkbox("Use example data", value=True)
        if use_example:
            self.data = self.example_data
            st.write("Example Data:")
            st.dataframe(self.data)
        else:
            uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
            if uploaded_file is not None:
                self.data = pd.read_csv(uploaded_file)
            else:
                input_data = st.data_editor(pd.DataFrame(columns=["X1", "X2"]), num_rows="dynamic")
                if not input_data.empty:
                    self.data = input_data
        
        if self.data is not None:
            available_columns = self.data.columns.tolist()
            self.columns = st.multiselect("Select two columns for PCA", options=available_columns, default=available_columns[:2] if len(available_columns) >=2 else [])
            if len(self.columns) != 2:
                st.warning("Please select exactly two columns.")
                return False
            return True
        return False
    
    def compute_covariance(self):
        self.pca = PCA(self.data, self.columns)
        self.pca.compute_covariance()
        st.session_state.covariance_done = True
    
    def compute_eigen(self):
        self.pca.compute_eigen()
        st.session_state.eigen_done = True
    
    def transform_data(self):
        self.pca.transform_data()
        st.session_state.transform_done = True
    
    def display_results(self):
        results = self.pca.get_results()
        st.subheader("Covariance Matrix")
        st.write(results["Covariance Matrix"])
        st.subheader("Eigenvalues")
        st.write(results["Eigenvalues"])
        st.subheader("Eigenvectors")
        st.write(results["Eigenvectors"])
        st.subheader("Transformed Data")
        st.write(pd.DataFrame(results["Transformed Data"], columns=["PC1", "PC2"]))
    
    def plot_results(self):
    
        # Create a 2x2 grid layout
        fig = plt.figure(figsize=(15, 12))
        gs = fig.add_gridspec(2, 2)
        
        # Original Data Plot
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.scatter(self.data[self.columns[0]], self.data[self.columns[1]])
        ax1.set_xlabel(self.columns[0])
        ax1.set_ylabel(self.columns[1])
        ax1.set_title("Original Data")
        
        # Transformed Data Plot
        ax2 = fig.add_subplot(gs[0, 1])
        transformed_df = pd.DataFrame(self.pca.transformed_data, columns=["PC1", "PC2"])
        ax2.scatter(transformed_df["PC1"], transformed_df["PC2"])
        ax2.set_xlabel("Principal Component 1")
        ax2.set_ylabel("Principal Component 2")
        ax2.set_title("Transformed Data (PCA)")
        
        # Covariance Matrix Heatmap
        ax3 = fig.add_subplot(gs[1, :])  # Spans entire bottom row
        sns.heatmap(
            self.pca.covariance_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            xticklabels=self.columns,
            yticklabels=self.columns,
            ax=ax3
        )
        ax3.set_title("Covariance Matrix Heatmap")
        plt.tight_layout()
        
        st.pyplot(fig)
