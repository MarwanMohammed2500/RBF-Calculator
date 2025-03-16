import numpy as np
import pandas as pd

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
