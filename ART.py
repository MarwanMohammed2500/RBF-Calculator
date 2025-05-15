import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class ART1:
    """
    <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Adla3 el gandobly >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    Adaptive Resonance Theory 1 (ART1) implementation for clustering binary vectors.
    Parameters:
    - n: Number of components in input vector.
    - m: Maximum number of clusters.
    - rho: Vigilance parameter.
    - L: Parameter for weight updates.
    - vectors: List of binary input vectors to cluster.
    """
    def __init__(self, n=4, m=3, rho=0.4, L=2):
        self.n = n  # Number of components
        self.m = m  # Maximum number of clusters
        self.rho = rho  # Vigilance parameter
        self.L = L  # Weight update parameter
        self.b = np.full((n, m), 1 / (1 + n))  # Initial bottom-up weights (1/(1+n))
        self.t = np.ones((n, m))  # Initial top-down weights (1)
        self.b_initial = self.b.copy()  # Store initial bottom-up weights
        self.t_initial = self.t.copy()  # Store initial top-down weights
        self.clusters = {}  # Dictionary to store vector to cluster assignments
        self.active_nodes = set()  # Set of active F2 nodes
        self.vectors = None  # Store input vectors for plotting

    def norm(self, x):
        """Calculate the norm of vector x (sum of components)."""
        return np.sum(x)

    def activation(self, x):
        """Compute bottom-up activation for F2 nodes."""
        return np.dot(x, self.b)

    def vigilance_test(self, x, j):
        """Perform vigilance test for node j."""
        t_j = self.t[:, j]
        intersection = np.minimum(x, t_j)
        return self.norm(intersection) / self.norm(x) >= self.rho

    def update_weights(self, x, j):
        """Update weights for the winning node j."""
        norm_x = self.norm(x)
        self.b[:, j] = self.L * np.minimum(x, self.t[:, j]) / (0.5 * self.L + norm_x)
        self.t[:, j] = x

    def cluster_vector(self, x):
        """Cluster a single input vector x."""
        while True:
            # Compute activations
            activations = self.activation(x)
            if not np.any(activations > 0):
                # No active node, create new cluster
                if len(self.active_nodes) < self.m:
                    j = len(self.active_nodes)
                    self.active_nodes.add(j)
                    self.update_weights(x, j)
                    self.clusters[tuple(x)] = j
                    break
                else:
                    self.clusters[tuple(x)] = -1  # No cluster available
                    break

            # Find the winner (first node with highest activation)
            j = np.argmax(activations)
            if j not in self.active_nodes:
                self.active_nodes.add(j)

            # Vigilance test
            if self.vigilance_test(x, j):
                self.update_weights(x, j)
                self.clusters[tuple(x)] = j
                break
            else:
                # Inhibit winner and try next
                activations[j] = -1
                if not np.any(activations > 0):
                    if len(self.active_nodes) < self.m:
                        j = len(self.active_nodes)
                        self.active_nodes.add(j)
                        self.update_weights(x, j)
                        self.clusters[tuple(x)] = j
                        break
                    else:
                        self.clusters[tuple(x)] = -1
                        break

    def fit(self, vectors):
        """Cluster all input vectors."""
        self.vectors = vectors  # Store vectors for plotting
        for x in vectors:
            self.cluster_vector(np.array(x))
        st.session_state.art1_done = True

    def plot_clusters(self):
        """Plot clusters using the first two components (note: 4D reduced to 2D)."""
        if not self.vectors or not self.clusters:
            st.write("No data to plot. Run clustering first.")
            return

        # Prepare data for plotting (using first two components)
        x_data = [v[0] for v in self.vectors]
        y_data = [v[1] for v in self.vectors]
        cluster_labels = [self.clusters[tuple(v)] for v in self.vectors]

        fig, ax = plt.subplots()
        scatter = ax.scatter(x_data, y_data, c=cluster_labels, cmap='viridis')
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.set_title("Cluster Visualization (First 2 Components)")
        plt.colorbar(scatter, label="Cluster")
        st.pyplot(fig)
        st.write("Note: Visualization uses first two components of 4D vectors. Full clustering is based on all components.")

    def plot_heatmaps(self):
        """Plot heatmaps for b_ij and t_ji before and after clustering."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        sns.heatmap(self.b_initial, annot=True, fmt=".2f", cmap="YlOrRd", ax=axes[0, 0])
        axes[0, 0].set_title("Initial b_ij")
        axes[0, 0].set_xlabel("F2 Nodes")
        axes[0, 0].set_ylabel("F1 Nodes")

        sns.heatmap(self.t_initial, annot=True, fmt=".2f", cmap="YlOrRd", ax=axes[0, 1])
        axes[0, 1].set_title("Initial t_ji")
        axes[0, 1].set_xlabel("F2 Nodes")
        axes[0, 1].set_ylabel("F1 Nodes")

        sns.heatmap(self.b, annot=True, fmt=".2f", cmap="YlOrRd", ax=axes[1, 0])
        axes[1, 0].set_title("Final b_ij")
        axes[1, 0].set_xlabel("F2 Nodes")
        axes[1, 0].set_ylabel("F1 Nodes")

        sns.heatmap(self.t, annot=True, fmt=".2f", cmap="YlOrRd", ax=axes[1, 1])
        axes[1, 1].set_title("Final t_ji")
        axes[1, 1].set_xlabel("F2 Nodes")
        axes[1, 1].set_ylabel("F1 Nodes")

        plt.tight_layout()
        st.pyplot(fig)

    def display_results(self):
        """Display clustering results and weights."""
        st.subheader("Clustering Results")
        df = pd.DataFrame(
            [(list(k), v) for k, v in self.clusters.items()],
            columns=["Vector", "Cluster"]
        )
        st.dataframe(df)

        st.subheader("Bottom-Up Weights (b_ij)")
        st.write(pd.DataFrame(self.b, index=[f"X{i+1}" for i in range(self.n)], columns=[f"Y{j+1}" for j in range(self.m)]))

        st.subheader("Top-Down Weights (t_ji)")
        st.write(pd.DataFrame(self.t, index=[f"X{i+1}" for i in range(self.n)], columns=[f"Y{j+1}" for j in range(self.m)]))