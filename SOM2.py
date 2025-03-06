import numpy as np
import streamlit as st
import pandas as pd

class SOM:
    def get_input(self):
        self.number_of_neurons = st.text_input("How many neurons exist?:") # Get the number of neurons
        self.number_of_clusters = st.text_input("How many clusters exist?:") # Get the number of clusters
        self.learning_rate = st.text_input("What is the initial learning rate?:") # get the learning rate
        if self.number_of_neurons and self.number_of_clusters and self.learning_rate: # check input
            try:
                self.number_of_neurons = int(self.number_of_neurons) # Cast it into "int"
                self.number_of_clusters = int(self.number_of_clusters) # Cast it into "int"
                self.learning_rate = float(self.learning_rate) # Cast it into "float"
                self.initialize_vectors()
            except ValueError:
                st.warning("Invalid input detected. Please enter correct values.")
        else:
            st.info("Please provide input data to proceed.")
        
    def initialize_vectors(self):
        if hasattr(self, 'number_of_neurons') and hasattr(self, 'number_of_clusters'):
            self.clusters_vectors = np.random.rand(self.number_of_neurons, self.number_of_clusters)
            data = {col: [] for col in range(self.number_of_neurons)}
            training_vects = st.data_editor(pd.DataFrame(data), num_rows="dynamic")
            self.training_vectors = pd.DataFrame(training_vects)
        else:
            st.warning("Please provide valid input for number of neurons and clusters.")
    
    def calculate_distance(self):
        num_rows_of_training_vectors = self.training_vectors.shape[0]
        num_cols_of_clusters = self.clusters_vectors.shape[1]
        
        # Initialize a matrix to store distances for all training vectors
        all_distances = np.zeros((num_rows_of_training_vectors, num_cols_of_clusters))
        
        for i in range(num_rows_of_training_vectors):
            # Initialize a temporary array to store distances for the current training vector
            distances = np.zeros(num_cols_of_clusters)
            
            for j in range(num_cols_of_clusters):
                # Calculate the squared Euclidean distance between the training vector and cluster vector
                distances[j] = np.sum((self.training_vectors.iloc[i, :] - self.clusters_vectors[:, j]) ** 2)
            
            # Store distances for the current training vector
            all_distances[i, :] = distances
            
            # Find the index of the closest cluster (minimum distance)
            min_index = np.argmin(distances)
            
            # Update the weights of the closest cluster
            self.clusters_vectors[:, min_index] = self.update_weights(self.training_vectors.iloc[i, :], self.clusters_vectors[:, min_index])
        
        # Decay the learning rate
        self.learning_rate = self.learning_rate * 0.5
        
        # Save the updated cluster vectors in session state
        st.session_state.clusters_vectors = self.clusters_vectors
        # st.write("All Distances:")
        # st.write(all_distances)
        return all_distances
    
    def update_weights(self, training_vec, cluster_vec):
        # Update the cluster vector using the SOM weight update rule
        updated_weights = cluster_vec + self.learning_rate * (training_vec - cluster_vec)
        return updated_weights
    
    def show_new_weights(self):
        """
        This function shows the table (DataFrame) in the streamlit UI, it can be used any time throughout execution (edit in main.py), but for now, main.py is made to use this only after
        executing calc_r and calc_phi
        """
        self.clusters_vectors = st.session_state.get("clusters_vectors", self.clusters_vectors)
        st.write("Updated Cluster Weights:")
        st.dataframe(self.clusters_vectors)