import numpy as np
import streamlit as st
import pandas as pd

class SOM:
    """
    This class uses a sequence of functions to calculate the Self-Organizing Map (AKA SOM)
    get_input():
        Used to get the user input from the UI, input like the number of neurons (inputs), number of clusters, learning rate.
    initialize_vectors():
        Used to initialize the cluster vectors and training vectors.
    calculate_distance():
        Used to calculate squared euclidean distance between each training vector and each cluster.
    update_weights():
        Used to update the weight given the equation W_new = W_old + learning_rate*(x - W_old).
    show_new_weights():
        Used to show the table of updated weights.
    """
    
    def get_input(self):
        """
        This function is responsible for getting user input from the interface and making sure the user inputs the right data types into their respective fields.
        Inputs:
        self.number_of_neurons --> The number of neurons (inputs) in the network
        self.number_of_clusters --> The number of clusters
        self.learning_rate --> The learning rate
        """
        self.number_of_neurons = st.text_input("How many neurons in the network?:") # Get the number of neurons
        self.number_of_clusters = st.text_input("How many clusters exist?:") # Get the number of clusters
        self.learning_rate = st.text_input("Set the initial learning rate:") # get the learning rate
        self.decay = st.text_input("What is the rate of decay of the learning rate?:") # get the rate of decay of the learning rate
        self.choice = st.selectbox("Would you like to randomly initialize the weights or have your own input?",
                              ("Random Initialization", "I have my own weights"))
        if self.number_of_neurons and self.number_of_clusters and self.learning_rate and self.choice and self.decay: # check input
            try:
                self.number_of_neurons = int(self.number_of_neurons) # Cast into "int"
                self.number_of_clusters = int(self.number_of_clusters) # Cast into "int"
                self.learning_rate = float(self.learning_rate) # Cast into "float"
                self.decay = float(self.decay) # Cast into "float"
                if not 0 < self.decay < 1:
                    st.warning("the rate of decay must be between 0 and 1")
                if self.choice == "Random Initialization":
                    self.initialize_vectors_random() # Call initialize_vectors_random after making sure all inputs are valid.
                else:
                    self.initialize_vectors_input() # Call initialize_vectors_input after making sure all inputs are valid.
            except ValueError:
                st.warning("Invalid input. Please insert valid values.")
        else:
            st.info("Please provide input data to proceed.")
        
    def initialize_vectors_random(self):
        """
        This function is responsible for initializing clusters vectors and training vectors randomly:
        self.clusters_vectors --> cluster weights matrix (randomly initialized)
        self.number_of_clusters --> The number of clusters
        self.training_vectors --> The training vectors
        """
        if hasattr(self, 'number_of_neurons') and hasattr(self, 'number_of_clusters'):
            self.clusters_vectors = np.random.rand(self.number_of_neurons, self.number_of_clusters) # Randomly Initialize the cluster weights
            data = {f"Vector Component #{col+1}": [] for col in range(self.number_of_neurons)} # Get the number of columns (number of neurons) for the next step
            st.write("Input training vectors (each row is a training vector):")
            training_vects_rndm = st.data_editor(pd.DataFrame(data), num_rows="dynamic") # Gets input from the user
            self.training_vectors = pd.DataFrame(training_vects_rndm) # Turns it into a DataFrame
        else:
            st.warning("Please provide valid input for number of neurons and clusters.")

    def initialize_vectors_input(self):
        """
        This function is responsible for initializing clusters vectors and training vectors:
        self.clusters_vectors --> cluster weights matrix (user input)
        self.number_of_clusters --> The number of clusters
        self.training_vectors --> The training vectors
        """
        if hasattr(self, 'number_of_neurons') and hasattr(self, 'number_of_clusters'):
            col = {f"Cluster #{col+1} weights": [] for col in range(self.number_of_clusters)} # Get the number of columns (number of clusters) for the next step
            st.write("Input cluster weights (each column is a weight vector):")
            clusters_vects = st.data_editor(pd.DataFrame(col), num_rows="dynamic") # Get user input to initialize the cluster weights
            self.clusters_vectors = np.array(clusters_vects) # Turns it into a NumPy array
            data = {f"Vector Component #{col+1}": [] for col in range(self.number_of_neurons)}
            st.write("Input training vectors (each row is a training vector):")
            training_vects_in = st.data_editor(pd.DataFrame(data), num_rows="dynamic") # Gets input from the user
            self.training_vectors = pd.DataFrame(training_vects_in) # Turns it into a DataFrame
        else:
            st.warning("Please provide valid input for number of neurons and clusters.")
    
    def calculate_distance(self):
        """
        This function calculates the squared Euclidean distance between each point and each cluster,.

        num_rows_of_training_vectors --> Number of rows in the training matrix
        num_cols_of_clusters --> Number of columns in the cluster matrix
        all_distances --> Saves all distances calculated per each iteration
        distances --> Temporary storage for the iteration's distances
        min_index --> Gets the index of the minimum value
        """
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
        self.learning_rate = self.learning_rate * self.decay
        
        # Save the updated cluster vectors in session state
        st.session_state.clusters_vectors = self.clusters_vectors
                
        # st.write("All Distances:")
        # st.write(all_distances)
        return all_distances
    
    def update_weights(self, training_vec, cluster_vec):
        """
        This function is responsible for updating the weights
        Inputs:
            cluster_vec --> Cluster vector we're updating
            training_vec --> Training vector we're iterating through
        Output:
            updated_weights --> the updated weights
        """
        # Update the cluster vector using the SOM weight update rule
        updated_weights = cluster_vec + self.learning_rate * (training_vec - cluster_vec)
        return updated_weights
    
    def find_winner(self):
        num_rows_of_training_vectors = self.training_vectors.shape[0]
        num_cols_of_clusters = self.clusters_vectors.shape[1]
        for i in range(num_rows_of_training_vectors):
            # Initialize a temporary array to store distances for the current training vector
            distances = np.zeros(num_cols_of_clusters)
            
            for j in range(num_cols_of_clusters):
                # Calculate the squared Euclidean distance between the training vector and cluster vector
                distances[j] = np.sum((self.training_vectors.iloc[i, :] - self.clusters_vectors[:, j]) ** 2)
            
            min_index = np.argmin(distances)
            st.write(f"Neuron {i+1} belongs to cluster {min_index}")
    
    def show_new_weights(self):
        """
        This function shows the table (DataFrame) in the streamlit UI.
        """
        self.clusters_vectors = st.session_state.get("clusters_vectors", self.clusters_vectors)
        st.write("Updated Cluster Weights:")
        st.dataframe(self.clusters_vectors)
