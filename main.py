import streamlit as st
from RBF import RBF  # RBF class created in RBF.py
from SOM import SOM  # SOM class created in SOM.py
from PCA import PCACalculator
from genetic import Genetic  # Genetic class created in Genetic.py

def main():
    st.title("Advanced Neural Networks")

    # Initialize classes in session state
    if "rbf" not in st.session_state:
        st.session_state.rbf = RBF()
    if "som" not in st.session_state:
        st.session_state.som = SOM()
    if "pca" not in st.session_state:
        st.session_state.pca = PCACalculator()
    if "ga" not in st.session_state:
        st.session_state.ga = Genetic()

    rbf = st.session_state.rbf
    som = st.session_state.som
    pca = st.session_state.pca
    ga = st.session_state.ga

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["RBF Calculator", "SOM Calculator", "PCA Calculator", "Genetic Algorithm Calculator"])

    # Tab 1: RBF Calculator
    with tab1:
        st.header("Radial Basis Function (RBF) Calculator")

        # Step 1: Get and Validate User Input
        st.subheader("Step 1: Enter Parameters")
        rbf.get_input()

        # Enable next step only when input is valid
        if rbf.c1 is not None and rbf.c2 is not None and rbf.sigma_sq is not None:
            st.session_state.step_1_done = True

        # Step 2: Calculate R (Enabled only if Step 1 is done)
        if st.session_state.get("step_1_done", False):
            st.subheader("Step 2: Compute R values")
            if st.button("Calculate R"):
                rbf.calc_r()
                st.session_state.step_2_done = True

        # Step 3: Calculate Phi (Enabled only if Step 2 is done)
        if st.session_state.get("step_2_done", False):
            st.subheader("Step 3: Compute Phi values")
            if st.button("Calculate Phi"):
                rbf.calc_phi()
                st.session_state.step_3_done = True

        # Step 4: Show Table (Enabled only if Step 3 is done)
        if st.session_state.get("step_3_done", False):
            st.subheader("Step 4: Show Table")
            rbf.show_table()

        # Step 5: Plot
        if st.session_state.get("step_3_done", False):
            st.subheader("Step 5: Plot")
            if st.button("Plot"):
                rbf.plot_x()
                rbf.plot_phi()

    # Tab 2: SOM Calculator
    with tab2:
        st.header("Self-Organizing Map (SOM) Calculator")

        # Step 1: Get and Validate User Input
        st.subheader("Step 1: Enter Parameters")
        som.get_input()

        # Enable next step only when input is valid
        if som.number_of_neurons is not None and som.number_of_clusters is not None and som.learning_rate is not None and som.choice is not None and som.decay is not None:
            st.session_state.step_1_done = True

        # Step 2: Calculate Distance (Enabled only if Step 1 is done)
        if st.session_state.get("step_1_done", False):
            st.subheader("Step 2: Compute Distance")
            if st.button("Calculate Distance"):
                som.calculate_distance()
                st.session_state.step_2_done = True
        
        # Step 3: Get the winner cluster for each neuron (Enabled only when step 2 is done)
        if st.session_state.get("step_2_done", False):
            st.subheader("Step 3: Get winner cluster for each neuron")
            som.find_winner()
            st.session_state.step_3_done = True

        # Step 4: Show Updated Cluster Weights (Enabled only if Step 3 is done)
        if st.session_state.get("step_3_done", False):
            st.subheader("Step 4: Show Updated Cluster Weights")
            som.show_new_weights()

    # Tab 3: PCA Calculator
    with tab3:
        st.header("Principal Component Analysis (PCA) Calculator")
        
        # Step 1: Input Data
        input_success = pca.get_input()
        
        if input_success:
            # Step 2: Compute Covariance
            st.subheader("Step 2: Compute Covariance Matrix")
            if st.button("Compute Covariance"):
                pca.compute_covariance()
            
            if st.session_state.get("covariance_done", False):
                # Step 3: Compute Eigenvalues/Vectors
                st.subheader("Step 3: Compute Eigenvalues & Eigenvectors")
                if st.button("Compute Eigen"):
                    pca.compute_eigen()
                
                if st.session_state.get("eigen_done", False):
                    # Step 4: Transform Data
                    st.subheader("Step 4: Transform Data")
                    if st.button("Transform Data"):
                        pca.transform_data()
                    
                    if st.session_state.get("transform_done", False):
                        # Step 5: Display Results
                        st.subheader("Step 5: Results")
                        pca.display_results()
                        pca.plot_results()

    # Tab 4: Genetic Algorithm Calculator
    with tab4:
        st.header("Genetic Algorithm Calculator")

        # Step 1: Initialize Population
        st.subheader("Step 1: Initialize Population")
        if st.button("Initialize Random Population"):
            ga.initialize_population()
            ga.display_population()

        # Step 2: Run Genetic Algorithm
        if st.session_state.get("ga_initialized", False):
            st.subheader("Step 2: Run Genetic Algorithm")
            if st.button("Run GA"):
                ga.run()
                ga.display_population()
                ga.get_results()
                ga.plot_fitness()

if __name__ == "__main__":
    main()