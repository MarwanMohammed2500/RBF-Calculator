import streamlit as st
from RBF import RBF
from SOM import SOM
from PCA import PCA
from genetic import Genetic
from ART import ART1
import numpy as np
import pandas as pd
def main():
    st.title("Advanced Neural Networks")

    # Initialize classes in session state
    if "rbf" not in st.session_state:
        st.session_state.rbf = RBF()
    if "som" not in st.session_state:
        st.session_state.som = SOM()
    if "pca" not in st.session_state:
        st.session_state.pca = PCA()
    if "ga" not in st.session_state:
        st.session_state.ga = Genetic()
    if "art1" not in st.session_state:
        st.session_state.art1 = ART1()

    rbf = st.session_state.rbf
    som = st.session_state.som
    pca = st.session_state.pca
    ga = st.session_state.ga
    art1 = st.session_state.art1

    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["RBF Calculator", "SOM Calculator", "PCA Calculator", "Genetic Algorithm Calculator", "ART1 Calculator"])

    # Tab 1: RBF Calculator
    with tab1:
        st.header("Radial Basis Function (RBF) Calculator")

        # Step 1: Get and Validate User Input
        st.subheader("Step 1: Enter Parameters")
        rbf.get_input()

        # Enable next step only when input is valid
        if rbf.c1 is not None and rbf.c2 is not None and rbf.sigma_sq is not None:
            st.session_state.rbf_step_1_done = True

        # Step 2: Calculate R (Enabled only if Step 1 is done)
        if st.session_state.get("rbf_step_1_done", False):
            st.subheader("Step 2: Compute R values")
            if st.button("Calculate R"):
                rbf.calc_r()
                st.session_state.rbf_step_2_done = True

        # Step 3: Calculate Phi (Enabled only if Step 2 is done)
        if st.session_state.get("rbf_step_2_done", False):
            st.subheader("Step 3: Compute Phi values")
            if st.button("Calculate Phi"):
                rbf.calc_phi()
                st.session_state.rbf_step_3_done = True

        # Step 4: Show Table (Enabled only if Step 3 is done)
        if st.session_state.get("rbf_step_3_done", False):
            st.subheader("Step 4: Show Table")
            rbf.show_table()

        # Step 5: Plot
        if st.session_state.get("rbf_step_3_done", False):
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
            st.session_state.som_step_1_done = True

        # Step 2: Calculate Distance (Enabled only if Step 1 is done)
        if st.session_state.get("som_step_1_done", False):
            st.subheader("Step 2: Compute Distance")
            if st.button("Calculate Distance"):
                som.calculate_distance()
                st.session_state.som_step_2_done = True
        
        # Step 3: Get the winner cluster for each neuron (Enabled only when step 2 is done)
        if st.session_state.get("som_step_2_done", False):
            st.subheader("Step 3: Get winner cluster for each neuron")
            som.find_winner()
            st.session_state.som_step_3_done = True

        # Step 4: Show Updated Cluster Weights (Enabled only if Step 3 is done)
        if st.session_state.get("som_step_3_done", False):
            st.subheader("Step 4: Show Updated Cluster Weights")
            som.show_new_weights()

    # Tab 3: PCA Calculator
    with tab3:
        st.header("Principal Component Analysis (PCA) Calculator")
        
        # Step 1: Input Data
        st.subheader("Insert Data")
        pca.get_input()
        #Ensure that data was set correctly to initialize next step
        if pca.data is not None:
            st.session_state.pca_step_1_done = True
        
        # Step 2: Centralize the data
        if st.session_state.get("pca_step_1_done", False):
            st.subheader("Step 2: Centralize the Data")
            if st.button("Centralize Data"):
                pca.centralize()
                st.session_state.pca_step_2_done = True
        
        # Step 3: Compute covariance
        if st.session_state.get("pca_step_2_done", False):
            st.subheader("Step 3: Compute covariance")
            if st.button("Compute Covariance"):
                pca.covariance()
                st.session_state.pca_step_3_done = True
             
        # Step 4: Compute the Eignes
        if st.session_state.get("pca_step_3_done", False):    
            st.subheader("Step 3: Compute the Eigens")
            if st.button("Compute Eigens"):
                pca.compute_eigens()
                st.session_state.pca_step_4_done = True
        
        # Step 5: Project Data into the principle component:
        if st.session_state.get("pca_step_4_done", False):
            st.subheader("Project Data into the Principle Component")
            if st.button("Project Data"):
                pca.project_data()
                st.session_state.pca_step_5_done = True
        
        # Final step: Display and plot
        if st.session_state.get("pca_step_5_done", False):
            st.subheader("Final Step: Display and Plot")
            if st.button("Display and Plot"):
                pca.display_results()
                pca.plot_results()
                 
        # Step 4: Results
        # if st.session_state.get("pca_step_4_done", False):
        #     st.subheader("Step 4: Results")
        #     pca.display_results()
        #     pca.plot_results()

    # Tab 4: Genetic Algorithm Calculator
    # with tab4:
    #     st.header("Genetic Algorithm Calculator")

    #     # Step 1: Initialize Population
    #     st.subheader("Step 1: Initialize Population")
    #     if st.button("Initialize Random Population"):
    #         ga.initialize_population()
    #         ga.display_population()
    #         st.session_state.gen_step_1_done = True

    #     # Step 2: Run Genetic Algorithm
    #     if st.session_state.get("gen_step_1_done", False):
    #         st.subheader("Step 2: Run Genetic Algorithm")
    #         if st.button("Run GA"):
    #             ga.run()
    #             ga.display_population()
    #             ga.get_results()
    #             ga.plot_fitness()

    # # Tab 5: ART1 Calculator
    # with tab5:
    #     st.header("ART1 Calculator")

    #     # Step 1: Select Example or Custom Input
    #     use_example = st.radio("Use Example Vectors?", ("Yes", "No"), index=0)
    #     if use_example == "Yes":
    #         n = 4
    #         m = 3
    #         rho = 0.4
    #         L = 2
    #         vectors = [(1,1,0,0), (0,0,0,1), (1,0,0,0), (0,0,1,1)]
    #         st.write("Example Vectors: ", vectors)
    #     else:
    #         st.subheader("Custom Input")
    #         n = st.number_input("Number of components (n)", min_value=1, value=4)
    #         m = st.number_input("Max number of clusters (m)", min_value=1, value=3)
    #         rho = st.number_input("Vigilance parameter (rho)", min_value=0.0, max_value=1.0, value=0.4)
    #         L = st.number_input("L parameter", min_value=1.1, value=2.0)
    #         vector_data = st.data_editor(
    #             pd.DataFrame(columns=[f"X{i+1}" for i in range(n)]),
    #             num_rows="dynamic"
    #         )
    #         vectors = [tuple(row.dropna().astype(int).tolist()) for _, row in vector_data.iterrows() if row.notna().any()]
    #         st.write("Entered Vectors:", vectors)

    #     # Initialize ART1 with user parameters
    #     art1.n = n
    #     art1.m = m
    #     art1.rho = rho
    #     art1.L = L
    #     art1.b = np.full((n, m), 1 / (1 + n))
    #     art1.t = np.ones((n, m))
    #     art1.b_initial = art1.b.copy()
    #     art1.t_initial = art1.t.copy()
    #     art1.clusters = {}
    #     art1.active_nodes = set()
        

    #     # Step 2: Run ART1 Clustering
    #     if st.button("Run ART1") and vectors:
    #         art1.fit(vectors)
    #         if st.session_state.get("art1_done", False):
    #             art1.display_results()
    #             art1.plot_clusters()
    #             art1.plot_heatmaps()

if __name__ == "__main__":
    main()