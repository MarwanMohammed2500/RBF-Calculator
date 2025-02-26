import streamlit as st
from RBF import RBF # RBF class created in RBF.py

def main():
    st.title("Radial Basis Function (RBF) Calculator")

    # Initialize RBF in session state
    if "rbf" not in st.session_state:
        st.session_state.rbf = RBF()

    rbf = st.session_state.rbf  # Access stored RBF object

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

if __name__ == "__main__":
    main()
