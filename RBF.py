import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

class RBF:
    """
    This class uses a sequence of functions to calculate the Radial Basis Function (AKA RBF)
    get_input():
        Used to get the user input from the UI, input like centroid locations, features (X1 and X2), category of the sample, and the variance.
    calc_r():
        Used to calculate the squared distance between the points and each centroid.
    calc_phi():
        Used to calculate the transformed features ∅1 and ∅2.
    plot_x():
        Used to plot the original feature space.
    plot_phi():
        Used to plot the transformed feature space.
    """
    def get_input(self):
        """
        This function is responsible for getting user input from the interface and making sure the user inputs the right data types into their respective fields.
        Inputs:
        self.df --> the DataFrame representing our table, from which we will read, and to which we will write
        self.c1 --> The first centroid coordinates
        self.c2 --> The second centroid coordinates
        self.sigma_sq --> The variance
        """

        __data = st.data_editor(pd.DataFrame({'category': [], 'x1': [], "x2":[]}), num_rows="dynamic")
        self.df = pd.DataFrame(__data)
        self.c1 = st.text_input("Enter c1 (separate values using a comma):")
        self.c2 = st.text_input("Enter c2 (separate values using a comma):")
        self.sigma_sq = st.text_input("Enter 𝝈^2 (either enter a floating point or an integer, no fractions allowed):")
        if self.c1 and self.c2 and self.sigma_sq:
            try:
                self.c1 = np.array(list(map(float, self.c1.split(','))))
                self.c2 = np.array(list(map(float, self.c2.split(','))))
                self.sigma_sq = float(self.sigma_sq)
            except ValueError:
                st.warning("Invalid input detected. Please enter correct values.")
        else:
            st.info("Please provide input data to proceed.")
    
    def calc_r(self):
        """
        This function calculates the squared Euclidean distance between each point and each centroid, where R1 represents the vector of the distances between the points and
        the first centroid and R2 represent the vector of the distances between the points and the second centroid.

        r1_sq --> Squared Euclidean distance between the points and C1
        r2_sq --> Squared Euclidean distance between the points and C2
        """

        r1_sq= (self.df.x1 - self.c1[0])**2 + (self.df.x2 - self.c1[1])**2
        r2_sq= (self.df.x1 - self.c2[0])**2 + (self.df.x2 - self.c2[1])**2
        self.df = self.df.join(pd.DataFrame({'r1 ^2': r1_sq, "r2 ^2": r2_sq}, index=self.df.index)) # Joins the new calculated columns with the original DataFrame
        st.session_state.df = self.df # Save the DataFrame in the session state so that it can be updated and read in the next step
    
    def calc_phi(self):
        """
        This function calculates the transformed features, ∅1 and ∅2, via the equation:
        ∅1 = (e^(-r1^2/2*𝝈^2))
        ∅2 = (e^(-r2^2/2*𝝈^2))
        """
        
        self.df = st.session_state.get("df", self.df) # Reads the session state DataFrame, so it gets the updated value after calculating R1^2 and R2^2
        phi1= np.exp(-self.df["r1 ^2"] / (2 * self.sigma_sq))
        phi2= np.exp(-self.df["r2 ^2"] / (2 * self.sigma_sq))
        self.df = self.df.join(pd.DataFrame({'∅1': phi1, "∅2": phi2}, index=self.df.index)) 
        st.session_state.df = self.df
    
    def plot_x(self):
        """
        This function plots the original feature space
        """
        
        fig, ax = plt.subplots()
        self.df = st.session_state.get("df", self.df)
        light_df = self.df[self.df["category"] == 1] # Light Category
        ax.scatter(light_df.x1, light_df.x2, linewidth=10,color='aquamarine', label="Light")

        dark_df = self.df[self.df["category"] == 0] # Dark Category
        ax.scatter(dark_df.x1, dark_df.x2, linewidth=10, color='darkred', label="Dark")

        plt.style.use('ggplot')
        plt.xlabel("X1")
        plt.ylabel("X2")
        plt.title("Original feature space")
        plt.legend()
        plt.grid(True)
        st.pyplot(fig)
    
    def plot_phi(self):
        """
        This function plots the transformed feature space
        """
        
        fig, ax = plt.subplots()
        self.df = st.session_state.get("df", self.df)
        light_df = self.df[self.df["category"] == 1]
        ax.scatter(light_df["∅1"], light_df["∅2"], linewidth=10, color='aquamarine', label="Light")

        dark_df = self.df[self.df["category"] == 0]
        ax.scatter(dark_df["∅1"], dark_df["∅2"], linewidth=10, color='darkred', label="Dark")

        plt.xlabel("∅1")
        plt.ylabel("∅2")
        plt.title("Transformed feature space")
        plt.legend()
        plt.grid(True)
        st.pyplot(fig)
    
    def show_table(self):
        """
        This function shows the table (DataFrame) in the streamlit UI, it can be used any time throughout execution (edit in main.py), but for now, main.py is made to use this only after
        executing calc_r and calc_phi
        """
        self.df = st.session_state.get("df", self.df)
        st.write("Original + Transformation Table:")
        st.dataframe(self.df)
