import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

class RBF:
    def __init__(self):
        __data = st.data_editor(pd.DataFrame({'category': [], 'x1': [], "x2":[]}), num_rows="dynamic") # Get data from the user
        self.df = pd.DataFrame(__data) # Transform the data into a pandas DataFrame
        self.c1 = st.text_input("Enter c1 (split values with a comma ','):") # get C1 from user
        self.c2 = st.text_input("Enter c2 (split values with a comma ','):") # get C2 from user
        self.c1 = np.array(list(map(float, self.c1.split(',')))) # Transform into a numpy array
        self.c2 = np.array(list(map(float, self.c2.split(','))))
        self.segma_sq = float(st.text_input("Enter 𝝈^2:")) # Get 𝝈^2 value from user

    def calc_r(self): # Calculate the values of R1 and R2
        r1_sq= (self.df.x1 - self.c1[0])**2 + (self.df.x2 - self.c1[1])**2
        r2_sq= (self.df.x1 - self.c2[0])**2 + (self.df.x2 - self.c2[1])**2
        self.df = self.df.join(pd.DataFrame({'r1 ^2': r1_sq, "r2 ^2": r2_sq}, index=self.df.index))
    
    def calc_phi(self): # Calculate the values of ∅1 and ∅2
        phi1= np.exp(-self.df["r1 ^2"] / (2 * self.segma_sq))
        phi2= np.exp(-self.df["r2 ^2"] / (2 * self.segma_sq))
        self.df = self.df.join(pd.DataFrame({'∅1': phi1, "∅2": phi2}, index=self.df.index))
    
    def plot_x(self): # Plot the X values
        fig, ax = plt.subplots()
        light_df = self.df[self.df["category"] == "1"]
        ax.scatter(light_df.x1, light_df.x2, linewidth=10,color='aquamarine', label="Light")

        dark_df = self.df[self.df["category"] == "0"]
        ax.scatter(dark_df.x1, dark_df.x2, linewidth=10, color='darkred', label="Dark")

        plt.style.use('ggplot')
        plt.xlabel("X1")
        plt.ylabel("X2")
        plt.legend()
        plt.grid(True)
        st.pyplot(fig)
    
    def plot_phi(self): # Plot the Phi values
        fig, ax = plt.subplots()
        light_df = self.df[self.df["category"] == "1"]
        ax.scatter(light_df["∅1"], light_df["∅2"], linewidth=10, color='aquamarine', label="Light")

        dark_df = self.df[self.df["category"] == "0"]
        ax.scatter(dark_df["∅1"], dark_df["∅2"], linewidth=10, color='darkred', label="Dark")

        plt.xlabel("∅1")
        plt.ylabel("∅2")
        plt.legend()
        plt.grid(True)
        st.pyplot(fig)
    
    def show_table(self): # Show the resultant table
        st.write("Full Table:")
        st.dataframe(self.df)
