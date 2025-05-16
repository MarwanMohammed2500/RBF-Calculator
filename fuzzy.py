import streamlit as st
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

class Fuzzy:

    def get_input(self):
        # Set up the page
        st.title("Fuzzy Smart AC Controller")
        st.write("Adjust the temperature and humidity to get the recommended AC power level.")

        # Create sliders for user input
        temperature_input = st.slider("Temperature (°C)", min_value=0, max_value=40, value=25)
        humidity_input = st.slider("Humidity (%)", min_value=0, max_value=100, value=50)
        
        # Define fuzzy variables
        temperature = ctrl.Antecedent(np.arange(0, 41, 1), 'temperature')
        humidity = ctrl.Antecedent(np.arange(0, 101, 1), 'humidity')
        ac_power = ctrl.Consequent(np.arange(0, 101, 1), 'ac_power')

        # Define membership functions for temperature
        temperature['cold'] = fuzz.trimf(temperature.universe, [0, 0, 20])
        temperature['comfortable'] = fuzz.trimf(temperature.universe, [15, 25, 30])
        temperature['hot'] = fuzz.trimf(temperature.universe, [25, 40, 40])

        # Define membership functions for humidity
        humidity['dry'] = fuzz.trimf(humidity.universe, [0, 0, 40])
        humidity['normal'] = fuzz.trimf(humidity.universe, [30, 50, 70])
        humidity['humid'] = fuzz.trimf(humidity.universe, [60, 100, 100])

        # Define membership functions for AC power
        ac_power['low'] = fuzz.trimf(ac_power.universe, [0, 0, 40])
        ac_power['medium'] = fuzz.trimf(ac_power.universe, [30, 50, 70])
        ac_power['high'] = fuzz.trimf(ac_power.universe, [60, 100, 100])

        # Define fuzzy rules
        rule1 = ctrl.Rule(temperature['cold'] & humidity['dry'], ac_power['low'])
        rule2 = ctrl.Rule(temperature['cold'] & humidity['normal'], ac_power['low'])
        rule3 = ctrl.Rule(temperature['cold'] & humidity['humid'], ac_power['medium'])
        rule4 = ctrl.Rule(temperature['comfortable'] & humidity['dry'], ac_power['low'])
        rule5 = ctrl.Rule(temperature['comfortable'] & humidity['normal'], ac_power['medium'])
        rule6 = ctrl.Rule(temperature['comfortable'] & humidity['humid'], ac_power['high'])
        rule7 = ctrl.Rule(temperature['hot'] & humidity['dry'], ac_power['medium'])
        rule8 = ctrl.Rule(temperature['hot'] & humidity['normal'], ac_power['high'])
        rule9 = ctrl.Rule(temperature['hot'] & humidity['humid'], ac_power['high'])

        # Create control system and simulation
        ac_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
        ac_simulation = ctrl.ControlSystemSimulation(ac_ctrl)

        # Pass user inputs to the simulation
        ac_simulation.input['temperature'] = temperature_input
        ac_simulation.input['humidity'] = humidity_input

        # Compute the result
        ac_simulation.compute()

        # Display the result
        st.subheader("Recommended AC Power Level:")
        st.write(f"{ac_simulation.output['ac_power']:.2f}%")

        # Optional: Visualize the output
        ac_power.view(sim=ac_simulation)