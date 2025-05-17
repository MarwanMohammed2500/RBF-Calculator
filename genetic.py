import numpy as np
import pandas as pd
import streamlit as st
import time
import math
import plotly.express as px

class Genetic:
    """
    Genetic Algorithm implementation with variables a to h (integers 0-9).
    Fitness function: f(x) = (a + b) - (c + d) + (e + f) - (g + h)
    User-configurable population size, crossover points (2 or 3), and mutation rate.
    Selection: Top 40% of chromosomes by fitness (rounded up).
    Mutation: Random change of each variable with given probability.
    Termination: Fitness reaches 36 or max 1000 iterations.
    """
    def __init__(self):
        self.population_size = 10  # Default, set by user
        self.chromosome_length = 8
        self.crossover_points = None  # Set by user
        self.num_crossover_points = 2  # Default: 2-point crossover
        self.mutation_rate = 0.1  # Default mutation probability
        self.population = None
        self.fitness_values = None
        self.best_fitness = None
        self.best_chromosome = None
        self.iteration = 0
        self.max_iterations = 1000
        self.optimal_fitness = 36
        self.start_time = None
        self.end_time = None
        self.fitness_history = []

    def set_parameters(self, population_size, num_crossover_points, crossover_points, mutation_rate):
        """Set user-defined parameters with validation."""
        if population_size < 4:
            st.error("Population size must be at least 4.")
            return False
        if num_crossover_points not in [2, 3]:
            st.error("Number of crossover points must be 2 or 3.")
            return False
        if len(crossover_points) != num_crossover_points:
            st.error(f"Please provide exactly {num_crossover_points} crossover points.")
            return False
        if not all(1 <= cp < self.chromosome_length for cp in crossover_points):
            st.error(f"Crossover points must be between 1 and {self.chromosome_length - 1}.")
            return False
        if sorted(crossover_points) != list(crossover_points):
            st.error("Crossover points must be in ascending order.")
            return False
        if len(set(crossover_points)) != len(crossover_points):
            st.error("Crossover points must be unique.")
            return False

        self.population_size = population_size
        self.num_crossover_points = num_crossover_points
        self.crossover_points = crossover_points
        self.mutation_rate = mutation_rate
        return True

    def initialize_population(self, population_size, num_crossover_points, crossover_points, mutation_rate):
        """Initialize a population with user-specified size and parameters."""
        if not self.set_parameters(population_size, num_crossover_points, crossover_points, mutation_rate):
            return False
        self.population = np.random.randint(0, 10, size=(self.population_size, self.chromosome_length))
        self.fitness_values = np.array([self.fitness(chrom) for chrom in self.population])
        self.iteration = 0
        self.fitness_history = []
        self.best_fitness = np.max(self.fitness_values)
        self.best_chromosome = self.population[np.argmax(self.fitness_values)]
        self.start_time = None
        self.end_time = None
        st.session_state.ga_initialized = True
        return True

    def fitness(self, chromosome):
        """Calculate fitness: (a + b) - (c + d) + (e + f) - (g + h)"""
        a, b, c, d, e, f, g, h = chromosome
        return (a + b) - (c + d) + (e + f) - (g + h)

    def selection(self):
        """Select top 40% of chromosomes based on fitness (rounded up)."""
        num_parents = math.ceil(self.population_size * 0.4)
        if num_parents % 2 != 0:
            num_parents += 1  # Ensure even number for pairing
        indices = np.argsort(self.fitness_values)[-num_parents:]
        return self.population[indices]

    def crossover(self, parents):
        """Perform 2-point or 3-point crossover at user-specified points."""
        offspring = []
        for i in range(0, len(parents), 2):  # Pair parents
            if i + 1 >= len(parents):
                break
            parent1, parent2 = parents[i], parents[i + 1]
            child1, child2 = parent1.copy(), parent2.copy()
            if self.num_crossover_points == 2:
                # 2-point crossover: swap segment between cp1 and cp2
                cp1, cp2 = self.crossover_points
                child1[cp1:cp2], child2[cp1:cp2] = parent2[cp1:cp2], parent1[cp1:cp2]
            else:
                # 3-point crossover: swap two segments: between cp1-cp2 and after cp3
                cp1, cp2, cp3 = self.crossover_points
                child1[cp1:cp2], child2[cp1:cp2] = parent2[cp1:cp2], parent1[cp1:cp2]
                child1[cp3:], child2[cp3:] = parent2[cp3:], parent1[cp3:]
            offspring.extend([child1, child2])
        return np.array(offspring)

    def mutation(self, offspring):
        """Mutate each gene with given probability."""
        for child in offspring:
            for i in range(self.chromosome_length):
                if np.random.random() < self.mutation_rate:
                    child[i] = np.random.randint(0, 10)
        return offspring

    def run_iteration(self):
        """Run one iteration of the GA."""
        self.iteration += 1
        if self.iteration > self.max_iterations:
            return False

        # Selection
        parents = self.selection()

        # Crossover
        offspring = self.crossover(parents)

        # Mutation
        offspring = self.mutation(offspring)

        # Create new population
        # Keep parents, add offspring, fill rest with random chromosomes
        num_parents = len(parents)
        num_offspring = len(offspring)
        num_random = self.population_size - num_parents - num_offspring
        new_population = np.vstack((parents, offspring))
        if num_random > 0:
            random_chromosomes = np.random.randint(0, 10, size=(num_random, self.chromosome_length))
            new_population = np.vstack((new_population, random_chromosomes))

        # Update fitness values
        self.population = new_population
        self.fitness_values = np.array([self.fitness(chrom) for chrom in self.population])
        self.best_fitness = np.max(self.fitness_values)
        self.best_chromosome = self.population[np.argmax(self.fitness_values)]
        self.fitness_history.append(self.best_fitness)

        # Check for optimal solution
        if self.best_fitness >= self.optimal_fitness:
            self.end_time = time.time()
            return False
        return True

    def run(self):
        """Run the GA until optimal solution or max iterations."""
        if self.start_time is None:
            self.start_time = time.time()
        while self.iteration < self.max_iterations and self.best_fitness < self.optimal_fitness:
            if not self.run_iteration():
                break
        st.session_state.ga_done = True

    def display_population(self):
        """Display the current population and fitness values."""
        df = pd.DataFrame(
            self.population,
            columns=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        )
        df['Fitness'] = self.fitness_values
        st.write("Current Population:")
        st.dataframe(df)
        st.write(f"Iteration: {self.iteration}")
        st.write(f"Best Fitness: {self.best_fitness}")

    def plot_fitness(self):
        """Plot the best fitness over iterations using Plotly."""
        # Create a DataFrame for Plotly
        df = pd.DataFrame({
            "Iteration": range(len(self.fitness_history)),
            "Best Fitness": self.fitness_history
        })
        # Create line plot with Plotly Express
        fig = px.line(
            df,
            x="Iteration",
            y="Best Fitness",
            title="Best Fitness Over Iterations",
            labels={"Iteration": "Iteration", "Best Fitness": "Fitness"},
            markers=True
        )
        # Update layout for grid and legend
        fig.update_layout(
            showlegend=True,
            xaxis_title="Iteration",
            yaxis_title="Fitness",
            grid={"rows": 1, "columns": 1, "pattern": "independent"},
            plot_bgcolor="white",
            paper_bgcolor="white",
            xaxis=dict(showgrid=True, gridcolor="lightgray"),
            yaxis=dict(showgrid=True, gridcolor="lightgray")
        )
        # Render plot in Streamlit
        st.plotly_chart(fig, use_container_width=True)

    def get_results(self):
        """Display final results and time taken."""
        if self.end_time is not None:
            time_taken = self.end_time - self.start_time
            st.write(f"Optimal solution found in {self.iteration} iterations.")
            st.write(f"Time taken: {time_taken:.2f} seconds.")
            st.write(f"Best Chromosome: {self.best_chromosome.tolist()}")
            st.write(f"Best Fitness: {self.best_fitness}")
        else:
            st.write("No optimal solution found within max iterations.")