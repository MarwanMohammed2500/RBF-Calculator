import numpy as np
import pandas as pd
import streamlit as st
import time

class Genetic:
    """
    Genetic Algorithm implementation with variables a to h (integers 0-9).
    Fitness function: f(x) = (a + b) - (c + d) + (e + f) - (g + h)
    Population size: 10 chromosomes
    Selection: Top 4 chromosomes by fitness
    Crossover: Split at middle (after variable d)
    Mutation: Random change of one variable per offspring
    Termination: Fitness reaches 36 or max 1000 iterations
    """
    def __init__(self):
        self.population_size = 10
        self.chromosome_length = 8
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

    def initialize_population(self):
        """Initialize a population of 10 random chromosomes."""
        self.population = np.random.randint(0, 10, size=(self.population_size, self.chromosome_length))
        self.fitness_values = np.array([self.fitness(chrom) for chrom in self.population])
        self.iteration = 0
        self.fitness_history = []
        self.best_fitness = np.max(self.fitness_values)
        self.best_chromosome = self.population[np.argmax(self.fitness_values)]
        self.start_time = None
        self.end_time = None
        st.session_state.ga_initialized = True

    def fitness(self, chromosome):
        """Calculate fitness: (a + b) - (c + d) + (e + f) - (g + h)"""
        a, b, c, d, e, f, g, h = chromosome
        return (a + b) - (c + d) + (e + f) - (g + h)

    def selection(self):
        """Select top 4 chromosomes based on fitness."""
        indices = np.argsort(self.fitness_values)[-4:]  # Get indices of top 4
        return self.population[indices]

    def crossover(self, parents):
        """Perform crossover at the middle (after variable d)."""
        offspring = []
        for i in range(0, 4, 2):  # Pair parents (0-1, 2-3)
            parent1, parent2 = parents[i], parents[i + 1]
            crossover_point = self.chromosome_length // 2  # Split after d (index 4)
            child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
            child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
            offspring.extend([child1, child2])
        return np.array(offspring)

    def mutation(self, offspring):
        """Mutate one random variable in each offspring."""
        for child in offspring:
            mutation_point = np.random.randint(0, self.chromosome_length)
            child[mutation_point] = np.random.randint(0, 10)
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
        # Keep 4 parents, add 4 offspring, and generate 2 random chromosomes
        new_population = np.vstack((parents, offspring))
        random_chromosomes = np.random.randint(0, 10, size=(2, self.chromosome_length))
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
        """Plot the best fitness over iterations."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(self.fitness_history, label="Best Fitness")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Fitness")
        ax.set_title("Best Fitness Over Iterations")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

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