"""
P-Bit Swarm - Module 7
Probabilistic Computing using P-Bits and Hopfield Networks.

This module implements probabilistic bits (p-bits) that fluctuate between
0 and 1 based on thermal fluctuations, useful for solving optimization
problems via energy minimization.
"""

import numpy as np
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


@dataclass
class PBitConfig:
    """Configuration for a P-Bit system."""
    n_bits: int
    temperature: float = 1.0
    coupling_strength: float = 1.0


class PBit:
    """
    A Probabilistic Bit (P-Bit) that fluctuates between 0 and 1.
    
    P-Bits are the fundamental units of probabilistic computing.
    Unlike classical bits (0 or 1) or quantum bits (superposition),
    P-Bits fluctuate stochastically based on their input and temperature.
    """
    
    def __init__(self, temperature: float = 1.0):
        """
        Initialize a P-Bit.
        
        Parameters
        ----------
        temperature : float
            Thermal temperature controlling fluctuation rate
        """
        self.temperature = temperature
        self.state = 0
        self.input = 0.0
    
    def sigmoid(self, x: float) -> float:
        """
        Sigmoid activation function.
        
        Parameters
        ----------
        x : float
            Input value
            
        Returns
        -------
        float
            Probability of being in state 1
        """
        return 1 / (1 + np.exp(-x / self.temperature))
    
    def update(self, input_signal: float) -> int:
        """
        Update the P-Bit state based on input signal.
        
        Parameters
        ----------
        input_signal : float
            Input signal (weighted sum of connections)
            
        Returns
        -------
        int
            New state (0 or 1)
        """
        self.input = input_signal
        prob_1 = self.sigmoid(input_signal)
        
        if np.random.random() < prob_1:
            self.state = 1
        else:
            self.state = 0
        
        return self.state
    
    def get_magnetization(self) -> float:
        """
        Get the expectation value of the bit (magnetization).
        
        Returns
        -------
        float
            Expected value in [-1, 1]
        """
        prob_1 = self.sigmoid(self.input)
        return 2 * prob_1 - 1


class PBitNetwork:
    """
    A network of P-Bits for solving optimization problems.
    
    This implements a Hopfield-like network where P-Bits interact
    through weighted connections to minimize an energy function.
    """
    
    def __init__(self, n_bits: int, temperature: float = 1.0):
        """
        Initialize a P-Bit network.
        
        Parameters
        ----------
        n_bits : int
            Number of P-Bits in the network
        temperature : float
            Initial temperature for all P-Bits
        """
        self.n_bits = n_bits
        self.pbits = [PBit(temperature) for _ in range(n_bits)]
        self.weights = np.zeros((n_bits, n_bits))
        self.biases = np.zeros(n_bits)
        self.temperature = temperature
        self.energy_history = []
        self.state_history = []
    
    def set_weights(self, weights: np.ndarray, biases: np.ndarray = None):
        """
        Set the connection weights and biases.
        
        Parameters
        ----------
        weights : np.ndarray
            Weight matrix (n_bits x n_bits)
        biases : np.ndarray, optional
            Bias vector (n_bits,)
        """
        self.weights = weights
        if biases is not None:
            self.biases = biases
    
    def set_hopfield_weights(self, patterns: List[np.ndarray]):
        """
        Set weights using Hebbian learning for Hopfield network.
        
        Parameters
        ----------
        patterns : List[np.ndarray]
            List of patterns to store (each pattern is a binary array)
        """
        n = self.n_bits
        
        # Convert {0,1} to {-1,1}
        converted_patterns = []
        for p in patterns:
            converted = 2 * p - 1
            converted_patterns.append(converted)
        
        # Hebbian learning rule
        self.weights = np.zeros((n, n))
        for pattern in converted_patterns:
            self.weights += np.outer(pattern, pattern)
        
        # Zero diagonal (no self-connections)
        np.fill_diagonal(self.weights, 0)
        
        # Normalize
        self.weights /= len(patterns)
    
    def calculate_energy(self, state: np.ndarray) -> float:
        """
        Calculate the energy of a state.
        
        Parameters
        ----------
        state : np.ndarray
            Binary state vector
            
        Returns
        -------
        float
            Energy of the state
        """
        # Convert {0,1} to {-1,1}
        s = 2 * state - 1
        
        # Hopfield energy: E = -0.5 * s^T W s - b^T s
        energy = -0.5 * np.dot(s, np.dot(self.weights, s))
        energy -= np.dot(self.biases, s)
        
        return energy
    
    def update_random(self) -> np.ndarray:
        """
        Update a random P-Bit.
        
        Returns
        -------
        np.ndarray
            Current state of the network
        """
        # Select random P-Bit
        i = np.random.randint(self.n_bits)
        
        # Calculate input from other P-Bits
        current_state = np.array([p.state for p in self.pbits])
        input_signal = np.dot(self.weights[i], current_state) + self.biases[i]
        
        # Update the selected P-Bit
        self.pbits[i].update(input_signal)
        
        return np.array([p.state for p in self.pbits])
    
    def update_all(self, sequential: bool = True) -> np.ndarray:
        """
        Update all P-Bits.
        
        Parameters
        ----------
        sequential : bool
            If True, update one at a time; if False, update in parallel
            
        Returns
        -------
        np.ndarray
            Current state of the network
        """
        if sequential:
            for i in range(self.n_bits):
                current_state = np.array([p.state for p in self.pbits])
                input_signal = np.dot(self.weights[i], current_state) + self.biases[i]
                self.pbits[i].update(input_signal)
        else:
            # Parallel update
            current_state = np.array([p.state for p in self.pbits])
            inputs = np.dot(self.weights, current_state) + self.biases
            for i, pbit in enumerate(self.pbits):
                pbit.update(inputs[i])
        
        return np.array([p.state for p in self.pbits])
    
    def run(self, n_steps: int, record: bool = True) -> np.ndarray:
        """
        Run the network for a number of steps.
        
        Parameters
        ----------
        n_steps : int
            Number of update steps
        record : bool
            Whether to record history
            
        Returns
        -------
        np.ndarray
            Final state
        """
        if record:
            self.energy_history = []
            self.state_history = []
        
        for _ in range(n_steps):
            state = self.update_random()
            
            if record:
                energy = self.calculate_energy(state)
                self.energy_history.append(energy)
                self.state_history.append(state.copy())
        
        return state
    
    def anneal(self, n_steps: int, initial_temp: float, final_temp: float,
               record: bool = True) -> np.ndarray:
        """
        Run simulated annealing on the network.
        
        Parameters
        ----------
        n_steps : int
            Number of update steps
        initial_temp : float
            Starting temperature
        final_temp : float
            Final temperature
        record : bool
            Whether to record history
            
        Returns
        -------
        np.ndarray
            Final state
        """
        if record:
            self.energy_history = []
            self.state_history = []
        
        for step in range(n_steps):
            # Linear temperature schedule
            t = step / n_steps
            temp = initial_temp + t * (final_temp - initial_temp)
            
            # Update all P-Bit temperatures
            for pbit in self.pbits:
                pbit.temperature = temp
            
            state = self.update_random()
            
            if record:
                energy = self.calculate_energy(state)
                self.energy_history.append(energy)
                self.state_history.append(state.copy())
        
        return state
    
    def get_state(self) -> np.ndarray:
        """
        Get the current state of the network.
        
        Returns
        -------
        np.ndarray
            Current binary state vector
        """
        return np.array([p.state for p in self.pbits])
    
    def set_state(self, state: np.ndarray):
        """
        Set the state of the network.
        
        Parameters
        ----------
        state : np.ndarray
            Binary state vector
        """
        for i, s in enumerate(state):
            self.pbits[i].state = int(s)
    
    def plot_energy_history(self):
        """Plot the energy history."""
        if not self.energy_history:
            print("No energy history recorded. Run with record=True first.")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.energy_history)
        plt.xlabel('Step')
        plt.ylabel('Energy')
        plt.title('P-Bit Network Energy Evolution')
        plt.grid(True, alpha=0.3)
        plt.show()


class OptimizationProblem:
    """
    Base class for optimization problems to solve with P-Bits.
    """
    
    def __init__(self, n_bits: int):
        """
        Initialize the optimization problem.
        
        Parameters
        ----------
        n_bits : int
            Number of bits needed to represent the problem
        """
        self.n_bits = n_bits
    
    def get_weights_and_biases(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the weight matrix and bias vector for the problem.
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Weights and biases
        """
        raise NotImplementedError
    
    def evaluate_solution(self, state: np.ndarray) -> float:
        """
        Evaluate the quality of a solution.
        
        Parameters
        ----------
        state : np.ndarray
            Binary state vector
            
        Returns
        -------
        float
            Solution quality (lower is better for minimization)
        """
        raise NotImplementedError


class MaxCutProblem(OptimizationProblem):
    """
    Maximum Cut problem: partition graph nodes to maximize cut weight.
    """
    
    def __init__(self, adjacency_matrix: np.ndarray):
        """
        Initialize MaxCut problem.
        
        Parameters
        ----------
        adjacency_matrix : np.ndarray
            Graph adjacency matrix
        """
        n_bits = adjacency_matrix.shape[0]
        super().__init__(n_bits)
        self.adjacency = adjacency_matrix
    
    def get_weights_and_biases(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get weights for MaxCut (Ising formulation)."""
        # For MaxCut: J_ij = -A_ij (want to minimize -sum A_ij * s_i * s_j)
        weights = -self.adjacency.copy()
        np.fill_diagonal(weights, 0)
        biases = np.zeros(self.n_bits)
        return weights, biases
    
    def evaluate_solution(self, state: np.ndarray) -> float:
        """Calculate cut weight (higher is better)."""
        s = 2 * state - 1  # Convert to {-1, 1}
        cut_value = 0
        for i in range(self.n_bits):
            for j in range(i + 1, self.n_bits):
                if s[i] != s[j]:  # Different partitions
                    cut_value += self.adjacency[i, j]
        return cut_value


class NumberPartitioningProblem(OptimizationProblem):
    """
    Number Partitioning: divide numbers into two sets with equal sums.
    """
    
    def __init__(self, numbers: List[int]):
        """
        Initialize number partitioning problem.
        
        Parameters
        ----------
        numbers : List[int]
            List of numbers to partition
        """
        self.numbers = np.array(numbers)
        n_bits = len(numbers)
        super().__init__(n_bits)
    
    def get_weights_and_biases(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get weights for number partitioning."""
        # We want to minimize (sum_i n_i * s_i)^2
        # This gives J_ij = n_i * n_j and h_i = 0
        weights = np.outer(self.numbers, self.numbers)
        np.fill_diagonal(weights, 0)
        biases = np.zeros(self.n_bits)
        return weights, biases
    
    def evaluate_solution(self, state: np.ndarray) -> float:
        """Calculate partition difference (lower is better)."""
        set_a = self.numbers[state == 1]
        set_b = self.numbers[state == 0]
        return abs(np.sum(set_a) - np.sum(set_b))


def demo_pbit_network():
    """
    Demonstrate P-Bit network for optimization.
    """
    print("=" * 60)
    print("P-BIT PROBABILISTIC COMPUTING DEMONSTRATION")
    print("=" * 60)
    
    # Example 1: Hopfield Network for Pattern Recognition
    print("\n1. Hopfield Network - Pattern Recognition")
    print("-" * 40)
    
    # Create patterns to store
    n_bits = 16
    patterns = [
        np.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]),  # Block pattern
        np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]),  # Striped pattern
    ]
    
    network = PBitNetwork(n_bits, temperature=0.5)
    network.set_hopfield_weights(patterns)
    
    # Corrupt a pattern and try to recover
    corrupted = patterns[0].copy()
    corrupted[:4] = 1 - corrupted[:4]  # Flip first 4 bits
    
    print(f"Original pattern:  {patterns[0]}")
    print(f"Corrupted pattern: {corrupted}")
    
    network.set_state(corrupted)
    recovered = network.run(1000)
    
    print(f"Recovered pattern: {recovered}")
    print(f"Match original: {np.array_equal(recovered, patterns[0])}")
    
    # Example 2: Number Partitioning
    print("\n2. Number Partitioning Problem")
    print("-" * 40)
    
    numbers = [4, 2, 7, 1, 3, 6, 5]
    print(f"Numbers to partition: {numbers}")
    
    problem = NumberPartitioningProblem(numbers)
    weights, biases = problem.get_weights_and_biases()
    
    network = PBitNetwork(len(numbers), temperature=2.0)
    network.set_weights(weights, biases)
    
    # Run with simulated annealing
    final_state = network.anneal(2000, initial_temp=2.0, final_temp=0.1)
    
    set_a = [numbers[i] for i in range(len(numbers)) if final_state[i] == 1]
    set_b = [numbers[i] for i in range(len(numbers)) if final_state[i] == 0]
    
    print(f"Set A: {set_a} (sum = {sum(set_a)})")
    print(f"Set B: {set_b} (sum = {sum(set_b)})")
    print(f"Difference: {abs(sum(set_a) - sum(set_b))}")
    
    return network


def demo_energy_landscape():
    """
    Demonstrate energy landscape visualization.
    """
    print("\n" + "=" * 60)
    print("ENERGY LANDSCAPE VISUALIZATION")
    print("=" * 60)
    
    # Create a simple 2-bit system
    network = PBitNetwork(2, temperature=0.5)
    
    # Set weights for a simple energy landscape
    network.weights = np.array([[0, 1], [1, 0]])
    network.biases = np.array([0.5, -0.5])
    
    # Enumerate all states and calculate energies
    states = []
    energies = []
    
    for s0 in [0, 1]:
        for s1 in [0, 1]:
            state = np.array([s0, s1])
            energy = network.calculate_energy(state)
            states.append(state)
            energies.append(energy)
            print(f"State {state}: Energy = {energy:.4f}")
    
    # Find minimum energy state
    min_idx = np.argmin(energies)
    print(f"\nMinimum energy state: {states[min_idx]} with energy {energies[min_idx]:.4f}")
    
    return network


if __name__ == "__main__":
    demo_pbit_network()
    demo_energy_landscape()
