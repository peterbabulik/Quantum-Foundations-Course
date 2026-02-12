"""
Unification Engine - Module 3
A custom 3D engine to visualize Group Theory (SU2) and rotations.

This module provides tools for visualizing quantum state transformations
on the Bloch sphere, demonstrating the geometric nature of quantum logic.
"""

import numpy as np
from typing import Tuple, List, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class BlochSphere:
    """
    A class for visualizing quantum states on the Bloch sphere.
    
    The Bloch sphere is a geometrical representation of the pure state
    space of a two-level quantum mechanical system (qubit).
    """
    
    def __init__(self):
        """Initialize the Bloch sphere visualization."""
        self.fig = None
        self._setup_sphere()
    
    def _setup_sphere(self):
        """Set up the base Bloch sphere with axes."""
        self.fig = go.Figure()
        
        # Draw the sphere
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        
        self.fig.add_trace(go.Surface(
            x=x, y=y, z=z,
            opacity=0.1,
            showscale=False,
            colorscale='Blues'
        ))
        
        # Add axes
        axis_length = 1.3
        axes = [
            ([-axis_length, axis_length], [0, 0], [0, 0], 'X', 'red'),
            ([0, 0], [-axis_length, axis_length], [0, 0], 'Y', 'green'),
            ([0, 0], [0, 0], [-axis_length, axis_length], 'Z', 'blue'),
        ]
        
        for x_line, y_line, z_line, label, color in axes:
            self.fig.add_trace(go.Scatter3d(
                x=x_line, y=y_line, z=z_line,
                mode='lines+text',
                line=dict(color=color, width=2),
                name=label
            ))
    
    def state_to_bloch(self, theta: float, phi: float) -> Tuple[float, float, float]:
        """
        Convert quantum state angles to Bloch sphere coordinates.
        
        Parameters
        ----------
        theta : float
            Polar angle (0 to π)
        phi : float
            Azimuthal angle (0 to 2π)
            
        Returns
        -------
        Tuple[float, float, float]
            Cartesian coordinates (x, y, z) on the Bloch sphere
        """
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        return x, y, z
    
    def add_state(self, theta: float, phi: float, label: str = "|ψ⟩", 
                  color: str = 'yellow', size: float = 8):
        """
        Add a quantum state to the Bloch sphere visualization.
        
        Parameters
        ----------
        theta : float
            Polar angle in radians
        phi : float
            Azimuthal angle in radians
        label : str
            Label for the state
        color : str
            Color of the state marker
        size : float
            Size of the marker
        """
        x, y, z = self.state_to_bloch(theta, phi)
        
        self.fig.add_trace(go.Scatter3d(
            x=[x], y=[y], z=[z],
            mode='markers+text',
            marker=dict(size=size, color=color),
            text=[label],
            textposition='top center',
            name=label
        ))
        
        # Add vector from origin
        self.fig.add_trace(go.Scatter3d(
            x=[0, x], y=[0, y], z=[0, z],
            mode='lines',
            line=dict(color=color, width=3),
            showlegend=False
        ))
    
    def add_rotation(self, start_theta: float, start_phi: float,
                     rotation_axis: str, angle: float, steps: int = 20):
        """
        Visualize a rotation on the Bloch sphere.
        
        Parameters
        ----------
        start_theta : float
            Initial polar angle
        start_phi : float
            Initial azimuthal angle
        rotation_axis : str
            Axis of rotation ('X', 'Y', or 'Z')
        angle : float
            Rotation angle in radians
        steps : int
            Number of steps in the rotation animation
        """
        # Generate rotation path
        x_path, y_path, z_path = [], [], []
        
        for t in np.linspace(0, angle, steps):
            if rotation_axis.upper() == 'X':
                theta, phi = self._rotate_x(start_theta, start_phi, t)
            elif rotation_axis.upper() == 'Y':
                theta, phi = self._rotate_y(start_theta, start_phi, t)
            elif rotation_axis.upper() == 'Z':
                theta, phi = self._rotate_z(start_theta, start_phi, t)
            else:
                raise ValueError(f"Unknown rotation axis: {rotation_axis}")
            
            x, y, z = self.state_to_bloch(theta, phi)
            x_path.append(x)
            y_path.append(y)
            z_path.append(z)
        
        self.fig.add_trace(go.Scatter3d(
            x=x_path, y=y_path, z=z_path,
            mode='lines',
            line=dict(color='orange', width=4),
            name=f'R{rotation_axis}({angle:.2f})'
        ))
    
    def _rotate_x(self, theta: float, phi: float, angle: float) -> Tuple[float, float]:
        """Rotate state around X-axis."""
        # Convert to Cartesian, rotate, convert back
        x, y, z = self.state_to_bloch(theta, phi)
        y_new = y * np.cos(angle) - z * np.sin(angle)
        z_new = y * np.sin(angle) + z * np.cos(angle)
        
        theta_new = np.arccos(np.clip(z_new, -1, 1))
        phi_new = np.arctan2(y_new, x)
        return theta_new, phi_new
    
    def _rotate_y(self, theta: float, phi: float, angle: float) -> Tuple[float, float]:
        """Rotate state around Y-axis."""
        x, y, z = self.state_to_bloch(theta, phi)
        x_new = x * np.cos(angle) + z * np.sin(angle)
        z_new = -x * np.sin(angle) + z * np.cos(angle)
        
        theta_new = np.arccos(np.clip(z_new, -1, 1))
        phi_new = np.arctan2(y, x_new)
        return theta_new, phi_new
    
    def _rotate_z(self, theta: float, phi: float, angle: float) -> Tuple[float, float]:
        """Rotate state around Z-axis."""
        return theta, phi + angle
    
    def show(self, title: str = "Bloch Sphere"):
        """
        Display the Bloch sphere visualization.
        
        Parameters
        ----------
        title : str
            Title for the plot
        """
        self.fig.update_layout(
            title=title,
            scene=dict(
                aspectmode='cube',
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            ),
            showlegend=True
        )
        self.fig.show()


class UnificationEngine:
    """
    The Unification Engine demonstrates that Classical Logic is just 
    Quantum Logic restricted to 90-degree rotations.
    
    This engine visualizes how quantum gates generalize classical logic
    operations through the geometry of SU(2).
    """
    
    def __init__(self):
        """Initialize the Unification Engine."""
        self.bloch = BlochSphere()
        self.gate_history = []
    
    def pauli_x(self) -> np.ndarray:
        """
        Pauli-X gate (quantum NOT gate).
        
        Returns
        -------
        np.ndarray
            The Pauli-X matrix
        """
        return np.array([[0, 1], [1, 0]], dtype=complex)
    
    def pauli_y(self) -> np.ndarray:
        """
        Pauli-Y gate.
        
        Returns
        -------
        np.ndarray
            The Pauli-Y matrix
        """
        return np.array([[0, -1j], [1j, 0]], dtype=complex)
    
    def pauli_z(self) -> np.ndarray:
        """
        Pauli-Z gate (phase flip).
        
        Returns
        -------
        np.ndarray
            The Pauli-Z matrix
        """
        return np.array([[1, 0], [0, -1]], dtype=complex)
    
    def hadamard(self) -> np.ndarray:
        """
        Hadamard gate - creates superposition.
        
        Returns
        -------
        np.ndarray
            The Hadamard matrix
        """
        return np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    
    def rotation_matrix(self, axis: str, angle: float) -> np.ndarray:
        """
        Generate a rotation matrix for a given axis and angle.
        
        Parameters
        ----------
        axis : str
            Rotation axis ('X', 'Y', or 'Z')
        angle : float
            Rotation angle in radians
            
        Returns
        -------
        np.ndarray
            2x2 unitary rotation matrix
        """
        c, s = np.cos(angle / 2), np.sin(angle / 2)
        
        if axis.upper() == 'X':
            return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)
        elif axis.upper() == 'Y':
            return np.array([[c, -s], [s, c]], dtype=complex)
        elif axis.upper() == 'Z':
            return np.array([[np.exp(-1j * angle / 2), 0], 
                           [0, np.exp(1j * angle / 2)]], dtype=complex)
        else:
            raise ValueError(f"Unknown axis: {axis}")
    
    def apply_gate(self, state: np.ndarray, gate: np.ndarray) -> np.ndarray:
        """
        Apply a quantum gate to a state.
        
        Parameters
        ----------
        state : np.ndarray
            Initial quantum state vector
        gate : np.ndarray
            Quantum gate matrix
            
        Returns
        -------
        np.ndarray
            Resulting quantum state
        """
        return gate @ state
    
    def state_to_angles(self, state: np.ndarray) -> Tuple[float, float]:
        """
        Convert a quantum state vector to Bloch sphere angles.
        
        Parameters
        ----------
        state : np.ndarray
            Quantum state vector [alpha, beta]
            
        Returns
        -------
        Tuple[float, float]
            Theta and phi angles
        """
        # Normalize state
        state = state / np.linalg.norm(state)
        
        # Extract angles
        theta = 2 * np.arccos(np.abs(state[0]))
        phi = np.angle(state[1]) - np.angle(state[0])
        
        return theta, phi
    
    def demonstrate_classical_as_quantum(self):
        """
        Demonstrate that classical NOT is a 180° rotation (quantum X gate).
        
        This shows the fundamental unification: classical bits are just
        qubits restricted to the poles of the Bloch sphere.
        """
        print("=" * 60)
        print("DEMONSTRATION: Classical Logic as Quantum Geometry")
        print("=" * 60)
        
        # Classical |0⟩ state (north pole)
        state_0 = np.array([1, 0], dtype=complex)
        
        # Classical |1⟩ state (south pole)
        state_1 = np.array([0, 1], dtype=complex)
        
        print("\n1. Classical NOT operation:")
        print(f"   |0⟩ = {state_0}")
        print(f"   X|0⟩ = {self.pauli_x() @ state_0} = |1⟩")
        print("   → Classical NOT is a 180° rotation around X-axis")
        
        print("\n2. Quantum Superposition:")
        print(f"   H|0⟩ = {self.hadamard() @ state_0}")
        print("   → Hadamard creates superposition (equator state)")
        
        print("\n3. The Unification:")
        print("   Classical bits: Only poles (θ = 0 or π)")
        print("   Quantum bits: Entire sphere surface")
        print("   Classical logic: 180° rotations")
        print("   Quantum logic: Arbitrary rotations")
        
        return self.bloch
    
    def visualize_gate_sequence(self, gates: List[Tuple[str, float]], 
                                 initial_state: np.ndarray = None):
        """
        Visualize a sequence of quantum gates on the Bloch sphere.
        
        Parameters
        ----------
        gates : List[Tuple[str, float]]
            List of (axis, angle) tuples representing gates
        initial_state : np.ndarray, optional
            Initial quantum state (defaults to |0⟩)
        """
        if initial_state is None:
            initial_state = np.array([1, 0], dtype=complex)
        
        state = initial_state.copy()
        theta, phi = self.state_to_angles(state)
        
        self.bloch.add_state(theta, phi, label="|0⟩", color='green')
        
        for i, (axis, angle) in enumerate(gates):
            gate = self.rotation_matrix(axis, angle)
            state = self.apply_gate(state, gate)
            theta, phi = self.state_to_angles(state)
            
            label = f"Step {i+1}"
            self.bloch.add_state(theta, phi, label=label, color='orange')
        
        self.bloch.show("Gate Sequence Visualization")


def demo_continuum_logic():
    """
    Demonstrate the continuum of logic from classical to quantum.
    """
    engine = UnificationEngine()
    
    print("\n" + "=" * 60)
    print("CONTINUUM LOGIC DEMONSTRATION")
    print("=" * 60)
    
    # Create visualization
    bloch = BlochSphere()
    
    # Add classical states
    bloch.add_state(0, 0, label="|0⟩ (Classical 0)", color='green')
    bloch.add_state(np.pi, 0, label="|1⟩ (Classical 1)", color='red')
    
    # Add quantum superposition states
    bloch.add_state(np.pi/2, 0, label="|+⟩", color='blue')
    bloch.add_state(np.pi/2, np.pi/2, label="|+i⟩", color='purple')
    bloch.add_state(np.pi/2, np.pi, label="|-⟩", color='cyan')
    bloch.add_state(np.pi/2, 3*np.pi/2, label="|-i⟩", color='magenta')
    
    # Add a rotation path
    bloch.add_rotation(0, 0, 'Y', np.pi, steps=30)
    
    bloch.show("Continuum of Logic: Classical to Quantum")
    
    return engine.demonstrate_classical_as_quantum()


if __name__ == "__main__":
    demo_continuum_logic()
