# Quantum Mechanics: From Equations to 133 Qubits
### A Pythonic Approach to the Physics of Information

**Author:** Peter Babulik  
**Status:** Live Course  
**Repository:** [github.com/peterbabulik/Quantum-Foundations-Course](https://github.com/peterbabulik/Quantum-Foundations-Course)

---

## 🌌 The Philosophy
Most Quantum Mechanics courses start with complex notation ($\langle \psi | \hat{H} | \psi \rangle$).  
**This course starts with `import numpy`.**

We believe that if you can program it, you understand it. This repository takes you on a computational journey from the First Principles of Physics (Planck & Bohr) to running neural networks on utility-scale Quantum Hardware (IBM Heron/Torino). 

We do not just run code; we build engines. We build the `UnificationEngine` to visualize logic, the `QuantumNativeEntropy` engine to generate randomness, and the `LargeScaleQuantumAI` class to classify data on 50+ qubits.

---

## 🚀 Course Roadmap & Directory Structure

### **Module 1: The Foundations (History via Code)**
*Before we could build the Qubit, we had to discover that nature is pixelated.*
*   📂 **`01_foundations/`**
    *   [`01_planck_radiation.ipynb`](https://github.com/peterbabulik/Quantum-Foundations-Course/blob/main/course_content/01_foundations/01_planck_radiation.ipynb): Simulating the **Ultraviolet Catastrophe**. We code Planck's Law vs. Rayleigh-Jeans to prove energy is discrete.
    *   [`02_bohr_atom.ipynb`](https://github.com/peterbabulik/Quantum-Foundations-Course/blob/main/course_content/01_foundations/02_bohr_atom.ipynb): A Python implementation of the **Rydberg Formula**. We calculate the exact color of photon emissions ($n=3 \to n=2$) in the Hydrogen atom.

### **Module 2: Wave Mechanics (The Math)**
*How continuous waves become discrete integers.*
*   📂 **`02_wave_mechanics/`**
    *   [`01_schrodinger_box.ipynb`](https://github.com/peterbabulik/Quantum-Foundations-Course/blob/main/course_content/02_wave_mechanics/01_schrodinger_box.ipynb): Solving the **Time-Independent Schrödinger Equation** for a particle in a box. Visualizing how confinement creates Basis States ($|0\rangle, |1\rangle$).
    *   [`02_entangled_system.ipynb`](https://github.com/peterbabulik/Quantum-Foundations-Course/blob/main/course_content/02_wave_mechanics/02_entangled_system.ipynb): A custom class simulating **Bell States**. We demonstrate that Entanglement is not magic, but a restriction of the 4D state space ($|00\rangle + |11\rangle$).

### **Module 3: The Unified Qubit (Logic & Geometry)**
*Demystifying the "Magic" by unifying Group Theory, Geometry, and Information.*
*   📂 **`03_unified_qubit/`**
    *   [`01_continuum_logic.ipynb`](https://github.com/peterbabulik/Quantum-Foundations-Course/blob/main/course_content/03_unified_qubit/01_continuum_logic.ipynb): **Differentiable Gates.** We prove the "Artificial Wall" between Classical and Quantum logic is false by training a continuous vector to act as a NOT gate.
    *   [`02_visualizing_bloch.ipynb`](https://github.com/peterbabulik/Quantum-Foundations-Course/blob/main/course_content/03_unified_qubit/02_visualizing_bloch.ipynb): The **Unification Engine**. A 3D Plotly simulation of the Bloch Sphere driven by SU(2) Group Theory rotations and Von Neumann Entropy calculations.

### **Module 4: Engineering Reality (Hardware & Scale)**
*Moving from Simulator to the IBM Heron Processor.*
*   📂 **`04_real_hardware/`**
    *   [`01_hilbert_scale.ipynb`](https://github.com/peterbabulik/Quantum-Foundations-Course/blob/main/course_content/04_real_hardware/01_hilbert_scale.ipynb): Visualizing the terrifying exponential growth ($2^N$) of the State Space. We benchmark data capacity from 1 to 20 qubits.
    *   [`02_zne_extrapolation.ipynb`](https://github.com/peterbabulik/Quantum-Foundations-Course/blob/main/course_content/04_real_hardware/02_zne_extrapolation.ipynb): **Zero Noise Extrapolation (ZNE).** We implement a custom Hyperbolic Tangent (Tanh) decay model to recover clean signals from noisy hardware data on the IBM Cloud.

### **Module 5: The Intelligent Future (Quantum AI)**
*Where Quantum Mechanics meets Machine Learning.*
*   📂 **`05_quantum_ai/`**
    *   [`01_hybrid_training.ipynb`](https://github.com/peterbabulik/Quantum-Foundations-Course/blob/main/course_content/05_quantum_ai/01_hybrid_training.ipynb): Building a **Quantum Neuron** using **PennyLane**. We use classical Gradient Descent to optimize quantum parameters (rotation angles).
    *   [`02_50_qubit_inference.ipynb`](https://github.com/peterbabulik/Quantum-Foundations-Course/blob/main/course_content/05_quantum_ai/02_50_qubit_inference.ipynb): **Utility-Scale Inference.** Deploying a Variational Quantum Classifier (VQC) on 50 physical qubits using `qiskit-ibm-runtime` and the `EfficientSU2` ansatz.

### **Module 6: Statistical Verification (True Randomness)**
*Proving that God DOES play dice.*
*   📂 **`06_entropy_verification/`**
    *   [`01_dieharder_tests.ipynb`](https://github.com/peterbabulik/Quantum-Foundations-Course/blob/main/course_content/06_entropy_verification/01_dieharder_tests.ipynb): The **Quantum Native Entropy Engine**. We generate raw bits from qubit superposition and validate them against the **NIST/Dieharder** statistical test battery (Monobit, Runs, Serial tests).

### **Module 7: Probabilistic Computing (Thermodynamics)**
*Computing with Energy Landscapes (P-Bits).*
*   📂 **`07_probabilistic_computing/`**
    *   [`01_energy_landscapes.ipynb`](https://github.com/peterbabulik/Quantum-Foundations-Course/blob/main/course_content/07_probabilistic_computing/01_energy_landscapes.ipynb): Using **Simulated Annealing** and **Hopfield Networks**. We solve NP-Hard optimization problems (Number Partitioning) and restore corrupted memories by minimizing the system's energy.

---

## 🛠 Tech Stack & Dependencies

*   **Core Math:** `numpy`, `scipy`
*   **Visualization:** `matplotlib`, `plotly`
*   **Quantum SDKs:** `qiskit`, `pennylane`, `qiskit-ibm-runtime`
*   **Data:** `pandas`

## 💻 Installation

To run this course locally:

```bash
# 1. Clone the repository
git clone https://github.com/peterbabulik/Quantum-Foundations-Course.git
cd Quantum-Foundations-Course

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the Course
jupyter notebook
```

## 🔑 Hardware Access
Modules **04** and **05** require access to real quantum hardware.
1.  Sign up at [quantum.ibm.com](https://quantum.ibm.com).
2.  Copy your API Key.
3.  Paste it into the `API_KEY` variable in the respective notebooks.

---

*"God does not play dice with the universe." — Albert Einstein*  
*"Stop telling God what to do." — Niels Bohr*  
*"Import numpy as np." — This Course*
