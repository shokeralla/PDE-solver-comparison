# PDE Solver Comparison

## Overview
This repository provides a systematic comparison between numerical methods (Finite Element Method - FEM) and probabilistic methods (Hamiltonian Monte Carlo - HMC and Multilevel Monte Carlo - MLMC) for solving nonlinear partial differential equations (PDEs). The study emphasizes accuracy, computational efficiency, and uncertainty quantification.

## Features
- Implementation of adaptive FEM for solving nonlinear PDEs.
- Probabilistic modeling using Hamiltonian Monte Carlo and Multilevel Monte Carlo.
- Comprehensive benchmarking on Burgers’ and Navier-Stokes equations.
- Visualization of solution behaviors and convergence properties.

## Methods Compared
- **Numerical Methods**: FEM with adaptive mesh refinement, Trust-region-reflective algorithms.
- **Probabilistic Methods**: Bayesian inference (HMC), MLMC for uncertainty quantification.

## Results Highlights
- FEM demonstrated high accuracy (MSE: 3.2×10⁻⁵) and efficiency (runtime: 0.5 hrs).
- HMC effectively quantified uncertainty with robust posterior distributions.

## Requirements
- Python 3.9+
- FEniCS library
- Pyro or PyMC for Bayesian inference
- NumPy and Matplotlib for analysis and visualization

## Usage
- Clone the repository.
- Install dependencies listed in requirements.
- Run scripts to reproduce simulations and analyses.

## Contributions
Feel free to submit issues or pull requests to enhance this repository.

## License
This project is licensed under the MIT License.

