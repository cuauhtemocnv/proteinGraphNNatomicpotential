# Graph Neural Network for Energy Prediction

This repository contains a Python implementation of a Graph Neural Network (GNN) designed to predict the energies of atoms in a molecular system. The code utilizes PyTorch for model definition and training, and the ASE library for handling atomic data.

## Functions and Classes

- `build_graph`: Builds a graph representation of a molecule from an `Atoms` object. It computes pairwise distances and creates feature vectors and an adjacency matrix.

- `FlexibleGNN`: A GNN class that defines the model architecture with a sequence of linear layers and ReLU activation functions, followed by an output layer for energy prediction.

- `compute_energy`: Calculates energies of atoms within a cutoff distance based on atomic interactions.

## Dataset Construction

The script generates a dataset of molecular configurations by perturbing the positions of atoms in an initial configuration and computing the corresponding energies.

## Training

The model is trained using the Mean Squared Error (MSE) loss function, an Adam optimizer, and a learning rate scheduler that adjusts the learning rate during training.

## Usage

To test the function and train the model, run the script as follows:

```python
python gnn_energy_prediction.py
